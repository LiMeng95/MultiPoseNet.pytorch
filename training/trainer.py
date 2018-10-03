from __future__ import print_function

import os
import sys
import datetime
import numpy as np
from collections import OrderedDict
import shutil

import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler
from torch.optim.optimizer import Optimizer

from lib.utils.log import logger
from lib.utils.timer import Timer
from lib.utils.path import mkdir
import lib.utils.meter as meter_utils
import network.net_utils as net_utils
from datasets.data_parallel import ListDataParallel


def get_learning_rates(optimizer):
    lrs = [pg['lr'] for pg in optimizer.param_groups]
    lrs = np.asarray(lrs, dtype=np.float)
    return lrs


def default_visualization_fn(writer, step, log_dict):
    """
    Visualization with tensorboard
    :type writer: SummaryWriter
    :type step: int
    :type log_dict: dict
    :return:
    """
    for k, v in log_dict.items():
        if isinstance(v, (float, int)):
            writer.add_scalar(k, v, step)
        elif isinstance(v, np.ndarray):
            writer.add_image(k, v, step)


class TrainParams(object):
    # required params
    exp_name = 'experiment_name'
    subnet_name = 'keypoint_subnet'
    batch_size = 32
    max_epoch = 30
    optimizer = None

    # learning rate scheduler
    lr_scheduler = None         # should be an instance of ReduceLROnPlateau or _LRScheduler
    max_grad_norm = np.inf

    # params based on your local env
    gpus = [0]
    save_dir = None             # default `save_dir` is `outputs/{exp_name}`

    # loading existing checkpoint
    ckpt = None                 # path to the ckpt file, will load the last ckpt in the `save_dir` if `None`
    re_init = False             # ignore ckpt if `True`
    zero_epoch = False          # force `last_epoch` to zero
    ignore_opt_state = False    # ignore the saved optimizer states

    # saving checkpoints
    save_freq_epoch = 1             # save one ckpt per `save_freq_epoch` epochs
    save_freq_step = sys.maxsize    # save one ckpt per `save_freq_setp` steps, default value is inf
    save_nckpt_max = sys.maxsize    # max number of saved ckpts

    # validation during training
    val_freq = 500              # run validation per `val_freq` steps
    val_nbatch = 10             # number of batches to be validated
    val_nbatch_end_epoch = 200  # max number of batches to be validated after each epoch

    # visualization
    print_freq = 20             # print log per `print_freq` steps
    use_tensorboard = False     # use tensorboardX if True
    visualization_fn = None     # custom function to handle `log_dict`, default value is `default_visualization_fn`

    def update(self, params_dict):
        state_dict = self.state_dict()
        for k, v in params_dict.items():
            if k in state_dict or hasattr(self, k):
                setattr(self, k, v)
            else:
                logger.warning('Unknown option: {}: {}'.format(k, v))

    def state_dict(self):
        state_dict = OrderedDict()
        for k in TrainParams.__dict__.keys():
            if not k.startswith('_'):
                state_dict[k] = getattr(self, k)
        del state_dict['update']
        del state_dict['state_dict']

        return state_dict

    def __str__(self):
        state_dict = self.state_dict()
        text = 'TrainParams {\n'
        for k, v in state_dict.items():
            text += '\t{}: {}\n'.format(k, v)
        text += '}\n'
        return text


class Trainer(object):

    TrainParams = TrainParams

    # hooks
    on_start_epoch_hooks = []
    on_end_epoch_hooks = []

    def __init__(self, model, train_params, batch_processor, train_data, val_data=None):
        assert isinstance(train_params, TrainParams)
        self.params = train_params

        # Data loaders
        self.train_data = train_data
        self.val_data = val_data # sDataLoader.copy(val_data) if isinstance(val_data, DataLoader) else val_data
        # self.val_stream = self.val_data.get_stream() if self.val_data else None

        self.batch_processor = batch_processor
        self.batch_per_epoch = len(self.train_data)

        # set CUDA_VISIBLE_DEVICES=gpus
        gpus = ','.join([str(x) for x in self.params.gpus])
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus
        self.params.gpus = tuple(range(len(self.params.gpus)))
        logger.info('Set CUDA_VISIBLE_DEVICES to {}...'.format(gpus))

        # Optimizer and learning rate
        self.last_epoch = 0
        self.optimizer = self.params.optimizer  # type: Optimizer
        if not isinstance(self.optimizer, Optimizer):
            logger.error('optimizer should be an instance of Optimizer, '
                         'but got {}'.format(type(self.optimizer)))
            raise ValueError
        self.lr_scheduler = self.params.lr_scheduler  # type: ReduceLROnPlateau or _LRScheduler
        if self.lr_scheduler and not isinstance(self.lr_scheduler, (ReduceLROnPlateau, _LRScheduler)):
            logger.error('lr_scheduler should be an instance of _LRScheduler or ReduceLROnPlateau, '
                         'but got {}'.format(type(self.lr_scheduler)))
            raise ValueError
        logger.info('Set lr_scheduler to {}'.format(type(self.lr_scheduler)))

        self.log_values = OrderedDict()
        self.batch_timer = Timer()
        self.data_timer = Timer()

        # load model
        self.model = model
        ckpt = self.params.ckpt
        if not self.params.save_dir:
            self.params.save_dir = os.path.join('outputs', self.params.exp_name)
        mkdir(self.params.save_dir)
        logger.info('Set output dir to {}'.format(self.params.save_dir))
        if ckpt is None:
            # find the last ckpt
            ckpts = [fname for fname in os.listdir(self.params.save_dir) if os.path.splitext(fname)[-1] == '.h5']
            ckpt = os.path.join(
                self.params.save_dir, sorted(ckpts, key=lambda name: int(os.path.splitext(name)[0].split('_')[-1]))[-1]
            ) if len(ckpts) > 0 else None

        if ckpt is not None and not self.params.re_init:
           self._load_ckpt(ckpt)
           logger.info('Load ckpt from {}'.format(ckpt))
        #elif hasattr(self.model, 'init_weight'):
        #    self.model.init_weight()
        #    logger.info("Re-init model weight")

        self.model = ListDataParallel(self.model, device_ids=self.params.gpus)
        self.model = self.model.cuda(self.params.gpus[0])
        self.model.train()
        self.model.module.freeze_bn()


    def train(self):
        best_loss = np.inf
        for epoch in range(self.last_epoch, self.params.max_epoch):
            self.last_epoch += 1
            logger.info('Start training epoch {}'.format(self.last_epoch))

            for fun in self.on_start_epoch_hooks:
                fun(self)

            # adjust learning rate
            if isinstance(self.lr_scheduler, _LRScheduler):
                cur_lrs = get_learning_rates(self.optimizer)
                self.lr_scheduler.step(self.last_epoch)
                logger.info('Set learning rates from {} to {}'.format(cur_lrs, get_learning_rates(self.optimizer)))

            train_loss = self._train_one_epoch()

            for fun in self.on_end_epoch_hooks:
                fun(self)

            # save model
            if (self.last_epoch % self.params.save_freq_epoch == 0) or (self.last_epoch == self.params.max_epoch - 1):
                save_name = 'ckpt_{}.h5'.format(self.last_epoch)
                save_to = os.path.join(self.params.save_dir, save_name)
                self._save_ckpt(save_to)

                # find best model
                if self.params.val_nbatch_end_epoch > 0:
                    val_loss = self._val_one_epoch(self.params.val_nbatch_end_epoch)
                    if val_loss < best_loss:
                        best_file = os.path.join(self.params.save_dir,
                                                 'ckpt_{}_{:.5f}.h5.best'.format(self.last_epoch, val_loss))
                        shutil.copyfile(save_to, best_file)
                        logger.info('Found a better ckpt ({:.5f} -> {:.5f}), '
                                    'saved to {}'.format(best_loss, val_loss, best_file))
                        best_loss = val_loss

                    if isinstance(self.lr_scheduler, ReduceLROnPlateau):
                        self.lr_scheduler.step(val_loss, self.last_epoch)

    def _save_ckpt(self, save_to):
        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        net_utils.save_net(save_to, model, epoch=self.last_epoch,
                           optimizers=[self.optimizer], rm_prev_opt=True, max_n_ckpts=self.params.save_nckpt_max)
        logger.info('Save ckpt to {}'.format(save_to))

    def _load_ckpt(self, ckpt):
        epoch, state_dicts = net_utils.load_net(ckpt, self.model, load_state_dict=True)
        if not self.params.ignore_opt_state and not self.params.zero_epoch and epoch >= 0:
            self.last_epoch = epoch
            logger.info('Set last epoch to {}'.format(self.last_epoch))
            if state_dicts is not None:
                self.optimizer.load_state_dict(state_dicts[0])
                net_utils.set_optimizer_state_devices(self.optimizer.state, self.params.gpus[0])
                logger.info('Load optimizer state from checkpoint, '
                            'new learning rate: {}'.format(get_learning_rates(self.optimizer)))

    def _train_one_epoch(self):
        self.batch_timer.clear()
        self.data_timer.clear()
        self.batch_timer.tic()
        self.data_timer.tic()
        total_loss = meter_utils.AverageValueMeter()
        for step, batch in enumerate(self.train_data):
            inputs, gts, _ = self.batch_processor(self, batch)

            self.data_timer.toc()

            # forward
            output, saved_for_loss = self.model(*inputs)

            loss, saved_for_log = self.model.module.build_loss(saved_for_loss, *gts)

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            total_loss.add(loss.item())

            # clip grad
            if not np.isinf(self.params.max_grad_norm):
                max_norm = nn.utils.clip_grad_norm(self.model.parameters(), self.params.max_grad_norm, float('inf'))
                saved_for_log['max_grad'] = max_norm

            self.optimizer.step(None)

            self._process_log(saved_for_log, self.log_values)
            self.batch_timer.toc()

            # print log
            reset = False

            if step % self.params.print_freq == 0:
                self._print_log(step, self.log_values, title='Training', max_n_batch=self.batch_per_epoch)
                reset = True

            if step % self.params.save_freq_step == 0 and step > 0:
                save_to = os.path.join(self.params.save_dir,
                                       'ckpt_{}.h5.ckpt'.format((self.last_epoch - 1) * self.batch_per_epoch + step))
                self._save_ckpt(save_to)

            if reset:
                self._reset_log(self.log_values)

            self.data_timer.tic()
            self.batch_timer.tic()

        total_loss, std = total_loss.value()
        return total_loss

    def _val_one_epoch(self, n_batch):
        training_mode = self.model.training
        self.model.eval()
        logs = OrderedDict()
        sum_loss = meter_utils.AverageValueMeter()
        logger.info('Val on validation set...')

        self.batch_timer.clear()
        self.data_timer.clear()
        self.batch_timer.tic()
        self.data_timer.tic()
        for step, batch in enumerate(self.val_data):
            self.data_timer.toc()
            if step > n_batch:
                break

            inputs, gts, _ = self.batch_processor(self, batch)
            _, saved_for_loss = self.model(*inputs)
            self.batch_timer.toc()

            loss, saved_for_log = self.model.module.build_loss(saved_for_loss, *gts)
            sum_loss.add(loss.item())
            self._process_log(saved_for_log, logs)

            if step % self.params.print_freq == 0:
                self._print_log(step, logs, 'Validation', max_n_batch=min(n_batch, len(self.val_data)))

            self.data_timer.tic()
            self.batch_timer.tic()

        mean, std = sum_loss.value()
        logger.info('Validation loss: mean: {}, std: {}'.format(mean, std))
        self.model.train(mode=training_mode)
        self.model.module.freeze_bn()
        return mean

    def _process_log(self, src_dict, dest_dict):
        for k, v in src_dict.items():
            if isinstance(v, (int, float)):
                dest_dict.setdefault(k, meter_utils.AverageValueMeter())
                dest_dict[k].add(float(v))
            else:
                dest_dict[k] = v

    def _print_log(self, step, log_values, title='', max_n_batch=None):
        log_str = '{}\n'.format(self.params.exp_name)
        log_str += '{}: epoch {}'.format(title, self.last_epoch)

        if max_n_batch:
            log_str += '[{}/{}], lr: {}'.format(step, max_n_batch, get_learning_rates(self.optimizer))

        i = 0
        # global_step = step + (self.last_epoch - 1) * self.batch_per_epoch
        for k, v in log_values.items():
            if isinstance(v, meter_utils.AverageValueMeter):
                mean, std = v.value()
                log_str += '\n\t{}: {:.10f}'.format(k, mean)
                i += 1

        if max_n_batch:
            # print time
            data_time = self.data_timer.duration + 1e-6
            batch_time = self.batch_timer.duration + 1e-6
            rest_seconds = int((max_n_batch - step) * batch_time)
            log_str += '\n\t({:.2f}/{:.2f}s,' \
                       ' fps:{:.1f}, rest: {})'.format(data_time, batch_time,
                                                       self.params.batch_size / batch_time,
                                                       str(datetime.timedelta(seconds=rest_seconds)))
            self.batch_timer.clear()
            self.data_timer.clear()

        logger.info(log_str)

    def _reset_log(self, log_values):
        for k, v in log_values.items():
            if isinstance(v, meter_utils.AverageValueMeter):
                v.reset()
