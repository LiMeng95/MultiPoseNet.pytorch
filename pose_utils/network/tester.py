from __future__ import print_function

import os
import cv2
import datetime
from PIL import Image
import numpy as np
from collections import OrderedDict
from plot_heatmap import plot_heatmap

import torch
import torch.nn as nn
from pose_utils.utils.log import logger
import pose_utils.utils.meter as meter_utils
import pose_utils.network.net_utils as net_utils
from pose_utils.utils.timer import Timer

import torchvision.transforms.functional as tv_F

class TestParams(object):

    trunk = 'vgg19'  # select the model

    testdata_dir = './extra/test_images/'
    testresult_dir = './extra/output/'
    gpus = [0]
    ckpt = './extra/models/keypoint101/ckpt_baseline.h5'  # checkpoint file to load, no need to change this

    # # required params
    exp_name = 'multipose101'
    batch_size = 32
    print_freq = 20

class Tester(object):

    TestParams = TestParams

    def __init__(self, model, train_params, batch_processor=None, val_data=None):
        assert isinstance(train_params, TestParams)
        self.params = train_params
        self.batch_timer = Timer()
        self.data_timer = Timer()
        self.val_data = val_data if val_data else None
        self.batch_processor = batch_processor if batch_processor else None

        # load model
        self.model = model
        ckpt = self.params.ckpt

        if ckpt is not None:
            self._load_ckpt(ckpt)
            logger.info('Load ckpt from {}'.format(ckpt))

        self.model = nn.DataParallel(self.model, device_ids=self.params.gpus)
        self.model = self.model.cuda(device=self.params.gpus[0])
        self.model.eval()

    def test(self):

        img_list = os.listdir(self.params.testdata_dir)

        for img_name in img_list:
            print('Processing image: ' + img_name)

            img = Image.open(os.path.join(self.params.testdata_dir, img_name))
            shape_dst = max(img.size)
            pad_size = abs(img.size[1] - img.size[0])
            img = tv_F.pad(img, (0, 0, pad_size, pad_size))
            img = tv_F.crop(img, 0, 0, shape_dst, shape_dst)
            img_resized = tv_F.resize(img, (384, 384))
            img_input = tv_F.to_tensor(img_resized)
            img_resized = np.asarray(img_resized)
            img_input = torch.unsqueeze(tv_F.normalize(img_input, [0.406, 0.485, 0.456], [0.225, 0.229, 0.224]), 0)
            with torch.no_grad():
                img_input = img_input.cuda(device=self.params.gpus[0])

            heatmaps, _ = self.model(img_input)
            heatmaps = heatmaps.cpu().detach().numpy()
            heatmaps = np.squeeze(heatmaps, 0)
            heatmaps = np.transpose(heatmaps, (1, 2, 0))
            # vgg_out = np.transpose(vgg_out.cpu().data.numpy(), (1, 2, 0))
            heatmap_max = np.max(heatmaps[:, :, :17], 2)
            # segment_map = heatmaps[:, :, 17]
            param = {'thre1': 0.1, 'thre2': 0.05, 'thre3': 0.5}
            to_plot= plot_heatmap(img_resized, param, heatmaps[:, :, :17])
            cv2.imwrite(os.path.join(self.params.testresult_dir, img_name.split('.', 1)[0] + '_1heatmap.png'), heatmap_max * 256)
            # cv2.imwrite(os.path.join(self.params.testresult_dir, img_name.split('.', 1)[0] + '_2seg.png'), segment_map * 256)
            cv2.imwrite(os.path.join(self.params.testresult_dir, img_name.split('.', 1)[0] + '_3keypoints.png'), to_plot)
            # cv2.imwrite(os.path.join(self.params.testresult_dir, img_name.split('.', 1)[0] + '_4associations.png'), canvas)
            print('completed...')

    def val(self):
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

            inputs, gts, _ = self.batch_processor(self, batch)
            _, saved_for_loss = self.model(*inputs)
            self.batch_timer.toc()

            loss, saved_for_log = self.model.module.build_loss(saved_for_loss, *gts)
            sum_loss.add(loss.item())
            self._process_log(saved_for_log, logs)

            if step % self.params.print_freq == 0:
                self._print_log(step, logs, 'Validation', max_n_batch=len(self.val_data))

            self.data_timer.tic()
            self.batch_timer.tic()

        mean, std = sum_loss.value()
        logger.info('\n\nValidation loss: mean: {}, std: {}'.format(mean, std))

    def _load_ckpt(self, ckpt):
        _, _ = net_utils.load_net(ckpt, self.model, load_state_dict=True)

    def _process_log(self, src_dict, dest_dict):
        for k, v in src_dict.items():
            if isinstance(v, (int, float)):
                dest_dict.setdefault(k, meter_utils.AverageValueMeter())
                dest_dict[k].add(float(v))
            else:
                dest_dict[k] = v

    def _print_log(self, step, log_values, title='', max_n_batch=None):
        log_str = '{}\n'.format(self.params.exp_name)
        log_str += '{}: epoch {}'.format(title, 0)

        log_str += '[{}/{}]'.format(step, max_n_batch)

        i = 0
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
