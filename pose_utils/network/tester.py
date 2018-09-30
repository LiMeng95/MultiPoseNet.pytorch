from __future__ import print_function

import os
import cv2
import datetime
import numpy as np
from collections import OrderedDict
from joint_utils import get_joint_list, plot_result
from tqdm import tqdm

import torch
import torch.nn as nn
from pose_utils.utils.log import logger
import pose_utils.utils.meter as meter_utils
import pose_utils.network.net_utils as net_utils
from pose_utils.utils.timer import Timer
from pose_utils.datasets.coco_data.preprocessing import resnet_preprocess

class TestParams(object):

    trunk = 'resnet101'  # select the model

    testdata_dir = './extra/test_images/'
    testresult_dir = './extra/output/'
    testresult_write_image = False  # write image results or not
    testresult_write_json = True  # write json results or not
    gpus = [0]
    ckpt = './extra/models/ckpt_baseline.h5'  # checkpoint file to load, no need to change this

    # # required params
    inp_size = 480  # input size 480*480
    exp_name = 'multipose101'
    subnet_name = 'keypoint_subnet'
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
        self.model.module.freeze_bn()

    def test(self):

        img_list = os.listdir(self.params.testdata_dir)
        multipose_results = []

        for img_name in tqdm(img_list):

            img = cv2.imread(os.path.join(self.params.testdata_dir, img_name)).astype(np.float32)
            shape_dst = np.max(img.shape)
            scale = float(shape_dst) / self.params.inp_size
            pad_size = np.abs(img.shape[1] - img.shape[0])
            img_resized = np.pad(img, ([0, pad_size], [0, pad_size], [0, 0]), 'constant')[:shape_dst, :shape_dst]
            img_resized = cv2.resize(img_resized, (self.params.inp_size, self.params.inp_size))
            img_input = resnet_preprocess(img_resized)
            img_input = torch.from_numpy(np.expand_dims(img_input, 0))

            with torch.no_grad():
                img_input = img_input.cuda(device=self.params.gpus[0])

            heatmaps, [scores, classification, transformed_anchors] = self.model([img_input, self.params.subnet_name])
            heatmaps = heatmaps.cpu().detach().numpy()
            heatmaps = np.squeeze(heatmaps, 0)
            heatmaps = np.transpose(heatmaps, (1, 2, 0))
            heatmap_max = np.max(heatmaps[:, :, :17], 2)
            # segment_map = heatmaps[:, :, 17]
            param = {'thre1': 0.1, 'thre2': 0.05, 'thre3': 0.5}
            joint_list = get_joint_list(img_resized, param, heatmaps[:, :, :17], scale)
            del img_resized

            # bounding box from retinanet
            scores = scores.cpu().detach().numpy()
            classification = classification.cpu().detach().numpy()
            transformed_anchors = transformed_anchors.cpu().detach().numpy()
            idxs = np.where(scores > 0.5)
            bboxs=[]
            for j in range(idxs[0].shape[0]):
                bbox = transformed_anchors[idxs[0][j], :]*scale
                if int(classification[idxs[0][j]]) == 0:  # class0=people
                    bboxs.append(bbox.tolist())

            result = {
                "image_name": img_name,
                "joint_list": joint_list.tolist(),
                "bbox_list": bboxs
            }
            multipose_results.append(result)

            if self.params.testresult_write_image:
                plot_bbox, plot_joints = plot_result(img, result)
                cv2.imwrite(os.path.join(self.params.testresult_dir, img_name.split('.', 1)[0] + '_1heatmap.png'), heatmap_max * 256)
                cv2.imwrite(os.path.join(self.params.testresult_dir, img_name.split('.', 1)[0] + '_2bbox.png'), plot_bbox)
                cv2.imwrite(os.path.join(self.params.testresult_dir, img_name.split('.', 1)[0] + '_3keypoints.png'), plot_joints)
                # cv2.imwrite(os.path.join(self.params.testresult_dir, img_name.split('.', 1)[0] + '_4associations.png'), canvas)

        if self.params.testresult_write_json:
            import json
            with open(self.params.testresult_dir+'multipose_results.json', "w") as f:
                json.dump(multipose_results, f)

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
