# -*- coding:utf-8 -*-
# keypoint subnet + detection subnet(RetinaNet) + PRN
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from collections import OrderedDict
from network.fpn import FPN50, FPN101
from torch.nn import init

from network.utils import BBoxTransform, ClipBoxes
from network.anchors import Anchors
import network.losses as losses
from lib.nms.pth_nms import pth_nms


def nms(dets, thresh):
    "Dispatch to either CPU or GPU NMS implementations.\
    Accept dets as tensor"""
    return pth_nms(dets, thresh)


class Concat(nn.Module):
    def __init__(self):
        super(Concat, self).__init__()

    def forward(self, up1, up2, up3, up4):
        return torch.cat((up1, up2, up3, up4), 1)


class RegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, feature_size=256):
        super(RegressionModel, self).__init__()

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * 4, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)

        # out is B x C x W x H, with C = 4*num_anchors
        out = out.permute(0, 2, 3, 1)

        return out.contiguous().view(out.shape[0], -1, 4)


class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, num_classes=80, prior=0.01, feature_size=256):
        super(ClassificationModel, self).__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * num_classes, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)
        out = self.output_act(out)

        # out is B x C x W x H, with C = n_classes + n_anchors
        out1 = out.permute(0, 2, 3, 1)

        batch_size, width, height, channels = out1.shape

        out2 = out1.view(batch_size, width, height, self.num_anchors, self.num_classes)

        return out2.contiguous().view(x.shape[0], -1, self.num_classes)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class Add(nn.Module):
    def forward(self, input1, input2):
        return torch.add(input1, input2)


class PRN(nn.Module):
    def __init__(self,node_count, coeff):
        super(PRN, self).__init__()
        self.flatten   = Flatten()
        self.height    = coeff*28
        self.width     = coeff*18
        self.dens1     = nn.Linear(self.height*self.width*17, node_count)
        self.bneck     = nn.Linear(node_count, node_count)
        self.dens2     = nn.Linear(node_count, self.height*self.width*17)
        self.drop      = nn.Dropout()
        self.add       = Add()
        self.softmax   = nn.Softmax(dim=1)

    def forward(self, x):
        res = self.flatten(x)
        out = self.drop(F.relu(self.dens1(res)))
        out = self.drop(F.relu(self.bneck(out)))
        out = F.relu(self.dens2(out))
        out = self.add(out, res)
        out = self.softmax(out)
        out = out.view(out.size()[0], self.height, self.width, 17)

        return out

class poseNet(nn.Module):
    def __init__(self, layers, prn_node_count=1024, prn_coeff=2):
        super(poseNet, self).__init__()
        if layers == 101:
            self.fpn = FPN101()
        if layers == 50:
            self.fpn = FPN50()

        ##################################################################################
        # keypoints subnet
        # 2 conv(kernel=3x3)，change channels from 256 to 128
        self.convt1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.convt2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.convt3 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.convt4 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.convs1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.convs2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.convs3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.convs4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.upsample1 = nn.Upsample(scale_factor=8, mode='nearest', align_corners=None)
        self.upsample2 = nn.Upsample(scale_factor=4, mode='nearest', align_corners=None)
        self.upsample3 = nn.Upsample(scale_factor=2, mode='nearest', align_corners=None)
        # self.upsample4 = nn.Upsample(size=(120,120),mode='bilinear',align_corners=True)

        self.concat = Concat()
        self.conv2 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.convfin = nn.Conv2d(256, 17, kernel_size=1, stride=1, padding=0)

        ##################################################################################
        # detection subnet
        self.regressionModel = RegressionModel(256)
        self.classificationModel = ClassificationModel(256, num_classes=80)
        self.anchors = Anchors()
        self.regressBoxes = BBoxTransform()
        self.clipBoxes = ClipBoxes()
        self.focalLoss = losses.FocalLoss()

        ##################################################################################
        # prn subnet
        self.prn = PRN(prn_node_count, prn_coeff)

        ##################################################################################
        # initialize weights
        self._initialize_weights_norm()
        prior = 0.01
        self.classificationModel.output.weight.data.fill_(0)
        self.classificationModel.output.bias.data.fill_(-math.log((1.0 - prior) / prior))
        self.regressionModel.output.weight.data.fill_(0)
        self.regressionModel.output.bias.data.fill_(0)

        self.freeze_bn()  # from retinanet

    def _initialize_weights_norm(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:  # resnet101 conv2d doesn't add bias
                    init.constant_(m.bias, 0.0)

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def forward(self, x):

        img_batch, subnet_name = x

        if subnet_name == 'keypoint_subnet':
            return self.keypoint_forward(img_batch)
        elif subnet_name == 'detection_subnet':
            return self.detection_forward(img_batch)
        elif subnet_name == 'prn_subnet':
            return self.prn_forward(img_batch)
        else:  # entire_net
            features = self.fpn(img_batch)
            p2, p3, p4, p5 = features[0]  # fpn features for keypoint subnet
            features = features[1]  # fpn features for keypoint subnet

            ##################################################################################
            # keypoints subnet
            dt5 = self.convt1(p5)
            d5 = self.convs1(dt5)
            dt4 = self.convt2(p4)
            d4 = self.convs2(dt4)
            dt3 = self.convt3(p3)
            d3 = self.convs3(dt3)
            dt2 = self.convt4(p2)
            d2 = self.convs4(dt2)

            up5 = self.upsample1(d5)
            up4 = self.upsample2(d4)
            up3 = self.upsample3(d3)

            concat = self.concat(up5, up4, up3, d2)
            smooth = F.relu(self.conv2(concat))
            predict_keypoint = self.convfin(smooth)

            ##################################################################################
            # detection subnet
            regression = torch.cat([self.regressionModel(feature) for feature in features], dim=1)
            classification = torch.cat([self.classificationModel(feature) for feature in features], dim=1)
            anchors = self.anchors(img_batch)

            transformed_anchors = self.regressBoxes(anchors, regression)
            transformed_anchors = self.clipBoxes(transformed_anchors, img_batch)

            scores = torch.max(classification, dim=2, keepdim=True)[0]

            scores_over_thresh = (scores > 0.05)[0, :, 0]

            if scores_over_thresh.sum() == 0:
                # no boxes to NMS, just return
                return predict_keypoint, [torch.zeros(0), torch.zeros(0), torch.zeros(0, 4)]

            classification = classification[:, scores_over_thresh, :]
            transformed_anchors = transformed_anchors[:, scores_over_thresh, :]
            scores = scores[:, scores_over_thresh, :]

            anchors_nms_idx = nms(torch.cat([transformed_anchors, scores], dim=2)[0, :, :], 0.5)  # threshold = 0.5, inpsize=480

            nms_scores, nms_class = classification[0, anchors_nms_idx, :].max(dim=1)

            return predict_keypoint, [nms_scores, nms_class, transformed_anchors[0, anchors_nms_idx, :]]


    def keypoint_forward(self, img_batch):
        saved_for_loss = []

        p2, p3, p4, p5 = self.fpn(img_batch)[0] # fpn features for keypoint subnet

        ##################################################################################
        # keypoints subnet
        dt5 = self.convt1(p5)
        d5 = self.convs1(dt5)
        dt4 = self.convt2(p4)
        d4 = self.convs2(dt4)
        dt3 = self.convt3(p3)
        d3 = self.convs3(dt3)
        dt2 = self.convt4(p2)
        d2 = self.convs4(dt2)

        up5 = self.upsample1(d5)
        up4 = self.upsample2(d4)
        up3 = self.upsample3(d3)

        concat = self.concat(up5, up4, up3, d2)
        smooth = F.relu(self.conv2(concat))
        predict_keypoint = self.convfin(smooth)
        saved_for_loss.append(predict_keypoint)

        return predict_keypoint, saved_for_loss

    def detection_forward(self, img_batch):
        saved_for_loss = []

        features = self.fpn(img_batch)[1]  # fpn features for detection subnet

        ##################################################################################
        # detection subnet
        regression = torch.cat([self.regressionModel(feature) for feature in features], dim=1)
        classification = torch.cat([self.classificationModel(feature) for feature in features], dim=1)
        anchors = self.anchors(img_batch)

        saved_for_loss.append(classification)
        saved_for_loss.append(regression)
        saved_for_loss.append(anchors)

        return [], saved_for_loss

    def prn_forward(self, img_batch):
        saved_for_loss = []

        res = self.prn.flatten(img_batch)
        out = self.prn.drop(F.relu(self.prn.dens1(res)))
        out = self.prn.drop(F.relu(self.prn.bneck(out)))
        out = F.relu(self.prn.dens2(out))
        out = self.prn.add(out,res)
        out = self.prn.softmax(out)
        out = out.view(out.size()[0], self.prn.height, self.prn.width, 17)

        saved_for_loss.append(out)

        return out, saved_for_loss

    @staticmethod
    def build_loss(saved_for_loss, *args):

        subnet_name = args[0]

        if subnet_name == 'keypoint_subnet':
            return build_keypoint_loss(saved_for_loss, args[1], args[2], args[3], args[4])
        elif subnet_name == 'detection_subnet':
            return build_detection_loss(saved_for_loss, args[1])
        elif subnet_name == 'prn_subnet':
            return build_prn_loss(saved_for_loss, args[1])
        else:
            return 0


def build_keypoint_loss(saved_for_loss, heat_temp, heat_weight, batch_size, gpus):

    names = build_names()
    saved_for_log = OrderedDict()
    criterion = nn.MSELoss(size_average=True).cuda()
    total_loss = 0
    div1 = 1.

    pred1 = saved_for_loss[0] * heat_weight
    gt1 = heat_weight * heat_temp

    # Compute losses
    loss1 = criterion(pred1, gt1) / div1
    total_loss += loss1

    # Get value from Tensor and save for log
    saved_for_log[names[0]] = loss1.item()
    saved_for_log['max_ht'] = torch.max(
        saved_for_loss[-1].data[:, 0:-1, :, :]).item()
    saved_for_log['min_ht'] = torch.min(
        saved_for_loss[-1].data[:, 0:-1, :, :]).item()

    return total_loss, saved_for_log

def build_detection_loss(saved_for_loss, anno):
    '''
    :param saved_for_loss: [classifications, regressions, anchors]
    :param anno: annotations
    :return: classification_loss, regression_loss
    '''
    saved_for_log = OrderedDict()

    # Compute losses
    focalLoss = losses.FocalLoss()
    classification_loss, regression_loss = focalLoss(*saved_for_loss, anno)
    classification_loss = classification_loss.mean()
    regression_loss = regression_loss.mean()
    total_loss = classification_loss + regression_loss

    # Get value from Tensor and save for log
    saved_for_log['total_loss'] = total_loss.item()
    saved_for_log['classification_loss'] = classification_loss.item()
    saved_for_log['regression_loss'] = regression_loss.item()

    return total_loss, saved_for_log

def build_prn_loss(saved_for_loss, label):
    '''
    :param saved_for_loss: [out]
    :param label: label
    :return: prn loss
    '''
    saved_for_log = OrderedDict()

    criterion = nn.BCELoss(size_average=True).cuda()
    total_loss = 0

    # Compute losses
    loss1 = criterion(saved_for_loss[0], label)
    total_loss += loss1

    # Get value from Tensor and save for log
    saved_for_log['PRN loss'] = loss1.item()

    return total_loss, saved_for_log

def build_names():
    names = []
    for j in range(1, 2):
        names.append('loss_end%d' % j)
    return names

