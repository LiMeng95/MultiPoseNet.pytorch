# coding=utf-8
import os

import cv2
import numpy as np

import torch
from datasets.coco_data.heatmap import putGaussianMaps
from datasets.coco_data.ImageAugmentation import (aug_croppad, aug_flip, aug_rotate, aug_scale,
                                                  aug_croppad_bbox, aug_flip_bbox, aug_rotate_bbox, aug_scale_bbox)
from datasets.coco_data.preprocessing import resnet_preprocess
from torch.utils.data import DataLoader, Dataset
from functools import partial, reduce

from pycocotools.coco import COCO, maskUtils

'''
train2014  : 82783 simages
val2014    : 40504 images

first 2644 of val2014 marked by 'isValidation = 1', as our minval dataset.
So all training data have 82783+40504-2644 = 120643 samples
'''

params_transform = dict()
params_transform['mode'] = 5
# === aug_scale ===
params_transform['scale_min'] = 0.8
params_transform['scale_max'] = 1.2
params_transform['scale_prob'] = 1
params_transform['target_dist'] = 0.6
# === aug_rotate ===
params_transform['max_rotate_degree'] = 40

# ===
params_transform['center_perterb_max'] = 40

# === aug_flip ===
params_transform['flip_prob'] = 0.3

params_transform['np'] = 56
params_transform['sigma'] = 7.0

def annToRLE(ann, height, width):
    """
    Convert annotation which can be polygons, uncompressed RLE to RLE.
    :return: binary mask (numpy 2D array)
    """
    segm = ann['segmentation']
    if isinstance(segm, list):
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = maskUtils.frPyObjects(segm, height, width)
        rle = maskUtils.merge(rles)
    elif isinstance(segm['counts'], list):
        # uncompressed RLE
        rle = maskUtils.frPyObjects(segm, height, width)
    else:
        # rle
        rle = ann['segmentation']
    return rle


def annToMask(ann, height, width):
    """
    Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
    :return: binary mask (numpy 2D array)
    """
    rle = annToRLE(ann, height, width)
    m = maskUtils.decode(rle)
    return m

class Cocokeypoints(Dataset):
    def __init__(self, root, mask_dir, index_list, data, inp_size, feat_stride, preprocess='resnet', transform=None,
                 target_transform=None):

        params_transform['crop_size_x'] = inp_size
        params_transform['crop_size_y'] = inp_size
        params_transform['stride'] = feat_stride

        # add preprocessing as a choice, so we don't modify it manually.
        self.preprocess = preprocess
        self.data = data
        self.mask_dir = mask_dir
        self.numSample = len(index_list)
        self.index_list = index_list
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

    def get_anno(self, meta_data):
        """
        get meta information
        """
        anno = dict()
        anno['dataset'] = meta_data['dataset']
        anno['img_height'] = int(meta_data['img_height'])
        anno['img_width'] = int(meta_data['img_width'])

        anno['isValidation'] = meta_data['isValidation']
        anno['people_index'] = int(meta_data['people_index'])
        anno['annolist_index'] = int(meta_data['annolist_index'])

        # (b) objpos_x (float), objpos_y (float)
        anno['objpos'] = np.array(meta_data['objpos'])
        anno['scale_provided'] = meta_data['scale_provided']
        anno['joint_self'] = np.array(meta_data['joint_self'])

        anno['numOtherPeople'] = int(meta_data['numOtherPeople'])
        anno['num_keypoints_other'] = np.array(
            meta_data['num_keypoints_other'])
        anno['joint_others'] = np.array(meta_data['joint_others'])
        anno['objpos_other'] = np.array(meta_data['objpos_other'])
        anno['scale_provided_other'] = meta_data['scale_provided_other']
        anno['bbox_other'] = meta_data['bbox_other']
        anno['segment_area_other'] = meta_data['segment_area_other']

        if anno['numOtherPeople'] == 1:
            anno['joint_others'] = np.expand_dims(anno['joint_others'], 0)
            anno['objpos_other'] = np.expand_dims(anno['objpos_other'], 0)
        return anno

    def add_neck(self, meta):
        '''
        MS COCO annotation order:
        0: nose	   		1: l eye		2: r eye	3: l ear	4: r ear
        5: l shoulder	6: r shoulder	7: l elbow	8: r elbow
        9: l wrist		10: r wrist		11: l hip	12: r hip	13: l knee
        14: r knee		15: l ankle		16: r ankle

        The order in this work:
        (0-'nose'	1-'neck' 2-'right_shoulder' 3-'right_elbow' 4-'right_wrist'
        5-'left_shoulder' 6-'left_elbow'	    7-'left_wrist'  8-'right_hip'
        9-'right_knee'	 10-'right_ankle'	11-'left_hip'   12-'left_knee'
        13-'left_ankle'	 14-'right_eye'	    15-'left_eye'   16-'right_ear'
        17-'left_ear' )
        '''
        our_order = [0, 17, 6, 8, 10, 5, 7, 9,
                     12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
        # Index 6 is right shoulder and Index 5 is left shoulder
        right_shoulder = meta['joint_self'][6, :]
        left_shoulder = meta['joint_self'][5, :]
        neck = (right_shoulder + left_shoulder) / 2
        if right_shoulder[2] == 2 or left_shoulder[2] == 2:
            neck[2] = 2
        elif right_shoulder[2] == 1 or left_shoulder[2] == 1:
            neck[2] = 1
        else:
            neck[2] = right_shoulder[2] * left_shoulder[2]

        neck = neck.reshape(1, len(neck))
        neck = np.round(neck)
        meta['joint_self'] = np.vstack((meta['joint_self'], neck))
        meta['joint_self'] = meta['joint_self'][our_order, :]
        temp = []

        for i in range(meta['numOtherPeople']):
            right_shoulder = meta['joint_others'][i, 6, :]
            left_shoulder = meta['joint_others'][i, 5, :]
            neck = (right_shoulder + left_shoulder) / 2
            if (right_shoulder[2] == 2 or left_shoulder[2] == 2):
                neck[2] = 2
            elif (right_shoulder[2] == 1 or left_shoulder[2] == 1):
                neck[2] = 1
            else:
                neck[2] = right_shoulder[2] * left_shoulder[2]
            neck = neck.reshape(1, len(neck))
            neck = np.round(neck)
            single_p = np.vstack((meta['joint_others'][i], neck))
            single_p = single_p[our_order, :]
            temp.append(single_p)
        meta['joint_others'] = np.array(temp)

        return meta

    def remove_illegal_joint(self, meta):
        crop_x = int(params_transform['crop_size_x'])
        crop_y = int(params_transform['crop_size_y'])
        mask = np.logical_or.reduce((meta['joint_self'][:, 0] >= crop_x,
                                     meta['joint_self'][:, 0] < 0,
                                     meta['joint_self'][:, 1] >= crop_y,
                                     meta['joint_self'][:, 1] < 0))
        # out_bound = np.nonzero(mask)
        # print(mask.shape)
        meta['joint_self'][mask == True, :] = (1, 1, 2)
        if (meta['numOtherPeople'] != 0):
            mask = np.logical_or.reduce((meta['joint_others'][:, :, 0] >= crop_x,
                                         meta['joint_others'][:, :, 0] < 0,
                                         meta['joint_others'][:,
                                                              :, 1] >= crop_y,
                                         meta['joint_others'][:, :, 1] < 0))
            meta['joint_others'][mask == True, :] = (1, 1, 2)

        return meta

    def get_ground_truth(self, meta, mask_miss):

        number_keypoints = 18

        stride = params_transform['stride']
        mode = params_transform['mode']
        crop_size_y = params_transform['crop_size_y']
        crop_size_x = params_transform['crop_size_x']
        num_parts = params_transform['np']
        nop = meta['numOtherPeople']
        grid_y = int(crop_size_y / stride)
        grid_x = int(crop_size_x / stride)
        channels = (num_parts + 1) * 2
        heatmaps = np.zeros((grid_y, grid_x, number_keypoints))

        mask_miss = cv2.resize(mask_miss, (0, 0), fx=1.0 / stride, fy=1.0 / stride,
                               interpolation=cv2.INTER_CUBIC).astype(np.float32)
        mask_miss = mask_miss / 255.
        mask_miss = np.expand_dims(mask_miss, axis=2)
        heat_mask = np.repeat(mask_miss, number_keypoints, axis=2)  # 19

        #mask_all = cv2.resize(mask_all, (0, 0), fx=1.0 / stride, fy=1.0 / stride,
        #                      interpolation=cv2.INTER_CUBIC).astype(np.float32)
        #mask_all = mask_all / 255.
        #mask_all = np.expand_dims(mask_all, axis=2)

        # confidance maps for body parts
        for i in range(number_keypoints):
            if (meta['joint_self'][i, 2] <= 1):
                center = meta['joint_self'][i, :2]
                gaussian_map = heatmaps[:, :, i]
                heatmaps[:, :, i] = putGaussianMaps(
                    center, gaussian_map, params_transform=params_transform)
            for j in range(nop):
                if (meta['joint_others'][j, i, 2] <= 1):
                    center = meta['joint_others'][j, i, :2]
                    gaussian_map = heatmaps[:, :, i]
                    heatmaps[:, :, i] = putGaussianMaps(
                        center, gaussian_map, params_transform=params_transform)

        return heat_mask, heatmaps

    def __getitem__(self, index):
        idx = self.index_list[index]
        img = cv2.imread(os.path.join(self.root, self.data[idx]['img_paths']))
        img_idx = self.data[idx]['img_paths'][-16:-3]
#        print img.shape
        if "COCO_val" in self.data[idx]['dataset']:
            mask_miss = cv2.imread(
                self.mask_dir + 'mask2014/val2014_mask_miss_' + img_idx + 'png', 0)
            #mask_all = cv2.imread(
            #    self.mask_dir + 'mask2014/val2014_mask_all_' + img_idx + 'png', 0)
        elif "COCO" in self.data[idx]['dataset']:
            mask_miss = cv2.imread(
                self.mask_dir + 'mask2014/train2014_mask_miss_' + img_idx + 'png', 0)
            #mask_all = cv2.imread(
            #    self.mask_dir + 'mask2014/train2014_mask_all_' + img_idx + 'png', 0)
        meta_data = self.get_anno(self.data[idx])

        meta_data = self.add_neck(meta_data)

        augmentations = [
            partial(aug_meth, params_transform=params_transform)
            for aug_meth in [
                aug_scale,
                aug_rotate,
                aug_croppad,
                aug_flip
            ]
        ]

        meta_data, img, mask_miss = reduce(
            lambda md_i_mm_ma, f: f(*md_i_mm_ma),
            augmentations,
            (meta_data, img, mask_miss)
        )

        meta_data = self.remove_illegal_joint(meta_data)

        heat_mask, heatmaps = self.get_ground_truth(
            meta_data, mask_miss)

        # image preprocessing, which comply the model
        # trianed on Imagenet dataset
        if self.preprocess == 'resnet':
            img = resnet_preprocess(img)

        img = torch.from_numpy(img)
        heatmaps = torch.from_numpy(
            heatmaps.transpose((2, 0, 1)).astype(np.float32))
        heat_mask = torch.from_numpy(
            heat_mask.transpose((2, 0, 1)).astype(np.float32))
        #mask_all = torch.from_numpy(
        #    mask_all.transpose((2, 0, 1)).astype(np.float32))

        return img, heatmaps, heat_mask#, mask_all

    def __len__(self):
        return self.numSample

class Cocobbox(Dataset):
    def __init__(self, root, mask_dir, index_list, data, inp_size, feat_stride, coco,
                 preprocess='resnet', training=True):

        params_transform['crop_size_x'] = inp_size
        params_transform['crop_size_y'] = inp_size
        params_transform['stride'] = feat_stride

        # add preprocessing as a choice, so we don't modify it manually.
        self.preprocess = preprocess
        self.data = data
        self.index_list = index_list
        self.numSample = len(self.index_list)
        self.training = training

        if self.training:
            img_path = os.path.join(root, 'train2017')
        else:
            img_path = os.path.join(root, 'val2017')

        self.instance_info_list, self.image_path_list = self.get_instance_info_list(img_path, coco)

    def get_instance_info_list(self, img_path, coco):

        instance_info_list = []
        image_path_list = []

        for idx in self.index_list:
            image_info = coco.loadImgs(int(self.data[idx]['image_id']))[0]
            image_path = os.path.join(img_path, image_info['file_name'])
            if not os.path.exists(image_path):
                print(
                    "[skip] json annotation found, but cannot found image: {}".format(image_path))
                continue
            image_path_list.append(image_path)

            annos_ids = coco.getAnnIds(imgIds=self.data[idx]['image_id'])
            annos_info = coco.loadAnns(annos_ids)
            instance_info = {}
            instance_info["anns"] = annos_info
            instance_info["height"] = image_info["height"]
            instance_info["width"] = image_info["width"]
            instance_info_list.append(instance_info)

        return instance_info_list, image_path_list

    def get_instance_mask(self, instance_info):
        height = instance_info['height']
        width = instance_info['width']
        anns = instance_info['anns']

        instance_masks = []
        class_ids = []
        for anno in anns:
            class_id = 1
            m = annToMask(anno, height, width)
            # Some objects are so small that they're less than 1 pixel area
            # and end up rounded out. Skip those objects.
            if m.max() < 1:
                continue
            # Is it a crowd? If so, use a negative class ID.
            if anno['iscrowd']:
                # Use negative class ID for crowds
                class_id = -1
                # For crowd masks, annToMask() sometimes returns a mask
                # smaller than the given dimensions. If so, resize it.
                if m.shape[0] != height or m.shape[1] != width:
                    m = np.ones([height, width], dtype=bool)
            instance_masks.append(m)
            class_ids.append(class_id)
        return instance_masks, class_ids

    def get_anno(self, meta_data, instance_info):
        """
        get meta information
        """
        anno = dict()

        # (b) objpos_x (float), objpos_y (float)
        anno['objpos'] = np.array(meta_data['objpos'])
        anno['scale_provided'] = meta_data['scale_provided']

        anno['instance_mask_list'], anno['instance_cls_list'] = self.get_instance_mask(instance_info)

        return anno

    def get_ground_truth(self, meta, instance_info):
        extracted_bbox = []

        for m_idx, m in enumerate(meta['instance_mask_list']):
            if meta['instance_cls_list'][m_idx] == -1:  # is_crowd = 1
                if instance_info['anns'][m_idx]['iscrowd'] != 1:
                    print('is_crowd error')
                continue
            horizontal_indicies = np.where(np.any(m, axis=0))[0]
            vertical_indicies = np.where(np.any(m, axis=1))[0]
            if horizontal_indicies.shape[0]:
                x1, x2 = horizontal_indicies[[0, -1]]
                y1, y2 = vertical_indicies[[0, -1]]
                # x2 and y2 should not be part of the box. Increment by 1.
                x2 += 1
                y2 += 1
                bbox_cls = 0
            else:
                # No mask for this instance. Might happen due to
                # resizing or cropping. Set bbox to zeros
                x1, x2, y1, y2, bbox_cls = -1, -1, -1, -1, -1
            extracted_bbox.append([x1, y1, x2, y2, bbox_cls])

        return extracted_bbox

    def __getitem__(self, index):
        img = cv2.imread(self.image_path_list[index])

        idx = self.index_list[index]
        meta_data = self.get_anno(self.data[idx], self.instance_info_list[index])

        augmentations = [
            partial(aug_meth, params_transform=params_transform)
            for aug_meth in [
                aug_scale_bbox,
                aug_rotate_bbox,
                aug_croppad_bbox,
                aug_flip_bbox
            ]
        ]

        meta_data, img = reduce(
            lambda md_i_mm_ma, f: f(*md_i_mm_ma),
            augmentations,
            (meta_data, img)
        )

        extracted_bbox = self.get_ground_truth(meta_data, self.instance_info_list[index])

        # image preprocessing, which comply the model
        # trianed on Imagenet dataset
        if self.preprocess == 'resnet':
            img = resnet_preprocess(img)

        img = torch.from_numpy(img)
        bbox = torch.from_numpy(np.array(extracted_bbox).astype(np.float32))

        return img, bbox

    def __len__(self):
        return self.numSample

def bbox_collater(data):
    imgs = torch.stack([s[0] for s in data], 0)
    bbox = [s[1] for s in data]

    max_num_annots = max(bb.shape[0] for bb in bbox)

    bbox_padded = torch.ones((len(bbox), max_num_annots, 5)) * -1
    #print(annot_padded.shape)
    if max_num_annots > 0:
        for idx, bb in enumerate(bbox):
            #print(annot.shape)
            if bb.shape[0] > 0:
                bbox_padded[idx, :bb.shape[0], :] = bb

    return imgs, bbox_padded