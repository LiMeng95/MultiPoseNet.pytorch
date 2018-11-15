# coding=utf-8

import random
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import misc, ndimage


"""The purpose of Augmentor is to automate image augmentation 
   in order to expand datasets as input for our algorithms.
:aut_scale : Scales them by dice2 (<1, so it is zoom out). 
:aug_croppad centerB: int with shape (2,), centerB will point to centerA.
:aug_flip: Mirrors the image around a vertical line running through its center.
:aug_rotate: Rotates the image. The angle of rotation, in degrees, 
             is specified by a random integer value that is included
             in the transform argument.
             
:param params_transform: store the value of stride and crop_szie_y, crop_size_x                 
"""


def aug_scale(meta, img, mask_miss, params_transform):
    dice = random.random()  # (0,1)
    if (dice > params_transform['scale_prob']):

        scale_multiplier = 1
    else:
        dice2 = random.random()
        # linear shear into [scale_min, scale_max]
        scale_multiplier = (
            params_transform['scale_max'] - params_transform['scale_min']) * dice2 + \
            params_transform['scale_min']
    scale_abs = params_transform['target_dist'] / meta['scale_provided']
    scale = scale_abs * scale_multiplier
    img = cv2.resize(img, (0, 0), fx=scale, fy=scale,
                     interpolation=cv2.INTER_CUBIC)

    mask_miss = cv2.resize(mask_miss, (0, 0), fx=scale,
                           fy=scale, interpolation=cv2.INTER_CUBIC)
    #mask_all = cv2.resize(mask_all, (0, 0), fx=scale,
    #                      fy=scale, interpolation=cv2.INTER_CUBIC)

    # modify meta data
    meta['objpos'] *= scale
    meta['joint_self'][:, :2] *= scale
    if (meta['numOtherPeople'] != 0):
        meta['objpos_other'] *= scale
        meta['joint_others'][:, :, :2] *= scale
    return meta, img, mask_miss#, mask_all


def aug_croppad(meta, img, mask_miss, params_transform):
    dice_x = random.random()
    dice_y = random.random()
    crop_x = int(params_transform['crop_size_x'])
    crop_y = int(params_transform['crop_size_y'])
    x_offset = int((dice_x - 0.5) * 2 *
                   params_transform['center_perterb_max'])
    y_offset = int((dice_y - 0.5) * 2 *
                   params_transform['center_perterb_max'])

    center = meta['objpos'] + np.array([x_offset, y_offset])
    center = center.astype(int)

    # pad up and down
    pad_v = np.ones((crop_y, img.shape[1], 3), dtype=np.uint8) * 128
    pad_v_mask_miss = np.ones(
        (crop_y, mask_miss.shape[1]), dtype=np.uint8) * 255

    img = np.concatenate((pad_v, img, pad_v), axis=0)
    mask_miss = np.concatenate(
        (pad_v_mask_miss, mask_miss, pad_v_mask_miss), axis=0)
    #mask_all = np.concatenate(
    #    (pad_v_mask_miss, mask_all, pad_v_mask_miss), axis=0)

    # pad right and left
    pad_h = np.ones((img.shape[0], crop_x, 3), dtype=np.uint8) * 128
    pad_h_mask_miss = np.ones(
        (mask_miss.shape[0], crop_x), dtype=np.uint8) * 255

    img = np.concatenate((pad_h, img, pad_h), axis=1)
    mask_miss = np.concatenate(
        (pad_h_mask_miss, mask_miss, pad_h_mask_miss), axis=1)
    #mask_all = np.concatenate(
    #    (pad_h_mask_miss, mask_all, pad_h_mask_miss), axis=1)

    img = img[center[1] + int(crop_y / 2):center[1] + int(crop_y / 2) + crop_y,
              center[0] + int(crop_x / 2):center[0] + int(crop_x / 2) + crop_x, :]

    mask_miss = mask_miss[center[1] + int(crop_y / 2):center[1] + int(crop_y / 2) + crop_y + 1, center[0] +int(crop_x / 2):center[0] + int(crop_x / 2) + crop_x + 1]
    #mask_all = mask_all[center[1] + int(crop_y / 2):center[1] + int(crop_y / 2) + crop_y + 1, center[0] + int(crop_x / 2):center[0] + int(crop_x / 2) + crop_x + 1]

    offset_left = crop_x / 2 - center[0]
    offset_up = crop_y / 2 - center[1]

    offset = np.array([offset_left, offset_up])
    meta['objpos'] += offset
    meta['joint_self'][:, :2] += offset
    mask = np.logical_or.reduce((meta['joint_self'][:, 0] >= crop_x,
                                 meta['joint_self'][:, 0] < 0,
                                 meta['joint_self'][:, 1] >= crop_y,
                                 meta['joint_self'][:, 1] < 0))

    meta['joint_self'][mask == True, 2] = 2
    if (meta['numOtherPeople'] != 0):
        meta['objpos_other'] += offset
        meta['joint_others'][:, :, :2] += offset
        mask = np.logical_or.reduce((meta['joint_others'][:, :, 0] >= crop_x,
                                     meta['joint_others'][:, :, 0] < 0,
                                     meta['joint_others'][:, :, 1] >= crop_y,
                                     meta['joint_others'][:, :, 1] < 0))

        meta['joint_others'][mask == True, 2] = 2

    return meta, img, mask_miss#, mask_all


def aug_flip(meta, img, mask_miss, params_transform):
    mode = params_transform['mode']
    num_other_people = meta['numOtherPeople']
    dice = random.random()
    doflip = dice <= params_transform['flip_prob']

    if doflip:
        img = img.copy()
        cv2.flip(src=img, flipCode=1, dst=img)
        w = img.shape[1]

        mask_miss = mask_miss.copy()
        #mask_all = mask_all.copy()
        cv2.flip(src=mask_miss, flipCode=1, dst=mask_miss)
        #cv2.flip(src=mask_all, flipCode=1, dst=mask_all)

        '''
        The order in this work:
            (0-'nose'   1-'neck' 2-'right_shoulder' 3-'right_elbow' 4-'right_wrist'
            5-'left_shoulder' 6-'left_elbow'        7-'left_wrist'  8-'right_hip'  
            9-'right_knee'   10-'right_ankle'   11-'left_hip'   12-'left_knee' 
            13-'left_ankle'  14-'right_eye'     15-'left_eye'   16-'right_ear' 
            17-'left_ear' )
        '''
        meta['objpos'][0] = w - 1 - meta['objpos'][0]
        meta['joint_self'][:, 0] = w - 1 - meta['joint_self'][:, 0]
        # print meta['joint_self']
        meta['joint_self'] = meta['joint_self'][[0, 1, 5, 6,
                                                 7, 2, 3, 4, 11, 12, 13, 8, 9, 10, 15, 14, 17, 16]]
        if (num_other_people != 0):
            meta['objpos_other'][:, 0] = w - 1 - meta['objpos_other'][:, 0]
            meta['joint_others'][:, :, 0] = w - \
                1 - meta['joint_others'][:, :, 0]
            for i in range(num_other_people):
                meta['joint_others'][i] = meta['joint_others'][i][[
                    0, 1, 5, 6, 7, 2, 3, 4, 11, 12, 13, 8, 9, 10, 15, 14, 17, 16]]

    return meta, img, mask_miss#, mask_all


def rotatepoint(p, R):
    point = np.zeros((3, 1))
    point[0] = p[0]
    point[1] = p[1]
    point[2] = 1

    new_point = R.dot(point)

    p[0] = new_point[0]

    p[1] = new_point[1]
    return p


# The correct way to rotation an image
# http://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/


def rotate_bound(image, angle, bordervalue):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT,
                          borderValue=bordervalue), M


def aug_rotate(meta, img, mask_miss, params_transform, type="random", input=0, fillType="nearest", constant=0):
    dice = random.random()
    degree = (dice - 0.5) * 2 * \
        params_transform['max_rotate_degree']  # degree [-40,40]

    img_rot, R = rotate_bound(img, np.copy(degree), (128, 128, 128))

    # Not sure it will cause mask_miss to rotate rightly, just avoid it fails
    # by np.copy().
    mask_miss_rot, _ = rotate_bound(mask_miss, np.copy(degree), (255))
    #mask_all_rot, _ = rotate_bound(mask_all, np.copy(degree), (255))

    # modify meta data
    meta['objpos'] = rotatepoint(meta['objpos'], R)

    for i in range(18):
        meta['joint_self'][i, :] = rotatepoint(meta['joint_self'][i, :], R)

    for j in range(meta['numOtherPeople']):

        meta['objpos_other'][j, :] = rotatepoint(meta['objpos_other'][j, :], R)

        for i in range(18):
            meta['joint_others'][j, i, :] = rotatepoint(
                meta['joint_others'][j, i, :], R)

    return meta, img_rot, mask_miss_rot#, mask_all_rot


def aug_scale_bbox(meta, img, params_transform):
    dice = random.random()  # (0,1)
    if (dice > params_transform['scale_prob']):

        scale_multiplier = 1
    else:
        dice2 = random.random()
        # linear shear into [scale_min, scale_max]
        scale_multiplier = (
            params_transform['scale_max'] - params_transform['scale_min']) * dice2 + \
            params_transform['scale_min']
    scale_abs = params_transform['target_dist'] / meta['scale_provided']
    scale = scale_abs * scale_multiplier
    img = cv2.resize(img, (0, 0), fx=scale, fy=scale,
                     interpolation=cv2.INTER_CUBIC)

    meta['objpos'] *= scale
    adjust_instance_list = []
    for m in meta['instance_mask_list']:
        m = cv2.resize(m, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        adjust_instance_list.append(m)
    meta['instance_mask_list'] = adjust_instance_list

    return meta, img


def aug_croppad_bbox(meta, img, params_transform):
    dice_x = random.random()
    dice_y = random.random()
    crop_x = int(params_transform['crop_size_x'])
    crop_y = int(params_transform['crop_size_y'])
    x_offset = int((dice_x - 0.5) * 2 *
                   params_transform['center_perterb_max'])
    y_offset = int((dice_y - 0.5) * 2 *
                   params_transform['center_perterb_max'])

    center = meta['objpos'] + np.array([x_offset, y_offset])
    center = center.astype(int)

    # pad up and down
    pad_v = np.ones((crop_y, img.shape[1], 3), dtype=np.uint8) * 128
    mask = meta['instance_mask_list'][0]
    pad_v_instance_mask = np.zeros(
        (crop_y, mask.shape[1]), dtype=np.uint8)

    img = np.concatenate((pad_v, img, pad_v), axis=0)
    adjust_instance_list = []
    for m in meta['instance_mask_list']:
        m = np.concatenate((pad_v_instance_mask, m, pad_v_instance_mask), axis=0)
        adjust_instance_list.append(m)
    meta['instance_mask_list'] = adjust_instance_list

    # pad right and left
    pad_h = np.ones((img.shape[0], crop_x, 3), dtype=np.uint8) * 128
    mask = meta['instance_mask_list'][0]
    pad_h_instance_mask = np.zeros(
        (mask.shape[0], crop_x), dtype=np.uint8)

    img = np.concatenate((pad_h, img, pad_h), axis=1)
    adjust_instance_list = []
    for m in meta['instance_mask_list']:
        m = np.concatenate((pad_h_instance_mask, m, pad_h_instance_mask), axis=1)
        m = m[center[1] + int(crop_y / 2):center[1] + int(crop_y / 2) + crop_y + 1,
            center[0] + int(crop_x / 2):center[0] + int(crop_x / 2) + crop_x + 1]
        adjust_instance_list.append(m)
    meta['instance_mask_list'] = adjust_instance_list

    img = img[center[1] + int(crop_y / 2):center[1] + int(crop_y / 2) + crop_y,
              center[0] + int(crop_x / 2):center[0] + int(crop_x / 2) + crop_x, :]

    return meta, img


def aug_flip_bbox(meta, img, params_transform):
    dice = random.random()
    doflip = dice <= params_transform['flip_prob']

    if doflip:
        img = img.copy()
        cv2.flip(src=img, flipCode=1, dst=img)

        adjust_instance_list = []
        for m in meta['instance_mask_list']:
            m = m.copy()
            cv2.flip(src=m, flipCode=1, dst=m)
            adjust_instance_list.append(m)
        meta['instance_mask_list'] = adjust_instance_list

    return meta, img


def aug_rotate_bbox(meta, img, params_transform, type="random", input=0, fillType="nearest", constant=0):
    dice = random.random()
    degree = (dice - 0.5) * 2 * \
        params_transform['max_rotate_degree']  # degree [-40,40]

    img_rot, _ = rotate_bound(img, np.copy(degree), (128, 128, 128))

    # Not sure it will cause mask_miss to rotate rightly, just avoid it fails
    # by np.copy().
    adjust_instance_list = []
    for m in meta['instance_mask_list']:
        m, _ = rotate_bound(m, np.copy(degree), (0))
        adjust_instance_list.append(m)
    meta['instance_mask_list'] = adjust_instance_list

    return meta, img_rot