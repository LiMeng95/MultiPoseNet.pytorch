import os
import json
from torchvision.transforms import ToTensor
from datasets.coco_data.COCO_data_pipeline import Cocokeypoints, Cocobbox, bbox_collater
from datasets.dataloader import sDataLoader
from pycocotools.coco import COCO


def get_loader(json_path, data_dir, mask_dir, inp_size, feat_stride, preprocess,
               batch_size, training=True, shuffle=True, num_workers=3, subnet='keypoint_subnet'):
    """ Build a COCO dataloader
    :param json_path: string, path to jso file
    :param datadir: string, path to coco data
    :returns : the data_loader
    """
    with open(json_path) as data_file:
        data_this = json.load(data_file)
        data = data_this['root']

    num_samples = len(data)
    train_indexes = []
    val_indexes = []

    if subnet == 'keypoint_subnet':
        for count in range(num_samples):
            if data[count]['isValidation'] != 0.:
                val_indexes.append(count)
            else:
                train_indexes.append(count)

        coco_data = Cocokeypoints(root=data_dir, mask_dir=mask_dir,
                                  index_list=train_indexes if training else val_indexes,
                                  data=data, inp_size=inp_size, feat_stride=feat_stride,
                                  preprocess=preprocess, transform=ToTensor())
        data_loader = sDataLoader(coco_data, batch_size=batch_size,
                                  shuffle=shuffle, num_workers=num_workers)

    elif subnet == 'detection_subnet':
        if training:
            anno_path = os.path.join(mask_dir, 'annotations', 'person_keypoints_train2017.json')
        else:
            anno_path = os.path.join(mask_dir, 'annotations', 'person_keypoints_val2017.json')
        coco = COCO(anno_path)
        images_ids = coco.getImgIds()

        data_indexes = []
        for count in range(num_samples):
            if int(data[count]['image_id']) in images_ids:
                data_indexes.append(count)

        coco_data = Cocobbox(root=data_dir, mask_dir=mask_dir, index_list=data_indexes,
                             data=data, inp_size=inp_size, feat_stride=feat_stride, coco=coco,
                             preprocess=preprocess, training=True if training else False)

        data_loader = sDataLoader(coco_data, batch_size=batch_size, shuffle=shuffle,
                                  num_workers=num_workers, collate_fn=bbox_collater)

    return data_loader
