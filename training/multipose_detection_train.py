import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import transforms

from batch_processor import batch_processor
from datasets.coco_data.RetinaNet_data_pipeline import (AspectRatioBasedSampler,
                                                        Augmenter, CocoDataset,
                                                        Normalizer, Resizer,
                                                        collater)
from network.posenet import poseNet
from pose_utils.network.trainer import Trainer

# Hyper-params
coco_root = '/data/COCO/'
backbone = 'resnet101'  # 'resnet50'
opt = 'adam'
weight_decay = 0.000

# model parameters in MultiPoseNet
fpn_resnet_para = ['conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4']
fpn_retinanet_para = ['conv6', 'conv7', 'latlayer1', 'latlayer2',
                      'latlayer3', 'toplayer0', 'toplayer1', 'toplayer2']
fpn_keypoint_para = ['toplayer', 'flatlayer1', 'flatlayer2',
                     'flatlayer3', 'smooth1', 'smooth2', 'smooth3']
retinanet_para = ['regressionModel', 'classificationModel']
keypoint_para = ['convt1', 'convt2', 'convt3', 'convt4', 'convs1', 'convs2', 'convs3', 'convs4', 'upsample1',
                 'upsample2', 'upsample3', 'conv2', 'convfin']
prn_para = ['prn']

#####################################################################
# Set Training parameters
params = Trainer.TrainParams()
params.exp_name = 'res101_detection/'
params.subnet_name = 'detection_subnet'
params.save_dir = './extra/models/{}'.format(params.exp_name)
params.ckpt = './extra/models/ckpt_baseline_resnet101.h5'
params.ignore_opt_state = True

params.max_epoch = 50
params.init_lr = 1.e-5
params.lr_decay = 0.1

params.gpus = [0]
params.batch_size = 2 * len(params.gpus)
params.val_nbatch_end_epoch = 400

params.print_freq = 50

# model
if backbone == 'resnet101':
    model = poseNet(101)
elif backbone == 'resnet50':
    model = poseNet(50)

# Train detection subnet (RetinaNet), Fix the weights in backbone (ResNet) ans Key-point Subnet
for name, module in model.fpn.named_children():
    if name in fpn_resnet_para:
        for para in module.parameters():
            para.requires_grad = False
for name, module in model.fpn.named_children():
    if name in fpn_keypoint_para:
        for para in module.parameters():
            para.requires_grad = False
for name, module in model.named_children():
    if name in keypoint_para:
        for para in module.parameters():
            para.requires_grad = False
for name, module in model.named_children():
    if name in prn_para:
        for para in module.parameters():
            para.requires_grad = False

print("Loading dataset...")
# load training data
dataset_train = CocoDataset(coco_root, set_name='train2017',
                            transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
sampler = AspectRatioBasedSampler(
    dataset_train, batch_size=params.batch_size, drop_last=False)
train_data = DataLoader(dataset_train, num_workers=3,
                        collate_fn=collater, batch_sampler=sampler)
print('train dataset len: {}'.format(len(train_data.dataset)))

# load validation data
dataset_val = CocoDataset(coco_root, set_name='val2017',
                          transform=transforms.Compose([Normalizer(), Resizer()]))
sampler_val = AspectRatioBasedSampler(
    dataset_val, batch_size=int(params.batch_size/2), drop_last=False)
valid_data = DataLoader(dataset_val, num_workers=3,
                        collate_fn=collater, batch_sampler=sampler_val)
print('val dataset len: {}'.format(len(valid_data.dataset)))

trainable_vars = [param for param in model.parameters() if param.requires_grad]
if opt == 'adam':
    print("Training with adam")
    params.optimizer = torch.optim.Adam(
        trainable_vars, lr=params.init_lr, weight_decay=weight_decay)

params.lr_scheduler = ReduceLROnPlateau(
    params.optimizer, 'min', factor=params.lr_decay, patience=3, verbose=True)
trainer = Trainer(model, params, batch_processor, train_data, valid_data)
trainer.train()
