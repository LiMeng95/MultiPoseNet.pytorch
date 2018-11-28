import os, sys
root_path = os.path.realpath(__file__).split('/training/multipose_prn_train.py')[0]
os.chdir(root_path)
sys.path.append(root_path)

import torch
import torch.backends.cudnn as cudnn
from pycocotools.coco import COCO
from torch.utils.data import DataLoader
from training.trainer import Trainer
from datasets.coco_data.prn_data_pipeline import PRN_CocoDataset
from network.posenet import poseNet
from training.batch_processor import batch_processor
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Hyper-params
coco_root = '/data/COCO/'
backbone='resnet101'  # 'resnet50'
opt = 'adam'
inp_size = 480  # input size 480*480
feat_stride = 4
node_count = 1024  # Hidden Layer Node Count
coeff = 2  # Coefficient of bbox size
threshold = 0.21  # BBOX threshold
num_of_keypoints = 3  # Minimum number of keypoints for each bbox in training

# model parameters in MultiPoseNet
prn_para = ['prn']

#####################################################################
# Set Training parameters
params = Trainer.TrainParams()
params.exp_name = 'prn_subnet/'
params.subnet_name = 'prn_subnet'
params.save_dir = './extra/models/{}'.format(params.exp_name)
params.ckpt = './demo/models/ckpt_baseline_resnet101.h5'
params.ignore_opt_state = True

params.max_epoch = 40
params.init_lr = 1.0e-3
params.lr_decay = 0.9

params.gpus = [0]
params.batch_size = 8 * len(params.gpus)
params.val_nbatch_end_epoch = 2000

params.print_freq = 1000

# model
if backbone == 'resnet101':
    model = poseNet(101, prn_node_count=node_count, prn_coeff=coeff)
elif backbone == 'resnet50':
    model = poseNet(50, prn_node_count=node_count, prn_coeff=coeff)

# Train Key-point Subnet, Fix the weights in detection subnet (RetinaNet)
for name, module in model.named_children():
    if name not in prn_para:
        for para in module.parameters():
            para.requires_grad = False

print("Loading dataset...")
# load training data
coco_train = COCO(os.path.join(coco_root, 'annotations/person_keypoints_train2017.json'))
train_data = DataLoader(dataset=PRN_CocoDataset(
    coco_train, num_of_keypoints=num_of_keypoints, coeff=coeff, threshold=threshold,
    inp_size=inp_size, feat_stride=feat_stride),batch_size=params.batch_size, num_workers=4, shuffle=True)
print('train dataset len: {}'.format(len(train_data.dataset)))

# load validation data
valid_data = None
if params.val_nbatch > 0:
    coco_val = COCO(os.path.join(coco_root, 'annotations/person_keypoints_val2017.json'))
    valid_data = DataLoader(dataset=PRN_CocoDataset(
        coco_val, num_of_keypoints=num_of_keypoints, coeff=coeff, threshold=threshold,
        inp_size=inp_size, feat_stride=feat_stride), batch_size=params.batch_size, num_workers=4, shuffle=True)
    print('val dataset len: {}'.format(len(valid_data.dataset)))

trainable_vars = [param for param in model.parameters() if param.requires_grad]
if opt == 'adam':
    print("Training with adam")
    params.optimizer = torch.optim.Adam(
        trainable_vars, lr=params.init_lr)

cudnn.benchmark = True
params.lr_scheduler = ReduceLROnPlateau(params.optimizer, 'min', factor=params.lr_decay, patience=2, verbose=True)
trainer = Trainer(model, params, batch_processor, train_data, valid_data)
trainer.train()
