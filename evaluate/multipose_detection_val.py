import os, sys
root_path = os.path.realpath(__file__).split('/evaluate/multipose_detection_val.py')[0]
os.chdir(root_path)
sys.path.append(root_path)

from training.batch_processor import batch_processor
from network.posenet import poseNet
from datasets.coco import get_loader
from evaluate.tester import Tester

# Hyper-params
coco_root = '/data/COCO/'
backbone = 'resnet101'  # 'resnet50'
data_dir = coco_root+'images/'
mask_dir = coco_root
json_path = coco_root+'COCO.json'
inp_size = 608  # input size 608*608
feat_stride = 4

# Set Training parameters
params = Tester.TestParams()
params.subnet_name = 'detection_subnet'
params.gpus = [0]
params.ckpt = './demo/models/ckpt_baseline_resnet101.h5'
params.batch_size = 25 * len(params.gpus)
params.print_freq = 100

# validation data
valid_data = get_loader(json_path, data_dir, mask_dir, inp_size, feat_stride,
                        preprocess='resnet', batch_size=params.batch_size-10*len(params.gpus), training=False,
                        shuffle=False, num_workers=8, subnet=params.subnet_name)
print('val dataset len: {}'.format(len(valid_data.dataset)))

# model
if backbone == 'resnet101':
    model = poseNet(101)
elif backbone == 'resnet50':
    model = poseNet(50)

for name, module in model.named_children():
    for para in module.parameters():
        para.requires_grad = False

tester = Tester(model, params, batch_processor, valid_data)
tester.val()
