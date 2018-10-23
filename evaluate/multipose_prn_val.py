import os, sys
root_path = os.path.realpath(__file__).split('/evaluate/multipose_prn_val.py')[0]
os.chdir(root_path)
sys.path.append(root_path)

from network.posenet import poseNet
from pycocotools.coco import COCO
from datasets.coco_data.prn_data_pipeline import PRN_CocoDataset
from torch.utils.data import DataLoader
from training.batch_processor import batch_processor
from evaluate.tester import Tester


# Hyper-params
coco_root = '/data/COCO/'
backbone='resnet101'  # 'resnet50'
inp_size = 480  # input size 480*480
feat_stride = 4
node_count = 1024  # Hidden Layer Node Count
coeff = 2  # Coefficient of bbox size
threshold = 0.21  # BBOX threshold
num_of_keypoints = 3  # Minimum number of keypoints for each bbox in training

# Set Training parameters
params = Tester.TestParams()
params.subnet_name = 'prn_subnet'
params.gpus = [0]
params.ckpt = './demo/models/ckpt_baseline_resnet101.h5'
params.batch_size = 8 * len(params.gpus)
params.print_freq = 500

# validation data
coco_val = COCO(os.path.join(coco_root, 'annotations/person_keypoints_val2017.json'))
valid_data = DataLoader(dataset=PRN_CocoDataset(
    coco_val, num_of_keypoints=num_of_keypoints, coeff=coeff, threshold=threshold,
    inp_size=inp_size, feat_stride=feat_stride), batch_size=params.batch_size, num_workers=4, shuffle=False)
print('val dataset len: {}'.format(len(valid_data.dataset)))

# model
if backbone == 'resnet101':
    model = poseNet(101, prn_node_count=node_count, prn_coeff=coeff)
elif backbone == 'resnet50':
    model = poseNet(50, prn_node_count=node_count, prn_coeff=coeff)

for name, module in model.named_children():
    for para in module.parameters():
        para.requires_grad = False

tester = Tester(model, params, batch_processor, valid_data)
tester.val()
