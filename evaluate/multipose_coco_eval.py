import os, sys
root_path = os.path.realpath(__file__).split('/evaluate/multipose_coco_eval.py')[0]
os.chdir(root_path)
sys.path.append(root_path)

from network.posenet import poseNet
from evaluate.tester import Tester

backbone = 'resnet101'

# Set Training parameters
params = Tester.TestParams()
params.subnet_name = 'both'
params.inp_size = 480  # input picture size = (inp_size, inp_size)
params.coeff = 2
params.in_thres = 0.21
params.coco_root = '/data/COCO/'
params.testresult_write_json = False  # Whether to write json result
params.coco_result_filename = './demo/multipose_coco2017_results.json'
params.ckpt = './demo/models/ckpt_baseline_resnet101.h5'

# model
if backbone == 'resnet101':
    model = poseNet(101)
elif backbone == 'resnet50':
    model = poseNet(50)

tester = Tester(model, params)
tester.coco_eval()  # pic_test
