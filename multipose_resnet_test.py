from network.posenet import poseNet
from pose_utils.network.tester import Tester

# Set Training parameters
params = Tester.TestParams()
backbone='resnet101'  # 'resnet50'
params.testdata_dir = './extra/test_images/'
params.testresult_dir = './extra/output/'
params.gpus = [0]
params.ckpt = './extra/models/ckpt_baseline.h5'

# model
# model
if backbone == 'resnet101':
    model = poseNet(101)
elif backbone == 'resnet50':
    model = poseNet(50)

tester = Tester(model, params)
tester.test()  # pic_test
