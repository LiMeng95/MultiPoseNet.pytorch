from network.posenet import poseNet
from pose_utils.network.tester import Tester

# Set Training parameters
params = Tester.TestParams()
# params.trunk = 'vgg19'
params.testdata_dir = './extra/test_images/'
params.testresult_dir = './extra/output/'
params.gpus = [0]
params.ckpt = './extra/models/keypoint101/ckpt_baseline.h5'

# model
model = poseNet(101)

tester = Tester(model, params)
tester.test()  # pic_test
