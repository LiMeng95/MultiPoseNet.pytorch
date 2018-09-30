from network.posenet import poseNet
from pose_utils.datasets.coco import get_loader
from batch_processor import batch_processor
from pose_utils.network.tester import Tester


# Hyper-params
coco_root = '/data/COCO/'
backbone='resnet101'  # 'resnet50'
data_dir = coco_root+'images/'
mask_dir = coco_root
json_path = coco_root+'COCO.json'
inp_size = 480  # input size 480*480
feat_stride = 4

# Set Training parameters
params = Tester.TestParams()
params.subnet_name = 'keypoint_subnet'
params.gpus = [0]
params.ckpt = './extra/models/ckpt_baseline_resnet101.h5'
params.batch_size = 6 * len(params.gpus)
params.print_freq = 50

# validation data
valid_data = get_loader(json_path, data_dir, mask_dir, inp_size,
                        feat_stride, preprocess='resnet', training=False,
                        batch_size=params.batch_size-2 * len(params.gpus), shuffle=False, num_workers=4)
print('val dataset len: {}'.format(len(valid_data.dataset)))

# model
if backbone == 'resnet101':
    model = poseNet(101)
elif backbone == 'resnet50':
    model = poseNet(50)


tester = Tester(model, params, batch_processor, valid_data)
tester.val()
