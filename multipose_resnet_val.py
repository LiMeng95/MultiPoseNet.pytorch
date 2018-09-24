from network.posenet import poseNet
from pose_utils.datasets.coco import get_loader
from batch_processor import batch_processor
from pose_utils.network.tester import Tester


# Hyper-params
coco_root = '/data/COCO2014/'
data_dir = coco_root+'images/'
mask_dir = coco_root
json_path = coco_root+'COCO.json'
inp_size = 384  # input size 368
feat_stride = 4
backbone='resnet101'  # 'resnet50'

# Set Training parameters
params = Tester.TestParams()
params.gpus = [0]
params.ckpt = './extra/models/ckpt_baseline.h5'
params.batch_size = 10 * len(params.gpus)

# validation data
valid_data = get_loader(json_path, data_dir, mask_dir, inp_size,
                        feat_stride, preprocess='resnet', training=False,
                        batch_size=params.batch_size-4 * len(params.gpus), shuffle=False, num_workers=8)
print('val dataset len: {}'.format(len(valid_data.dataset)))

# model
if backbone == 'resnet101':
    model = poseNet(101)
elif backbone == 'resnet50':
    model = poseNet(50)


tester = Tester(model, params, batch_processor, valid_data)
tester.val()
