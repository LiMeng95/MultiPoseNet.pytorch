import torch
import torch.utils.model_zoo as model_zoo
from pose_utils.datasets.coco import get_loader
from pose_utils.network.trainer import Trainer
from network.posenet import poseNet
from batch_processor import batch_processor
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Hyper-params
coco_root = '/data/COCO2014/'
data_dir = coco_root+'images/'
mask_dir = coco_root
json_path = coco_root+'COCO.json'
backbone='resnet101'  # 'resnet50'
opt = 'adam'
momentum = 0.9
weight_decay = 0.000
nesterov = True
inp_size = 384  # input size 368*368
feat_stride = 4

model_path = './models/'

# Set Training parameters
params = Trainer.TrainParams()
params.exp_name = 'your_exp_name/'
params.save_dir = './extra/models/{}'.format(params.exp_name) #./extra/models/{}'.format(params.exp_name)
params.ckpt = None  #None checkpoint file to load
params.re_init = False

params.max_epoch = 50
params.init_lr = 1.e-4
params.lr_decay = 0.1

params.gpus = [0]
params.batch_size = 10 * len(params.gpus)
params.val_nbatch = 2
params.val_nbatch_epoch = 200
params.save_freq = 3000

params.print_freq = 50
params.tensorboard_freq = 200
params.tensorboard_hostname = None  # '127.0.0.1'

print("Loading dataset...")
# load data
train_data = get_loader(json_path, data_dir,
                        mask_dir, inp_size, feat_stride,
                        'vgg', params.batch_size,
                        shuffle=True, training=True, num_workers=4)
print('train dataset len: {}'.format(len(train_data.dataset)))

# validation data
valid_data = None
if params.val_nbatch > 0:
    valid_data = get_loader(json_path, data_dir, mask_dir, inp_size,
                            feat_stride, preprocess='resnet', training=False,
                            batch_size=params.batch_size-4*len(params.gpus), shuffle=False, num_workers=4)
    print('val dataset len: {}'.format(len(valid_data.dataset)))

# model
if backbone == 'resnet101':
    model = poseNet(101)
elif backbone == 'resnet50':
    model = poseNet(50)

# load pretrained
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}
if params.ckpt is None:
    model.fpn.load_state_dict(model_zoo.load_url(model_urls[backbone]), strict=False)

trainable_vars = [param for param in model.parameters() if param.requires_grad]
if opt == 'adam':
    print("Training with adam")
    params.optimizer = torch.optim.Adam(
        trainable_vars, lr=params.init_lr, weight_decay=weight_decay)

params.lr_scheduler = ReduceLROnPlateau(params.optimizer, 'min', factor=params.lr_decay, patience=3, verbose=True)
trainer = Trainer(model, params, batch_processor, train_data, valid_data)
trainer.train()
