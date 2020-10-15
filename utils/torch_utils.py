import math
import os
import time
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F


def select_device(device='', apex=False):
    # device = 'cpu' or '0' or '0,1,2,3'
    cpu_request = device.lower() == 'cpu'
    cuda = False if cpu_request else torch.cuda.is_available()
    if cuda:
        c = 1024 ** 2  # bytes to MB
        x = torch.cuda.get_device_properties(int(device))
        s = 'Using CUDA ' + ('Apex ' if apex else '')  # apex for mixed precision
        print("%sdevice:%g '%s' Memory=%dMB" % (s, int(device), x.name, x.total_memory / c))
    else:
        print('Using CPU')
    return torch.device('cuda' if cuda else 'cpu')


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-4
            m.momentum = 0.03
        elif t in [nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
            m.inplace = True


def find_modules(model, mclass=nn.Conv2d):
    # finds layer indices matching module class 'mclass'
    return [i for i, m in enumerate(model.module_list) if isinstance(m, mclass)]


def fuse_conv_and_bn(conv, bn):
    # https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    with torch.no_grad():
        # init
        fusedconv = torch.nn.Conv2d(conv.in_channels,
                                    conv.out_channels,
                                    kernel_size=conv.kernel_size,
                                    stride=conv.stride,
                                    padding=conv.padding,
                                    bias=True)

        # prepare filters
        w_conv = conv.weight.clone().view(conv.out_channels, -1)
        w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
        fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.size()))

        # prepare spatial bias
        if conv.bias is not None:
            b_conv = conv.bias
        else:
            b_conv = torch.zeros(conv.weight.size(0))
        b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
        fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

        return fusedconv




def load_classifier(name='resnet101', n=2):
    # Loads a pretrained model reshaped to n-class output
    import pretrainedmodels
    model = pretrainedmodels.__dict__[name](num_classes=1000, pretrained='imagenet')

    # Display model properties
    for x in ['model.input_size', 'model.input_space', 'model.input_range', 'model.mean', 'model.std']:
        print(x + ' =', eval(x))

    # Reshape output to n classes
    filters = model.last_linear.weight.shape[1]
    model.last_linear.bias = torch.nn.Parameter(torch.zeros(n))
    model.last_linear.weight = torch.nn.Parameter(torch.zeros(n, filters))
    model.last_linear.out_features = n
    return model


def scale_img(img, ratio=1.0, same_shape=True):  # img(16,3,256,416), r=ratio
    # scales img(bs,3,y,x) by ratio
    h, w = img.shape[2:]
    s = (int(h * ratio), int(w * ratio))  # new size
    img = F.interpolate(img, size=s, mode='bilinear', align_corners=False)  # resize
    if not same_shape:  # pad/crop img
        gs = 64  # (pixels) grid size
        h, w = [math.ceil(x * ratio / gs) * gs for x in (h, w)]
    return F.pad(img, [0, w - s[1], 0, h - s[0]], value=0.447)  # value = imagenet mean


class ModelEMA:
    """ Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    A smoothed version of the weights is necessary for some training schemes to perform well.
    E.g. Google's hyper-params for training MNASNet, MobileNet-V3, EfficientNet, etc that use
    RMSprop with a short 2.4-3 epoch decay period and slow LR decay rate of .96-.99 requires EMA
    smoothing of weights to match results. Pay attention to the decay constant you are using
    relative to your update count per epoch.
    To keep EMA from using GPU resources, set device='cpu'. This will save a bit of memory but
    disable validation of the EMA weights. Validation will have to be done manually in a separate
    process, or after the training stops converging.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    I've tested with the sequence in my own train.py for torch.DataParallel, apex.DDP, and single-GPU.
    """

    def __init__(self, model, decay=0.9999, device=''):
        # make a copy of the model for accumulating moving average of weights
        self.ema = deepcopy(model)
        self.ema.eval()
        self.updates = 0  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))  # decay exponential ramp (to help early epochs)
        self.device = device  # perform ema on different device from model if set
        if device:
            self.ema.to(device=device)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        self.updates += 1
        d = self.decay(self.updates)
        with torch.no_grad():
            if type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel):
                msd, esd = model.module.state_dict(), self.ema.module.state_dict()
            else:
                msd, esd = model.state_dict(), self.ema.state_dict()

            for k, v in esd.items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1. - d) * msd[k].detach()

    def update_attr(self, model):
        # Assign attributes (which may change during training)
        for k in model.__dict__.keys():
            if not k.startswith('_'):
                setattr(self.ema, k, getattr(model, k))
