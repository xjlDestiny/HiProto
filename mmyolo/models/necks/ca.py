from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmyolo.registry import MODELS
from mmyolo.models.layers.yolo_bricks import CSPLayerWithTwoConv
from mmyolo.models.utils import make_divisible, make_round
from mmyolo.models.plugins.coordatt import CoordAtt

# from mmcv.ops import SAConv2d
# from mmcv.cnn import ConvTranspose2d


@MODELS.register_module()
class CA(nn.Module):
    def __init__(self, widen_factor=1, in_channels=[256, 512, 1024], out_channels=[256, 512, 1024]):
        super(CA, self).__init__()

        in_channels = [make_divisible(i, widen_factor) for i in in_channels]
        out_channels = [make_divisible(i, widen_factor) for i in out_channels]

        self.ca1 = CoordAtt(out_channels[0], out_channels[0])
        self.ca2 = CoordAtt(out_channels[1], out_channels[1])
        self.ca3 = CoordAtt(out_channels[2], out_channels[2])


    def forward(self, x):
        x1, x2, x3 = x

        out1 = self.ca1(x1)
        out2 = self.ca2(x2)
        out3 = self.ca3(x3)

        return tuple([out1, out2, out3])
