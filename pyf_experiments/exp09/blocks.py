import sys
sys.path.insert(0,'.')
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
import pytorch_to_caffe
from torchsummaryX import summary
import csv
import argparse


# resnet block
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class basicblock(nn.Module):
    def __init__(self, layer=10, channels=32):
        super(basicblock, self).__init__()
      
        layers = []
        for i in range(layer):
            layers.append(BasicBlock(inplanes=channels, planes=channels, stride=1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class bottleneck(nn.Module):
    def __init__(self, layer=10, channels=32):
        super(bottleneck, self).__init__()
      
        layers = []
        for i in range(layer):
            layers.append(Bottleneck(inplanes=channels, planes=int(channels/4)))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


# MobileNetv2 block
class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, norm_layer=None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            norm_layer(out_planes),
            nn.ReLU(inplace=True)
        )

class DepthwiseConv(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, norm_layer=None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        super(DepthwiseConv, self).__init__(
            ConvBNReLU(in_planes, in_planes, kernel_size, stride, groups=in_planes, norm_layer=norm_layer),
            ConvBNReLU(in_planes, out_planes, 1, 1, norm_layer=norm_layer)
        )

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, norm_layer=None):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class convbnrelu(nn.Module):
    def __init__(self, layer=10, channels=32):
        super(convbnrelu, self).__init__()
      
        layers = []
        for i in range(layer):
            layers.append(ConvBNReLU(in_planes=channels, out_planes=channels, kernel_size=3))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class invertedresidual(nn.Module):
    def __init__(self, layer=10, channels=32):
        super(invertedresidual, self).__init__()
      
        layers = []
        for i in range(layer):
            layers.append(InvertedResidual(inp=channels, oup=channels, stride=1, expand_ratio=6))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


# ResNext 
class bottleneck_resnext(nn.Module):
    def __init__(self, layer=1, channels=32):
        super(bottleneck_resnext, self).__init__()
      
        layers = []
        for i in range(layer):
            downsample = nn.Sequential(
                conv1x1(channels, channels * Bottleneck.expansion, stride=1),
                nn.BatchNorm2d(channels * Bottleneck.expansion),
            )
            layers.append(Bottleneck(inplanes=channels, planes=channels, groups=32, base_width=4, downsample=downsample))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class depthwise_conv(nn.Module):
    def __init__(self, layer=1, channels=32):
        super(depthwise_conv, self).__init__()
      
        layers = []
        for i in range(layer):
            layers.append(DepthwiseConv(channels, channels))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


if __name__ == '__main__':

    input_tensor = Variable(torch.randn([8, 64, 224, 224]))
    model = invertedresidual(layer=1, channels=64)
    name = f'exp09/blocks'
    caffe_model_name = f'invertedresidual_b8_s224_l1_c64_r300_fp16'
    model.eval()

    save_path = '/home/pyf/codeforascend/PytorchToCaffe/converted_models'
    # name = f'exp07/ConvolutionLayers'
    print(f'{save_path}/{name}')
    os.system(f'mkdir {save_path}/{name}')

    # caffe_model_name = f'ConvolutionLayers'
    pytorch_to_caffe.trans_net(model, input_tensor, caffe_model_name)
    pytorch_to_caffe.save_prototxt(f'{save_path}/{name}/{caffe_model_name}.prototxt')
    pytorch_to_caffe.save_caffemodel(f'{save_path}/{name}/{caffe_model_name}.caffemodel')