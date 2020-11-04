import sys
sys.path.insert(0,'.')
import os
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import pytorch_to_caffe
import torch.nn.functional as F

import csv


### pdn
def convert_NewLGC_NewTLGC_pdn_a_g8(args):
    args.stages = '2-4-6-8-4'
    args.growth = '8-8-16-32-64'
    args.stages = list(map(int, args.stages.split('-')))
    args.growth = list(map(int, args.growth.split('-')))
    args.condense_factor = 8
    args.trans_factor = 8
    args.group_1x1 = 8
    args.group_3x3 = 8
    args.group_trans = 8
    args.bottleneck = 4
    args.dataset = 'imagenet'
    args.num_classes = 1000
    return ConvertLGCTLGCPDN(args)


def convert_NewLGC_NewTLGC_pdn_a_g4(args):
    args.stages = '2-4-6-8-4'
    args.growth = '8-8-16-32-64'
    args.stages = list(map(int, args.stages.split('-')))
    args.growth = list(map(int, args.growth.split('-')))
    args.condense_factor = 4
    args.trans_factor = 4
    args.group_1x1 = 4
    args.group_3x3 = 4
    args.group_trans = 4
    args.bottleneck = 4
    args.dataset = 'imagenet'
    args.num_classes = 1000
    return ConvertLGCTLGCPDN(args)


def convert_NewLGC_NewTLGC_pdn_a_g2(args):
    args.stages = '2-4-6-8-4'
    args.growth = '8-8-16-32-64'
    args.stages = list(map(int, args.stages.split('-')))
    args.growth = list(map(int, args.growth.split('-')))
    args.condense_factor = 2
    args.trans_factor = 2
    args.group_1x1 = 2
    args.group_3x3 = 2
    args.group_trans = 2
    args.bottleneck = 4
    args.dataset = 'imagenet'
    args.num_classes = 1000
    return ConvertLGCTLGCPDN(args)


def convert_NewLGC_NewTLGC_pdn_a_g1(args):
    args.stages = '2-4-6-8-4'
    args.growth = '8-8-16-32-64'
    args.stages = list(map(int, args.stages.split('-')))
    args.growth = list(map(int, args.growth.split('-')))
    args.condense_factor = 1
    args.trans_factor = 1
    args.group_1x1 = 1
    args.group_3x3 = 1
    args.group_trans = 1
    args.bottleneck = 4
    args.dataset = 'imagenet'
    args.num_classes = 1000
    return ConvertLGCTLGCPDN(args)


# ---------------------------Mudules------------------------------
# class HS(nn.Module):

#     def __init__(self):
#         super(HS, self).__init__()
#         self.relu6 = nn.ReLU6(inplace=True)

#     def forward(self, inputs):
#         return inputs * self.relu6(inputs + 3) / 6


def ShuffleLayer(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    ### reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)
    ### transpose
    x = torch.transpose(x, 1, 2).contiguous()
    ### reshape
    x = x.view(batchsize, -1, height, width)
    return x


def ShuffleLayerTrans(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    ### reshape
    x = x.view(batchsize, channels_per_group, groups, height, width)
    ### transpose
    x = torch.transpose(x, 1, 2).contiguous()
    ### reshape
    x = x.view(batchsize, -1, height, width)
    return x


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, groups=1, activation='ReLU'):
        super(Conv, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(in_channels))
        if activation == 'ReLU':
            # self.add_module('relu', nn.ReLU(inplace=True))
            self.add_module('activation', nn.ReLU(inplace=True))
        else:
            pass
        self.add_module('conv', nn.Conv2d(in_channels, out_channels,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          padding=padding, bias=False,
                                          groups=groups))


class CondenseLinear(nn.Module):
    def __init__(self, in_features, out_features, drop_rate=0.5):
        super(CondenseLinear, self).__init__()
        self.in_features = int(in_features * (1 - drop_rate))
        self.out_features = out_features
        self.linear = nn.Linear(self.in_features, self.out_features)
        self.register_buffer('index', torch.LongTensor(self.in_features))

    def forward(self, x):
        x = torch.index_select(x, 1, self.index)
        x = self.linear(x)
        return x


class CondenseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, groups=1):
        super(CondenseConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.norm = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(self.in_channels, self.out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              groups=self.groups,
                              bias=False)
        self.register_buffer('index', torch.LongTensor(self.in_channels))
        self.index.fill_(0)

    def forward(self, x):
        # x = torch.index_select(x, 1, self.index)
        # x = x[:, self.index, :, :]
        x = self.norm(x)
        x = self.relu(x)
        x = self.conv(x)
        x = ShuffleLayer(x, self.groups)
        return x


class CondenseConvTransIncludeUpdate(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, groups=1):
        super(CondenseConvTransIncludeUpdate, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.norm = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(self.in_channels, self.out_channels,
                              kernel_size=kernel_size,
                              padding=padding,
                              groups=self.groups,
                              bias=False,
                              stride=stride)
        self.register_buffer('index', torch.LongTensor(self.groups, self.out_channels // self.groups))
        self.index.fill_(0)

    def forward(self, x, xl):
        y = self.norm(xl)
        y = self.relu(y)
        y = ShuffleLayerTrans(y, self.groups)
        y = self.conv(y) # SIZE: N, C, H, W
        # for i in range(self.groups):
        #     x[:, self.index[i, :], :, :] += y[:, i * (self.out_channels // self.groups):
        #                              (i + 1) * (self.out_channels // self.groups), :, :]
        return y


class _Transition(nn.Module):
    def __init__(self, in_channels, args):
        super(_Transition, self).__init__()
        # self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.pool = nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=2, groups=in_channels)
    def forward(self, x):
        x = self.pool(x)
        return x


# ---------------------------PDN------------------------------
class _ConvertLGCTLGCPDNDenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, args):
        super(_ConvertLGCTLGCPDNDenseLayer, self).__init__()
        self.group_1x1 = args.group_1x1
        self.group_3x3 = args.group_3x3
        self.group_trans = args.group_trans
        ### 1x1 conv i --> b*k
        self.conv_1 = CondenseConv(in_channels, args.bottleneck * growth_rate,
                                   kernel_size=1, groups=self.group_1x1)
        ### 3x3 conv b*k --> k
        self.conv_2 = Conv(args.bottleneck * growth_rate, growth_rate,
                           kernel_size=3, padding=1, groups=self.group_3x3)
        ### 1x1 res conv k(8-16-32)--> i (k*l)
        self.res = CondenseConvTransIncludeUpdate(growth_rate, in_channels, kernel_size=1,
                                                  groups=self.group_trans)

    def forward(self, x):
        x_ = x
        x = self.conv_1(x)
        x = self.conv_2(x)
        x_ = self.res(x_, x)
        return torch.cat([x_, x], 1)


class _ConvertLGCTLGCPDNDenseBlock(nn.Sequential):
    def __init__(self, num_layers, in_channels, growth_rate, args):
        super(_ConvertLGCTLGCPDNDenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _ConvertLGCTLGCPDNDenseLayer(in_channels + i * growth_rate, growth_rate, args)
            self.add_module('denselayer_%d' % (i + 1), layer)


class ConvertLGCTLGCPDN(nn.Module):
    def __init__(self, args):

        super(ConvertLGCTLGCPDN, self).__init__()

        self.stages = args.stages
        self.growth = args.growth
        assert len(self.stages) == len(self.growth)
        self.args = args
        self.progress = 0.0
        if args.dataset in ['cifar10', 'cifar100']:
            self.init_stride = 1
            self.pool_size = 8
        else:
            self.init_stride = 2
            self.pool_size = 7

        self.features = nn.Sequential()
        ### Initial nChannels should be 3
        self.num_features = 2 * self.growth[0]
        ### Dense-block 1 (224x224)
        self.features.add_module('init_conv', nn.Conv2d(3, self.num_features,
                                                        kernel_size=3,
                                                        stride=self.init_stride,
                                                        padding=1,
                                                        bias=False))
        for i in range(len(self.stages)):
            ### Dense-block i
            self.add_block(i)
        ### Linear layer
        # self.classifier = nn.Linear(self.num_features, args.num_classes)

        ### initialize
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
        return

    def add_block(self, i):
        ### Check if ith is the last one
        last = (i == len(self.stages) - 1)
        block = _ConvertLGCTLGCPDNDenseBlock(
            num_layers=self.stages[i],
            in_channels=self.num_features,
            growth_rate=self.growth[i],
            args=self.args,
        )
        self.features.add_module('denseblock_%d' % (i + 1), block)
        self.num_features += self.stages[i] * self.growth[i]
        if not last:
            trans = _Transition(in_channels=self.num_features,
                                args=self.args)
            self.features.add_module('transition_%d' % (i + 1), trans)
        else:
            self.features.add_module('norm_last',
                                     nn.BatchNorm2d(self.num_features))
            self.features.add_module('relu_last',
                                     nn.ReLU(inplace=True))
            # self.features.add_module('pool_last',
            #                          nn.AvgPool2d(self.pool_size))

    def forward(self, x, progress=None):
        features = self.features(x)
        # out = features.view(features.size(0), -1)
        # out = self.classifier(out)
        return features


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Condensed Convolutional Networks')
    args = parser.parse_args()

    model = convert_NewLGC_NewTLGC_pdn_a_g1(args)
    # model = convert_NewLGC_NewTLGC_pdn_a_g2(args)
    # model = convert_NewLGC_NewTLGC_pdn_a_g4(args)
    # model = convert_NewLGC_NewTLGC_pdn_a_g8(args)


    """ main program """

    model.eval()
    input_tensor = torch.ones([32, 3, 224, 224])

    save_path = '/home/pyf/codeforascend/PytorchToCaffe/converted_models/'
    name = f'jhj_01/pdn_a_g1'
    print(f'{save_path}/{name}')
    os.system(f'mkdir {save_path}/{name}')
    with open(f'{save_path}/{name}/model_config.txt', 'w') as fd:
            fd.write(model.__repr__() + '\n')
    

    caffe_model_name = f'pdn_a_g1'

    pytorch_to_caffe.trans_net(model, input_tensor, caffe_model_name)
    pytorch_to_caffe.save_prototxt(f'{save_path}/{name}/{caffe_model_name}.prototxt')
    pytorch_to_caffe.save_caffemodel(f'{save_path}/{name}/{caffe_model_name}.caffemodel')