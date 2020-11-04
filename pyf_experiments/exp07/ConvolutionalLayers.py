import sys
sys.path.insert(0,'.')
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
import pytorch_to_caffe
from torchsummaryX import summary
import csv
import argparse

parser = argparse.ArgumentParser(description='PyTorch To Caffe')
parser.add_argument('--block', type=str, help='block name of the model')
parser.add_argument('--depth_config', nargs='+', type=int, help='depth of the model')
parser.add_argument('--width', type=float, help='width of the model')
parser.add_argument('--input_size', nargs='+', type=int, help='input feature map size of the model')
args = parser.parse_args()


class Conv2d(nn.Module):

    def __init__(self):
        super(Conv2d, self).__init__()
        self.conv2d = nn.Conv2d(16, 33, kernel_size=1, padding=1, stride=2)

    def forward(self, x):
        # x = torch.randn(20, 16, 50, 100)
        x = self.conv2d(x)
        return x


class ConvTranspose2d(nn.Module):

    def __init__(self):
        super(ConvTranspose2d, self).__init__()
        self.convtranspose2d = nn.ConvTranspose2d(16, 33, 3, stride=2)

    def forward(self, x):
        # x = torch.randn(20, 16, 50, 100)
        x = self.convtranspose2d(x)
        return x


class Linear(nn.Module):

    def __init__(self):
        super(Linear, self).__init__()
        self.fc = nn.Linear(20, 30)

    def forward(self, x):
        # x = torch.randn(128, 20)
        x = self.fc(x)
        return x


class Split(nn.Module):

    def __init__(self):
        super(Split, self).__init__()

    def forward(self, x):
        torch.split(x, 2)
        return x


class MaxPool2d(nn.Module):

    def __init__(self):
        super(MaxPool2d, self).__init__()
        self.layer = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        # x = torch.randn(20, 16, 50, 32)
        x = self.layer(x)
        return x

"""
class AvgPool2d(nn.Module):

    def __init__(self):
        super(AvgPool2d, self).__init__()
        self.layer = nn.AvgPool2d((3, 2), stride=(2, 1))

    def forward(self, x):
        # x = torch.randn(20, 16, 50, 32)
        x = self.layer(x)
        return x
"""

class AdaptiveMaxPool2d(nn.Module):

    def __init__(self):
        super(AdaptiveMaxPool2d, self).__init__()
        self.layer = nn.AdaptiveMaxPool2d((5, 7))

    def forward(self, x):
        # x = torch.randn(1, 64, 8, 9)
        x = self.layer(x)
        return x


class Max(nn.Module):

    def __init__(self):
        super(Max, self).__init__()

    def forward(self, x):
        # x = torch.randn(1, 3)
        x = torch.max(x)
        return x


class Cat(nn.Module):

    def __init__(self):
        super(Cat, self).__init__()

    def forward(self, x):
        # x = torch.randn(2, 3)
        x = torch.cat((x, x), 0)
        return x


class Dropout(nn.Module):

    def __init__(self):
        super(Dropout, self).__init__()
        self.layer = nn.Dropout(p=0.2)

    def forward(self, x):
        # x = torch.randn(20, 16)
        x = self.layer(x)
        return x

"""
class Threshold(nn.Module):

    def __init__(self):
        super(Threshold, self).__init__()
        self.layer = nn.Threshold(0.1, 20)

    def forward(self, x):
        # x = torch.randn(2)
        x = self.layer(x)
        return x
"""

class ReLU(nn.Module):

    def __init__(self):
        super(ReLU, self).__init__()
        self.layer = nn.ReLU()

    def forward(self, x):
        # x = torch.randn(1, 3, 7, 7)
        x = self.layer(x)
        return x


class PReLU(nn.Module):

    def __init__(self):
        super(PReLU, self).__init__()
        self.layer = nn.PReLU()

    def forward(self, x):
        # x = torch.randn(1, 3, 7, 7)
        x = self.layer(x)
        return x


class LeakyReLU(nn.Module):

    def __init__(self):
        super(LeakyReLU, self).__init__()
        self.layer = nn.LeakyReLU()

    def forward(self, x):
        # x = torch.randn(1, 3, 7, 7)
        x = self.layer(x)
        return x


class Tanh(nn.Module):

    def __init__(self):
        super(Tanh, self).__init__()
        self.layer = nn.Tanh()

    def forward(self, x):
        # x = torch.randn(1, 3, 7, 7)
        x = self.layer(x)
        return x


class Softmax(nn.Module):

    def __init__(self):
        super(Softmax, self).__init__()
        self.layer = nn.Softmax()

    def forward(self, x):
        # x = torch.randn(2, 3)
        x = self.layer(x)
        return x


class BatchNorm2d(nn.Module):

    def __init__(self):
        super(BatchNorm2d, self).__init__()
        self.layer = nn.BatchNorm2d(100)

    def forward(self, x):
        # x = torch.randn(20, 100, 35, 45)
        x = self.layer(x)
        return x


class InstanceNorm2d(nn.Module):

    def __init__(self):
        super(InstanceNorm2d, self).__init__()
        self.layer = nn.InstanceNorm2d(100)

    def forward(self, x):
        # x = torch.randn(20, 100, 35, 45)
        x = self.layer(x)
        return x


class Interpolate(nn.Module):

    def __init__(self):
        super(Interpolate, self).__init__()

    def forward(self, x):
        # x = torch.randn(20, 100, 35, 45)
        x = F.interpolate(x, scale_factor=8, mode='nearest', align_corners=None)
        return x


class Sigmoid(nn.Module):

    def __init__(self):
        super(Sigmoid, self).__init__()
        self.layer = nn.Sigmoid()

    def forward(self, x):
        # x = torch.randn(20, 100, 35, 45)
        x = self.layer(x)
        return x


class Hardtanh(nn.Module):

    def __init__(self):
        super(Hardtanh, self).__init__()
        self.layer = nn.Hardtanh(-2, 2)

    def forward(self, x):
        # x = torch.randn(20, 100, 35, 45)
        x = self.layer(x)
        return x


class Div(nn.Module):

    def __init__(self):
        super(Div, self).__init__()

    def forward(self, x):
        # x = torch.randn(20, 100, 35, 45)
        x = torch.div(x, 0.5)
        return x


class View(nn.Module):

    def __init__(self):
        super(View, self).__init__()

    def forward(self, x):
        # x = torch.randn(20, 100, 35, 45)
        x = x.view(-1, 8)
        return x


class Mean(nn.Module):

    def __init__(self):
        super(Mean, self).__init__()

    def forward(self, x):
        # x = torch.randn(1, 3)
        x = torch.mean(x)
        return x


class Add(nn.Module):

    def __init__(self):
        super(Add, self).__init__()

    def forward(self, x):
        # x = torch.randn(1, 3)
        x = torch.add(x, 20)
        return x


class Sub(nn.Module):

    def __init__(self):
        super(Sub, self).__init__()

    def forward(self, x):
        # x = torch.randn(1, 3)
        x = torch.sub(x, 20)
        return x


class Mul(nn.Module):

    def __init__(self):
        super(Mul, self).__init__()

    def forward(self, x):
        # x = torch.randn(1, 3)
        x = torch.mul(x, 20)
        return x


class Permute(nn.Module):

    def __init__(self):
        super(Permute, self).__init__()

    def forward(self, x):
        # x = torch.randn(1, 3, 7, 7)
        x = x.permute(1, 0, 3, 2)
        return x


class Contiguous(nn.Module):

    def __init__(self):
        super(Contiguous, self).__init__()

    def forward(self, x):
        # x = torch.randn(1, 3, 7, 7)
        x = x.contiguous()
        return x


class Pow(nn.Module):

    def __init__(self):
        super(Pow, self).__init__()

    def forward(self, x):
        # x = torch.randn(3)
        x = torch.pow(x, 2)
        return x


class Sum(nn.Module):

    def __init__(self):
        super(Sum, self).__init__()

    def forward(self, x):
        # x = torch.randn(3)
        x = torch.sum(x)
        return x


class Sqrt(nn.Module):

    def __init__(self):
        super(Sqrt, self).__init__()

    def forward(self, x):
        # x = torch.randn(3)
        x = torch.sqrt(x)
        return x


class Unsqueeze(nn.Module):

    def __init__(self):
        super(Unsqueeze, self).__init__()

    def forward(self, x):
        # x = torch.randn([1, 2, 3, 4])
        x = torch.unsqueeze(x, 0)
        return x


class Expand(nn.Module):

    def __init__(self):
        super(Expand, self).__init__()

    def forward(self, x):
        # x = torch.randn([[1], [2], [3]])
        x = x.expand(3, 4)
        return x

if __name__ == '__main__':

    input_tensor = Variable(torch.tensor([[1], [2], [3]]))
    model = Expand()
    name = f'exp07/Expand'
    caffe_model_name = f'Expand'
    model.eval()

    moduel_list = [
        'AdaptiveMaxPool2d',
        'BatchNorm2d',
        'Conv2d',
        'ConvTranspose2d',
        'Dropout',
        'Hardtanh',
        'InstanceNorm2d',
        'Interpolate',  # only support mode='nearest', align_corners=None
        'LeakyReLU',
        'Linear',
        'MaxPool2d',
        'PReLU',
        'ReLU',
        'Sigmoid',
        'Softmax',
        'Tanh',
    ]
    tensor_list = [
        'Add',
        'Cat',
        'Contiguous',
        'Div',
        'Expand',
        'Max',
        'Mean',
        'Mul',
        'Permute',  # only for 4 dims tensor
        'Pow',
        'Split',
        'Sqrt',
        'Sub',
        'Sum',
        'Unsqueeze',
        'View',
    ]

    save_path = '/home/pyf/codeforascend/PytorchToCaffe/converted_models'
    # name = f'exp07/ConvolutionLayers'
    print(f'{save_path}/{name}')
    os.system(f'mkdir {save_path}/{name}')

    # caffe_model_name = f'ConvolutionLayers'
    pytorch_to_caffe.trans_net(model, input_tensor, caffe_model_name)
    pytorch_to_caffe.save_prototxt(f'{save_path}/{name}/{caffe_model_name}.prototxt')
    pytorch_to_caffe.save_caffemodel(f'{save_path}/{name}/{caffe_model_name}.caffemodel')

