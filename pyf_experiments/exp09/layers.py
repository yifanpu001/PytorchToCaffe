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


class conv1x1(nn.Module):
    def __init__(self, layer=10, channels=32):
        super(conv1x1, self).__init__()
      
        layers = []
        for i in range(layer):
            layers.append(nn.Conv2d(channels,channels,1,1,0))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class conv3x3(nn.Module):
    def __init__(self, layer=10, channels=32):
        super(conv3x3, self).__init__()
      
        layers = []
        for i in range(layer):
            layers.append(nn.Conv2d(channels,channels,3,1,1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class conv5x5(nn.Module):
    def __init__(self, layer=10, channels=32):
        super(conv5x5, self).__init__()
      
        layers = []
        for i in range(layer):
            layers.append(nn.Conv2d(channels,channels,5,1,2))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class convt1x1(nn.Module):
    def __init__(self, layer=10, channels=32):
        super(convt1x1, self).__init__()
      
        layers = []
        for i in range(layer):
            layers.append(nn.ConvTranspose2d(channels,channels,1,1,0))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class convt3x3(nn.Module):
    def __init__(self, layer=10, channels=32):
        super(convt3x3, self).__init__()
      
        layers = []
        for i in range(layer):
            layers.append(nn.ConvTranspose2d(channels,channels,3,1,1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class convt5x5(nn.Module):
    def __init__(self, layer=10, channels=32):
        super(convt5x5, self).__init__()
      
        layers = []
        for i in range(layer):
            layers.append(nn.ConvTranspose2d(channels,channels,5,1,2))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class batchnorm(nn.Module):
    def __init__(self, layer=10, channels=32):
        super(batchnorm, self).__init__()
      
        layers = []
        for i in range(layer):
            layers.append(nn.BatchNorm2d(channels))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class relu(nn.Module):
    def __init__(self, layer=10, channels=32):
        super(relu, self).__init__()
      
        layers = []
        for i in range(layer):
            layers.append(nn.ReLU(inplace=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class leakyrelu(nn.Module):
    def __init__(self, layer=10, channels=32):
        super(leakyrelu, self).__init__()
      
        layers = []
        for i in range(layer):
            layers.append(nn.LeakyReLU(inplace=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class maxpool(nn.Module):
    def __init__(self, layer=10, channels=32):
        super(maxpool, self).__init__()
      
        layers = []
        for i in range(layer):
            layers.append(nn.MaxPool2d(3,1,1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class softmax(nn.Module):
    def __init__(self, layer=10, channels=32):
        super(softmax, self).__init__()
      
        layers = []
        for i in range(layer):
            layers.append(nn.Softmax(dim=1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


if __name__ == '__main__':

    input_tensor = Variable(torch.randn([8, 32, 224, 224]))
    model = softmax(layer=10, channels=32)
    name = f'exp09'
    caffe_model_name = f'softmax_b8_s224_l10_c32_r300_fp16'
    model.eval()

    save_path = '/home/pyf/codeforascend/PytorchToCaffe/converted_models'
    # name = f'exp07/ConvolutionLayers'
    print(f'{save_path}/{name}')
    os.system(f'mkdir {save_path}/{name}')

    # caffe_model_name = f'ConvolutionLayers'
    pytorch_to_caffe.trans_net(model, input_tensor, caffe_model_name)
    pytorch_to_caffe.save_prototxt(f'{save_path}/{name}/{caffe_model_name}.prototxt')
    pytorch_to_caffe.save_caffemodel(f'{save_path}/{name}/{caffe_model_name}.caffemodel')