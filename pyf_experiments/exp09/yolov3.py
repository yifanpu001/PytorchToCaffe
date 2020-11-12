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

class toy_yolov3(nn.Module):

    def __init__(self):
        super(toy_yolov3, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, stride=2, padding=1)

        self.conv2_1 = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)

        self.conv4 = nn.Conv2d(256, 255, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(255, 255, kernel_size=3, stride=2, padding=1)
        self.conv6 = nn.Conv2d(255, 255, kernel_size=3, stride=2, padding=1)


    def forward(self, x):
        # [  3, 104, 104]
        x = self.conv1(x)
        identity = x
        # [128,  52,  52]
        out = self.conv2_1(x)
        out = self.conv2_2(out)
        out += identity
        # [128,  52,  52]
        out = self.conv3(out)
        # [128,  26,  26]
        out = F.interpolate(out, scale_factor=2)
        # [128,  52,  52]
        out = torch.cat((out, identity), dim=1)

        out1 = self.conv4(out)
        out2 = self.conv5(out1)
        out3 = self.conv6(out2)

        return out3


if __name__ == '__main__':

    input_tensor = torch.randn([1, 3, 104, 104])
    model = toy_yolov3()

    out3 = model(input_tensor)
    
    name = f'exp09'
    caffe_model_name = f'toy_yolov3'
    model.eval()

    save_path = '/home/pyf/codeforascend/PytorchToCaffe/converted_models'
    # name = f'exp07/ConvolutionLayers'
    print(f'{save_path}/{name}')
    os.system(f'mkdir {save_path}/{name}')

    # caffe_model_name = f'ConvolutionLayers'
    pytorch_to_caffe.trans_net(model, input_tensor, caffe_model_name)
    pytorch_to_caffe.save_prototxt(f'{save_path}/{name}/{caffe_model_name}.prototxt')
    pytorch_to_caffe.save_caffemodel(f'{save_path}/{name}/{caffe_model_name}.caffemodel')