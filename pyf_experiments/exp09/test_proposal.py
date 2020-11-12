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

class toy_fasterrcnn(nn.Module):

    def __init__(self):
        super(toy_fasterrcnn, self).__init__()
        self.conv1 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)

        self.conv2_1 = nn.Conv2d(512, 36, kernel_size=1, stride=1, padding=0)
        self.conv2_2 = nn.Conv2d(512, 18, kernel_size=3, stride=1, padding=1)

        # self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)

        # self.conv4 = nn.Conv2d(256, 255, kernel_size=3, stride=1, padding=1)
        # self.conv5 = nn.Conv2d(255, 255, kernel_size=3, stride=2, padding=1)
        # self.conv6 = nn.Conv2d(255, 255, kernel_size=3, stride=2, padding=1)


    def forward(self, x):

        x = self.conv1(x)

        x1 = self.conv2_1(x)
        x2 = self.conv2_2(x)

        rois, actual_rois_num = F.proposal(x1, x2)

        return rois, actual_rois_num


if __name__ == '__main__':

    input_tensor = torch.randn([1, 512, 38, 50])
    model = toy_fasterrcnn()

    out3 = model(input_tensor)
    
    name = f'exp09'
    caffe_model_name = f'toy_fasterrcnn'
    model.eval()

    save_path = '/home/pyf/codeforascend/PytorchToCaffe/converted_models'
    # name = f'exp07/ConvolutionLayers'
    print(f'{save_path}/{name}')
    os.system(f'mkdir {save_path}/{name}')

    # caffe_model_name = f'ConvolutionLayers'
    pytorch_to_caffe.trans_net(model, input_tensor, caffe_model_name)
    pytorch_to_caffe.save_prototxt(f'{save_path}/{name}/{caffe_model_name}.prototxt')
    pytorch_to_caffe.save_caffemodel(f'{save_path}/{name}/{caffe_model_name}.caffemodel')