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
        
        self.backbone_conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.backbone_conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.relu = nn.ReLU(inplace=True)

        self.backbone_conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.backbone_conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.backbone_conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.backbone_conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.backbone_conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.backbone_conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.backbone_conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.backbone_conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv0 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(512, 36, kernel_size=1, stride=1, padding=0)
        self.conv4_2 = nn.Conv2d(512, 18, kernel_size=3, stride=1, padding=1)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        
        # backbone
        x = self.backbone_conv1_1(x)
        x = self.relu(x)
        x = self.backbone_conv1_2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        print(x.shape)

        x = self.backbone_conv2_1(x)
        x = self.relu(x)
        x = self.backbone_conv2_2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        print(x.shape)

        x = self.backbone_conv3_1(x)
        x = self.relu(x)
        x = self.backbone_conv3_2(x)
        x = self.relu(x)
        x = self.backbone_conv3_3(x)
        x = self.relu(x)
        x = self.maxpool(x)
        print(x.shape)

        x = self.backbone_conv4_1(x)
        x = self.relu(x)
        x = self.backbone_conv4_2(x)
        x = self.relu(x)
        x = self.backbone_conv4_3(x)
        x = self.relu(x)
        x = self.maxpool(x)
        print(x.shape)

        x = self.conv0(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        feature_map = x
        x = self.conv3(x)
        x = self.relu(x)
        print(x.shape)

        x1 = self.conv4_1(x)
        x2 = self.conv4_2(x)  # rpn_cls_score
        x2 = x2.view(1, 2, -1, x2.shape[3])
        x2 = self.softmax(x2)
        x2 = x2.view(1, 18, -1, x2.shape[3])

        rois, _ = F.proposal(x1, x2)

        # pool = F.roipooling(rois, feature_map)

        return None


if __name__ == '__main__':

    input_tensor = torch.randn([1, 3, 600, 800])
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
