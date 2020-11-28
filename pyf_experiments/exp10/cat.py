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


class Cat(nn.Module):
    def __init__(self):
        super(Cat, self).__init__()
    
    def forward(self, x):
        addition = torch.split(x, 2, dim=1)[0]
        print(addition.shape)
        x = torch.cat([x, addition], dim=1)
        return x


if __name__ == '__main__':

    input_tensor = torch.randn([1, 2048, 24, 24])
    model = Cat()

    out3 = model(input_tensor)
    
    name = f'exp10'
    caffe_model_name = f'cat'
    model.eval()

    save_path = '/home/pyf/codeforascend/PytorchToCaffe/converted_models'
    print(f'{save_path}/{name}')
    os.system(f'mkdir {save_path}/{name}')

    # caffe_model_name = f'ConvolutionLayers'
    pytorch_to_caffe.trans_net(model, input_tensor, caffe_model_name)
    pytorch_to_caffe.save_prototxt(f'{save_path}/{name}/{caffe_model_name}.prototxt')
    pytorch_to_caffe.save_caffemodel(f'{save_path}/{name}/{caffe_model_name}.caffemodel')