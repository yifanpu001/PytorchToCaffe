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



class ppyolo(nn.Module):
    def __init__(self):
        super(ppyolo, self).__init__()
        ##############
        # backbone(bb)
        #   layer 0
        self.bb_conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bb_bn1 = nn.BatchNorm2d(64)
        self.bb_relu = nn.ReLU(inplace=True)
        self.bb_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)
        #   layer 1
        #         1 - 0
        self.bb_1_0_conv1 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.bb_1_0_bn1 = nn.BatchNorm2d(64)
        self.bb_1_0_conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bb_1_0_bn2 = nn.BatchNorm2d(64)
        self.bb_1_0_conv3 = nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bb_1_0_bn3 = nn.BatchNorm2d(256)
        self.bb_1_0_relu = nn.ReLU(inplace=True)
        self.bb_1_0_convds = nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bb_1_0_bnds = nn.BatchNorm2d(256)
        #         1 - 1
        self.bb_1_1_conv1 = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.bb_1_1_bn1 = nn.BatchNorm2d(64)
        self.bb_1_1_conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bb_1_1_bn2 = nn.BatchNorm2d(64)
        self.bb_1_1_conv3 = nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bb_1_1_bn3 = nn.BatchNorm2d(256)
        self.bb_1_1_relu = nn.ReLU(inplace=True)
        #         1 - 2
        self.bb_1_2_conv1 = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.bb_1_2_bn1 = nn.BatchNorm2d(64)
        self.bb_1_2_conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bb_1_2_bn2 = nn.BatchNorm2d(64)
        self.bb_1_2_conv3 = nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bb_1_2_bn3 = nn.BatchNorm2d(256)
        self.bb_1_2_relu = nn.ReLU(inplace=True)

        #   layer 2
        #         2 - 0
        self.bb_2_0_conv1 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.bb_2_0_bn1 = nn.BatchNorm2d(128)
        self.bb_2_0_conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bb_2_0_bn2 = nn.BatchNorm2d(128)
        self.bb_2_0_conv3 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bb_2_0_bn3 = nn.BatchNorm2d(512)
        self.bb_2_0_relu = nn.ReLU(inplace=True)
        self.bb_2_0_convds = nn.Conv2d(256, 512, kernel_size=1, stride=2, padding=0, bias=False)
        self.bb_2_0_bnds = nn.BatchNorm2d(512)
        #         2 - 1
        self.bb_2_1_conv1 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.bb_2_1_bn1 = nn.BatchNorm2d(128)
        self.bb_2_1_conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bb_2_1_bn2 = nn.BatchNorm2d(128)
        self.bb_2_1_conv3 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bb_2_1_bn3 = nn.BatchNorm2d(512)
        self.bb_2_1_relu = nn.ReLU(inplace=True)
        #         2 - 2
        self.bb_2_2_conv1 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.bb_2_2_bn1 = nn.BatchNorm2d(128)
        self.bb_2_2_conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bb_2_2_bn2 = nn.BatchNorm2d(128)
        self.bb_2_2_conv3 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bb_2_2_bn3 = nn.BatchNorm2d(512)
        self.bb_2_2_relu = nn.ReLU(inplace=True)
        #         2 - 3
        self.bb_2_3_conv1 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.bb_2_3_bn1 = nn.BatchNorm2d(128)
        self.bb_2_3_conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bb_2_3_bn2 = nn.BatchNorm2d(128)
        self.bb_2_3_conv3 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bb_2_3_bn3 = nn.BatchNorm2d(512)
        self.bb_2_3_relu = nn.ReLU(inplace=True)

        #   layer 3
        #         3 - 0
        self.bb_3_0_conv1 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bb_3_0_bn1 = nn.BatchNorm2d(256)
        self.bb_3_0_conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.bb_3_0_bn2 = nn.BatchNorm2d(256)
        self.bb_3_0_conv3 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bb_3_0_bn3 = nn.BatchNorm2d(1024)
        self.bb_3_0_relu = nn.ReLU(inplace=True)
        self.bb_3_0_convds = nn.Conv2d(512, 1024, kernel_size=1, stride=2, padding=0, bias=False)
        self.bb_3_0_bnds = nn.BatchNorm2d(1024)
        #         3 - 1
        self.bb_3_1_conv1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bb_3_1_bn1 = nn.BatchNorm2d(256)
        self.bb_3_1_conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bb_3_1_bn2 = nn.BatchNorm2d(256)
        self.bb_3_1_conv3 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bb_3_1_bn3 = nn.BatchNorm2d(1024)
        self.bb_3_1_relu = nn.ReLU(inplace=True)
        #         3 - 2
        self.bb_3_2_conv1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bb_3_2_bn1 = nn.BatchNorm2d(256)
        self.bb_3_2_conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bb_3_2_bn2 = nn.BatchNorm2d(256)
        self.bb_3_2_conv3 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bb_3_2_bn3 = nn.BatchNorm2d(1024)
        self.bb_3_2_relu = nn.ReLU(inplace=True)
        #         3 - 3
        self.bb_3_3_conv1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bb_3_3_bn1 = nn.BatchNorm2d(256)
        self.bb_3_3_conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bb_3_3_bn2 = nn.BatchNorm2d(256)
        self.bb_3_3_conv3 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bb_3_3_bn3 = nn.BatchNorm2d(1024)
        self.bb_3_3_relu = nn.ReLU(inplace=True)
        #         3 - 4
        self.bb_3_4_conv1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bb_3_4_bn1 = nn.BatchNorm2d(256)
        self.bb_3_4_conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bb_3_4_bn2 = nn.BatchNorm2d(256)
        self.bb_3_4_conv3 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bb_3_4_bn3 = nn.BatchNorm2d(1024)
        self.bb_3_4_relu = nn.ReLU(inplace=True)
        #         3 - 5
        self.bb_3_5_conv1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bb_3_5_bn1 = nn.BatchNorm2d(256)
        self.bb_3_5_conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bb_3_5_bn2 = nn.BatchNorm2d(256)
        self.bb_3_5_conv3 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bb_3_5_bn3 = nn.BatchNorm2d(1024)
        self.bb_3_5_relu = nn.ReLU(inplace=True)


        #   layer 4
        #         4 - 0
        self.bb_4_0_conv1 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bb_4_0_bn1 = nn.BatchNorm2d(512)
        self.bb_4_0_conv2 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.bb_4_0_bn2 = nn.BatchNorm2d(512)
        self.bb_4_0_conv3 = nn.Conv2d(512, 2048, kernel_size=1, stride=1, padding=0, bias=False)
        self.bb_4_0_bn3 = nn.BatchNorm2d(2048)
        self.bb_4_0_relu = nn.ReLU(inplace=True)
        self.bb_4_0_convds = nn.Conv2d(1024, 2048, kernel_size=1, stride=2, padding=0, bias=False)
        self.bb_4_0_bnds = nn.BatchNorm2d(2048)
        #         4 - 1
        self.bb_4_1_conv1 = nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bb_4_1_bn1 = nn.BatchNorm2d(512)
        self.bb_4_1_conv2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bb_4_1_bn2 = nn.BatchNorm2d(512)
        self.bb_4_1_conv3 = nn.Conv2d(512, 2048, kernel_size=1, stride=1, padding=0, bias=False)
        self.bb_4_1_bn3 = nn.BatchNorm2d(2048)
        self.bb_4_1_relu = nn.ReLU(inplace=True)
        #         4 - 2
        self.bb_4_2_conv1 = nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bb_4_2_bn1 = nn.BatchNorm2d(512)
        self.bb_4_2_conv2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bb_4_2_bn2 = nn.BatchNorm2d(512)
        self.bb_4_2_conv3 = nn.Conv2d(512, 2048, kernel_size=1, stride=1, padding=0, bias=False)
        self.bb_4_2_bn3 = nn.BatchNorm2d(2048)
        self.bb_4_2_relu = nn.ReLU(inplace=True)

        ##############
        # neck: PPFPN
        #   c5 -> p5
        self.fpn_p5_conv1x1_conv = nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.fpn_p5_conv1x1_bn = nn.BatchNorm2d(512)
        self.fpn_p5_conv1x1_leaky_relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.fpn_p5_convblock1_conv1 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1, bias=False)
        self.fpn_p5_convblock1_bn1 = nn.BatchNorm2d(1024)
        self.fpn_p5_convblock1_leaky_relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.fpn_p5_convblock1_conv2 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.fpn_p5_convblock1_bn2 = nn.BatchNorm2d(512)
        self.fpn_p5_convblock1_leaky_relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.fpn_p5_spp_maxpool1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2, dilation=1, ceil_mode=False)
        self.fpn_p5_spp_maxpool2 = nn.MaxPool2d(kernel_size=9, stride=1, padding=4, dilation=1, ceil_mode=False)
        self.fpn_p5_spp_maxpool3 = nn.MaxPool2d(kernel_size=13, stride=1, padding=6, dilation=1, ceil_mode=False)
        self.fpn_p5_spp_conv = nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.fpn_p5_spp_bn = nn.BatchNorm2d(512)
        self.fpn_p5_spp_leaky_relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.fpn_p5_convblock2_conv1 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1, bias=False)
        self.fpn_p5_convblock2_bn1 = nn.BatchNorm2d(1024)
        self.fpn_p5_convblock2_leaky_relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.fpn_p5_convblock2_conv2 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.fpn_p5_convblock2_bn2 = nn.BatchNorm2d(512)
        self.fpn_p5_convblock2_leaky_relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        #   p5 upsample

        self.p5_upsample_conv = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.p5_upsample_bn = nn.BatchNorm2d(256)
        self.p5_upsample_leaky_relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        #   self interpolate


        #   c4 -> p4

        self.fpn_p4_conv1x1_conv = nn.Conv2d(1280, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.fpn_p4_conv1x1_bn = nn.BatchNorm2d(256)
        self.fpn_p4_conv1x1_leaky_relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.fpn_p4_convblock1_conv1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.fpn_p4_convblock1_bn1 = nn.BatchNorm2d(512)
        self.fpn_p4_convblock1_leaky_relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.fpn_p4_convblock1_conv2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.fpn_p4_convblock1_bn2 = nn.BatchNorm2d(256)
        self.fpn_p4_convblock1_leaky_relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.fpn_p4_convblock2_conv1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.fpn_p4_convblock2_bn1 = nn.BatchNorm2d(512)
        self.fpn_p4_convblock2_leaky_relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.fpn_p4_convblock2_conv2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.fpn_p4_convblock2_bn2 = nn.BatchNorm2d(256)
        self.fpn_p4_convblock2_leaky_relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)


        #   p4 upsample

        self.p4_upsample_conv = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.p4_upsample_bn = nn.BatchNorm2d(128)
        self.p4_upsample_leaky_relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)


        #   c3 -> p3

        self.fpn_p3_conv1x1_conv = nn.Conv2d(640, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.fpn_p3_conv1x1_bn = nn.BatchNorm2d(128)
        self.fpn_p3_conv1x1_leaky_relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.fpn_p3_convblock1_conv1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.fpn_p3_convblock1_bn1 = nn.BatchNorm2d(256)
        self.fpn_p3_convblock1_leaky_relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.fpn_p3_convblock1_conv2 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.fpn_p3_convblock1_bn2 = nn.BatchNorm2d(128)
        self.fpn_p3_convblock1_leaky_relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.fpn_p3_convblock2_conv1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.fpn_p3_convblock2_bn1 = nn.BatchNorm2d(256)
        self.fpn_p3_convblock2_leaky_relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.fpn_p3_convblock2_conv2 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.fpn_p3_convblock2_bn2 = nn.BatchNorm2d(128)
        self.fpn_p3_convblock2_leaky_relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        ################
        # head: PPHead
        # for out_4

        self.head_p5_conv = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1, bias=False)
        self.head_p5_bn = nn.BatchNorm2d(1024)
        self.head_p5_leaky_relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.head_p5_linear = nn.Conv2d(1024, 255, kernel_size=1, stride=1, padding=0, bias=False)

        # for out_3

        self.head_p4_conv = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.head_p4_bn = nn.BatchNorm2d(512)
        self.head_p4_leaky_relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.head_p4_linear = nn.Conv2d(512, 255, kernel_size=1, stride=1, padding=0, bias=False)

        # for out_2

        self.head_p3_conv = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.head_p3_bn = nn.BatchNorm2d(256)
        self.head_p3_leaky_relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.head_p3_linear = nn.Conv2d(256, 255, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        # backbone
        #   layer0
        x = self.bb_conv1(x)
        x = self.bb_bn1(x)
        x = self.bb_relu(x)
        x = self.bb_maxpool(x)
        #   layer 1
        #         1 - 0
        identity = self.bb_1_0_convds(x)
        identity = self.bb_1_0_bnds(identity)
        out = self.bb_1_0_conv1(x)
        out = self.bb_1_0_bn1(out)
        out = self.bb_1_0_relu(out)
        out = self.bb_1_0_conv2(out)
        out = self.bb_1_0_bn2(out)
        out = self.bb_1_0_relu(out)
        out = self.bb_1_0_conv3(out)
        out = self.bb_1_0_bn3(out)
        x = out + identity
        x = self.bb_1_0_relu(x)
        #         1 - 1
        identity = x
        out = self.bb_1_1_conv1(x)
        out = self.bb_1_1_bn1(out)
        out = self.bb_1_1_relu(out)
        out = self.bb_1_1_conv2(out)
        out = self.bb_1_1_bn2(out)
        out = self.bb_1_1_relu(out)
        out = self.bb_1_1_conv3(out)
        out = self.bb_1_1_bn3(out)
        x = out + identity
        x = self.bb_1_1_relu(x)
        #         1 - 2
        identity = x
        out = self.bb_1_2_conv1(x)
        out = self.bb_1_2_bn1(out)
        out = self.bb_1_2_relu(out)
        out = self.bb_1_2_conv2(out)
        out = self.bb_1_2_bn2(out)
        out = self.bb_1_2_relu(out)
        out = self.bb_1_2_conv3(out)
        out = self.bb_1_2_bn3(out)
        x = out + identity
        x = self.bb_1_2_relu(x)
        out_1 = x  # .clone()
        #   layer 2
        #         2 - 0
        identity = self.bb_2_0_convds(x)
        identity = self.bb_2_0_bnds(identity)
        out = self.bb_2_0_conv1(x)
        out = self.bb_2_0_bn1(out)
        out = self.bb_2_0_relu(out)
        out = self.bb_2_0_conv2(out)
        out = self.bb_2_0_bn2(out)
        out = self.bb_2_0_relu(out)
        out = self.bb_2_0_conv3(out)
        out = self.bb_2_0_bn3(out)
        x = out + identity
        x = self.bb_2_0_relu(x)
        #         2 - 1
        identity = x
        out = self.bb_2_1_conv1(x)
        out = self.bb_2_1_bn1(out)
        out = self.bb_2_1_relu(out)
        out = self.bb_2_1_conv2(out)
        out = self.bb_2_1_bn2(out)
        out = self.bb_2_1_relu(out)
        out = self.bb_2_1_conv3(out)
        out = self.bb_2_1_bn3(out)
        x = out + identity
        x = self.bb_2_1_relu(x)
        #         2 - 2
        identity = x
        out = self.bb_2_2_conv1(x)
        out = self.bb_2_2_bn1(out)
        out = self.bb_2_2_relu(out)
        out = self.bb_2_2_conv2(out)
        out = self.bb_2_2_bn2(out)
        out = self.bb_2_2_relu(out)
        out = self.bb_2_2_conv3(out)
        out = self.bb_2_2_bn3(out)
        x = out + identity
        x = self.bb_2_2_relu(x)
        #         2 - 3
        identity = x
        out = self.bb_2_3_conv1(x)
        out = self.bb_2_3_bn1(out)
        out = self.bb_2_3_relu(out)
        out = self.bb_2_3_conv2(out)
        out = self.bb_2_3_bn2(out)
        out = self.bb_2_3_relu(out)
        out = self.bb_2_3_conv3(out)
        out = self.bb_2_3_bn3(out)
        x = out + identity
        x = self.bb_2_3_relu(x)
        out_2 = x  # .clone()
        #   layer 3
        #         3 - 0
        identity = self.bb_3_0_convds(x)
        identity = self.bb_3_0_bnds(identity)
        out = self.bb_3_0_conv1(x)
        out = self.bb_3_0_bn1(out)
        out = self.bb_3_0_relu(out)
        out = self.bb_3_0_conv2(out)
        out = self.bb_3_0_bn2(out)
        out = self.bb_3_0_relu(out)
        out = self.bb_3_0_conv3(out)
        out = self.bb_3_0_bn3(out)
        x = out + identity
        x = self.bb_3_0_relu(x)
        #         3 - 1
        identity = x
        out = self.bb_3_1_conv1(x)
        out = self.bb_3_1_bn1(out)
        out = self.bb_3_1_relu(out)
        out = self.bb_3_1_conv2(out)
        out = self.bb_3_1_bn2(out)
        out = self.bb_3_1_relu(out)
        out = self.bb_3_1_conv3(out)
        out = self.bb_3_1_bn3(out)
        x = out + identity
        x = self.bb_3_1_relu(x)
        #         3 - 2
        identity = x
        out = self.bb_3_2_conv1(x)
        out = self.bb_3_2_bn1(out)
        out = self.bb_3_2_relu(out)
        out = self.bb_3_2_conv2(out)
        out = self.bb_3_2_bn2(out)
        out = self.bb_3_2_relu(out)
        out = self.bb_3_2_conv3(out)
        out = self.bb_3_2_bn3(out)
        x = out + identity
        x = self.bb_3_2_relu(x)
        #         3 - 3
        identity = x
        out = self.bb_3_3_conv1(x)
        out = self.bb_3_3_bn1(out)
        out = self.bb_3_3_relu(out)
        out = self.bb_3_3_conv2(out)
        out = self.bb_3_3_bn2(out)
        out = self.bb_3_3_relu(out)
        out = self.bb_3_3_conv3(out)
        out = self.bb_3_3_bn3(out)
        x = out + identity
        x = self.bb_3_3_relu(x)
        #         3 - 4
        identity = x
        out = self.bb_3_4_conv1(x)
        out = self.bb_3_4_bn1(out)
        out = self.bb_3_4_relu(out)
        out = self.bb_3_4_conv2(out)
        out = self.bb_3_4_bn2(out)
        out = self.bb_3_4_relu(out)
        out = self.bb_3_4_conv3(out)
        out = self.bb_3_4_bn3(out)
        x = out + identity
        x = self.bb_3_4_relu(x)
        #         3 - 5
        identity = x
        out = self.bb_3_5_conv1(x)
        out = self.bb_3_5_bn1(out)
        out = self.bb_3_5_relu(out)
        out = self.bb_3_5_conv2(out)
        out = self.bb_3_5_bn2(out)
        out = self.bb_3_5_relu(out)
        out = self.bb_3_5_conv3(out)
        out = self.bb_3_5_bn3(out)
        x = out + identity
        x = self.bb_3_5_relu(x)
        out_3 = x  # .clone()
        #   layer 4
        #         4 - 0
        identity = self.bb_4_0_convds(x)
        identity = self.bb_4_0_bnds(identity)
        out = self.bb_4_0_conv1(x)
        out = self.bb_4_0_bn1(out)
        out = self.bb_4_0_relu(out)
        out = self.bb_4_0_conv2(out)
        out = self.bb_4_0_bn2(out)
        out = self.bb_4_0_relu(out)
        out = self.bb_4_0_conv3(out)
        out = self.bb_4_0_bn3(out)
        x = out + identity
        x = self.bb_4_0_relu(x)
        #         4 - 1
        identity = x
        out = self.bb_4_1_conv1(x)
        out = self.bb_4_1_bn1(out)
        out = self.bb_4_1_relu(out)
        out = self.bb_4_1_conv2(out)
        out = self.bb_4_1_bn2(out)
        out = self.bb_4_1_relu(out)
        out = self.bb_4_1_conv3(out)
        out = self.bb_4_1_bn3(out)
        x = out + identity
        x = self.bb_4_1_relu(x)
        #         4 - 2
        identity = x
        out = self.bb_4_2_conv1(x)
        out = self.bb_4_2_bn1(out)
        out = self.bb_4_2_relu(out)
        out = self.bb_4_2_conv2(out)
        out = self.bb_4_2_bn2(out)
        out = self.bb_4_2_relu(out)
        out = self.bb_4_2_conv3(out)
        out = self.bb_4_2_bn3(out)
        x = out + identity
        x = self.bb_4_2_relu(x)
        out_4 = x  # .clone()

        
        # neck: PPFPN
        #   c5 -> p5

        out_4 = self.fpn_p5_conv1x1_conv(out_4)
        out_4 = self.fpn_p5_conv1x1_bn(out_4)
        out_4 = self.fpn_p5_conv1x1_leaky_relu(out_4)
        
        out_4 = self.fpn_p5_convblock1_conv1(out_4)
        out_4 = self.fpn_p5_convblock1_bn1(out_4)
        out_4 = self.fpn_p5_convblock1_leaky_relu1(out_4)

        out_4 = self.fpn_p5_convblock1_conv2(out_4)
        out_4 = self.fpn_p5_convblock1_bn2(out_4)
        out_4 = self.fpn_p5_convblock1_leaky_relu2(out_4)
        
        out_4_k5 = self.fpn_p5_spp_maxpool1(out_4)
        out_4_k9 = self.fpn_p5_spp_maxpool2(out_4)
        out_4_k13 = self.fpn_p5_spp_maxpool3(out_4)
        out_4 = torch.cat([out_4_k5,
                           out_4_k9, 
                           out_4_k13,
                           out_4],
                          dim=1)
        out_4 = self.fpn_p5_spp_conv(out_4)
        out_4 = self.fpn_p5_spp_bn(out_4)
        out_4 = self.fpn_p5_spp_leaky_relu(out_4)
        
        out_4 = self.fpn_p5_convblock2_conv1(out_4)
        out_4 = self.fpn_p5_convblock2_bn1(out_4)
        out_4 = self.fpn_p5_convblock2_leaky_relu1(out_4)

        out_4 = self.fpn_p5_convblock2_conv2(out_4)
        out_4 = self.fpn_p5_convblock2_bn2(out_4)
        out_4 = self.fpn_p5_convblock2_leaky_relu2(out_4)
        

        #   p5 upsample

        upto4 = self.p5_upsample_conv(out_4)
        upto4 = self.p5_upsample_bn(upto4)
        upto4 = self.p5_upsample_leaky_relu(upto4)
        upto4 = F.interpolate(upto4, scale_factor=2, mode='nearest')


        #   c4 -> p4
        out_3 = torch.cat([out_3, upto4], dim=1)

        out_3 = self.fpn_p4_conv1x1_conv(out_3)
        out_3 = self.fpn_p4_conv1x1_bn(out_3)
        out_3 = self.fpn_p4_conv1x1_leaky_relu(out_3)

        out_3 = self.fpn_p4_convblock1_conv1(out_3)
        out_3 = self.fpn_p4_convblock1_bn1(out_3)
        out_3 = self.fpn_p4_convblock1_leaky_relu1(out_3)

        out_3 = self.fpn_p4_convblock1_conv2(out_3)
        out_3 = self.fpn_p4_convblock1_bn2(out_3)
        out_3 = self.fpn_p4_convblock1_leaky_relu2(out_3)

        out_3 = self.fpn_p4_convblock2_conv1(out_3)
        out_3 = self.fpn_p4_convblock2_bn1(out_3)
        out_3 = self.fpn_p4_convblock2_leaky_relu1(out_3)

        out_3 = self.fpn_p4_convblock2_conv2(out_3)
        out_3 = self.fpn_p4_convblock2_bn2(out_3)
        out_3 = self.fpn_p4_convblock2_leaky_relu2(out_3)


        #   p4 upsample

        upto3 = self.p4_upsample_conv(out_3)
        upto3 = self.p4_upsample_bn(upto3)
        upto3 = self.p4_upsample_leaky_relu(upto3)
        upto3 = F.interpolate(upto3, scale_factor=2, mode='nearest')


        #   c3 -> p3
        out_2 = torch.cat([out_2, upto3], dim=1)

        out_2 = self.fpn_p3_conv1x1_conv(out_2)
        out_2 = self.fpn_p3_conv1x1_bn(out_2)
        out_2 = self.fpn_p3_conv1x1_leaky_relu(out_2)

        out_2 = self.fpn_p3_convblock1_conv1(out_2)
        out_2 = self.fpn_p3_convblock1_bn1(out_2)
        out_2 = self.fpn_p3_convblock1_leaky_relu1(out_2)

        out_2 = self.fpn_p3_convblock1_conv2(out_2)
        out_2 = self.fpn_p3_convblock1_bn2(out_2)
        out_2 = self.fpn_p3_convblock1_leaky_relu2(out_2)

        out_2 = self.fpn_p3_convblock2_conv1(out_2)
        out_2 = self.fpn_p3_convblock2_bn1(out_2)
        out_2 = self.fpn_p3_convblock2_leaky_relu1(out_2)

        out_2 = self.fpn_p3_convblock2_conv2(out_2)
        out_2 = self.fpn_p3_convblock2_bn2(out_2)
        out_2 = self.fpn_p3_convblock2_leaky_relu2(out_2)

        # head
        # for out_4

        out_4 = self.head_p5_conv(out_4)
        out_4 = self.head_p5_bn(out_4)
        out_4 = self.head_p5_leaky_relu(out_4)
        out_4 = self.head_p5_linear(out_4)
        # for out_3

        out_3 = self.head_p4_conv(out_3)
        out_3 = self.head_p4_bn(out_3)
        out_3 = self.head_p4_leaky_relu(out_3)
        out_3 = self.head_p4_linear(out_3)
        # for out_2

        out_2 = self.head_p3_conv(out_2)
        out_2 = self.head_p3_bn(out_2)
        out_2 = self.head_p3_leaky_relu(out_2)
        out_2 = self.head_p3_linear(out_2)
        
        return None

    # def add_2_channels(self, input_tensor):
    #     addition = torch.split(input_tensor, 2, dim=1)[0]
    #     output = torch.cat([input_tensor, addition], dim=1)
    #     return output


if __name__ == '__main__':

    input_tensor = torch.randn([1, 3, 608, 608])
    model = ppyolo()

    out3 = model(input_tensor)
    
    name = f'exp10'
    caffe_model_name = f'ppyolo_weak_pure_608_resnet50'
    model.eval()

    save_path = '/home/pyf/codeforascend/PytorchToCaffe/converted_models'
    print(f'{save_path}/{name}')
    os.system(f'mkdir {save_path}/{name}')

    # caffe_model_name = f'ConvolutionLayers'
    pytorch_to_caffe.trans_net(model, input_tensor, caffe_model_name)
    pytorch_to_caffe.save_prototxt(f'{save_path}/{name}/{caffe_model_name}.prototxt')
    pytorch_to_caffe.save_caffemodel(f'{save_path}/{name}/{caffe_model_name}.caffemodel')
