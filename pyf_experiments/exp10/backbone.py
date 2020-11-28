import torch
import torch.nn as nn

class ppyolo(nn.Module):
    def __init__(self):
        super(ppyolo, self).__init__()
        ##############
        # backbone(bb)
        #   layer 0
        self.bb_conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bb_bn1 = nn.BatchNorm2d(64)
        self.bb_relu = nn.ReLU(inplace=True)
        self.bb_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
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

        # head: PPHead


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
        out_1 = x.clone()
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
        out_2 = x.clone()
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
        out_3 = x.clone()
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
        out_4 = x.clone()

        return out_4

if __name__ == '__main__':
    input_tensor = torch.randn([1, 3, 224, 224])
    model = ppyolo()
    model(input_tensor)