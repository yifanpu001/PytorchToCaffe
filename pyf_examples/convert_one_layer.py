import sys
sys.path.insert(0,'.')
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
import pytorch_to_caffe


class OneLayer(nn.Module):

    def __init__(self):
        super(OneLayer, self).__init__()
        self.layer = nn.Linear(1024, 256)


    def forward(self, x):
        x = self.layer(x)
        return x


if __name__=='__main__':
    name='resnet18'
    net=OneLayer()
    net.eval()
    input=Variable(torch.ones([64, 1024]))
    pytorch_to_caffe.trans_net(net, input, name)
    pytorch_to_caffe.save_prototxt('/home/pyf/codeforascend/PytorchToCaffe/converted_models/{}.prototxt'.format(name))
    pytorch_to_caffe.save_caffemodel('/home/pyf/codeforascend/PytorchToCaffe/converted_models/{}.caffemodel'.format(name))