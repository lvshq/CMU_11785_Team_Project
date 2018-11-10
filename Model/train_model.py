import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.nn import Parameter

class Flatten(nn.Module):
    def forward(self, x):
        batch, channel, height, width = x.size()
        return x.view(batch, channel*height*width)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, 11, 4)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(3,2)
        self.lrn_norm1 = nn.LocalResponseNorm(5)

        self.conv2 = nn.Conv2d(96,256,5,1,2)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(3,2)
        self.lrn_norm2 = nn.LocalResponseNorm(5)

        self.conv3 = nn.Conv2d(256, 384, 3, 1, 1)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(384, 384, 3, 1, 1, groups=2)
        self.relu4 = nn.ReLU(inplace=True)

        self.conv5 = nn.Conv2d(384, 256, 3, 1, 1, groups=2)
        self.relu5 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(3, 2)

        self.flatten = Flatten()
        fc_input_neurons = 2304
        self.fc6 = nn.Linear(fc_input_neurons, 4096)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout(0.5)

        self.fc7 = nn.Linear(4096, 4096)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout(0.5)

        self.fc8 = nn.Linear(4096, 128)
        self.fc9 = nn.Linear(128, 10)
        pass


    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.lrn_norm1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.lrn_norm2(x)

        x = self.conv3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.relu4(x)

        x = self.conv5(x)
        x = self.relu5(x)
        x = self.pool5(x)

        x = self.flatten(x)
        x = self.fc6(x)
        x = self.relu6(x)
        x = self.drop6(x)

        x = self.fc7(x)
        x = self.relu7(x)
        x = self.drop7(x)

        x = self.fc8(x)
        x = self.fc9(x)

        return x

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data,0,0.001)
                m.bias.data.zero_()
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0, 0.01)
                m.bias.data.zero_()

