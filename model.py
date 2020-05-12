import time

import torch
import torchvision
from torch import nn


class BasicModule(nn.Module):

    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))  # 默认名字

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def save(self, name=None):
        if name is None:
            prefix = './checkpoints/' + self.model_name + '_'
            name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def get_optimizer(self, lr, weight_decay, momentum):
        opt = torch.optim.SGD(self.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        # opt = torch.optim.AdamW(self.parameters())
        return opt


class Flat(nn.Module):
    def __init__(self):
        super(Flat, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class mobilenet(BasicModule):
    def __init__(self, num_classes=2):
        super(mobilenet, self).__init__()
        self.model_name = 'mobilenet'

        net = torchvision.models.mobilenet_v2(pretrained=False, num_classes=1000)

        self.features = nn.Sequential(
            *list(net.children())[:-1],
        )
        self.classifer = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        x = self.classifer(x)
        return x
