import torch
import torch.nn as nn

class Head(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Head, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.cls_head = nn.Conv2d(128, num_classes, kernel_size=1)
        self.reg_head = nn.Conv2d(128, 4, kernel_size=1)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        cls_output = self.cls_head(x)
        reg_output = self.reg_head(x)
        return cls_output, reg_output
