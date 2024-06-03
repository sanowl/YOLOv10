import torch
import torch.nn as nn
from models.backbone import Backbone
from models.head import Head

class YOLOv10(nn.Module):
    def __init__(self, num_classes):
        super(YOLOv10, self).__init__()
        self.backbone = Backbone()
        self.one_to_many_head = Head(512, num_classes)
        self.one_to_one_head = Head(512, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        cls_output_1, reg_output_1 = self.one_to_many_head(features)
        cls_output_2, reg_output_2 = self.one_to_one_head(features)
        return cls_output_1, reg_output_1, cls_output_2, reg_output_2
