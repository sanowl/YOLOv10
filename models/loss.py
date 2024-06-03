import torch
import torch.nn as nn

class YOLOLoss(nn.Module):
    def __init__(self, num_classes):
        super(YOLOLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, cls_pred, reg_pred, cls_true, reg_true):
        cls_loss = nn.CrossEntropyLoss()(cls_pred, cls_true)
        reg_loss = nn.SmoothL1Loss()(reg_pred, reg_true)
        return cls_loss + reg_loss
