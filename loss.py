import torch
import torch.nn as nn

class YOLOLoss(nn.Module):

    def __init__(self, classes, bboxes):
        self.classes = classes
        self.bboxes = bboxes

    def compute_iou(self, box1, box2):
        pass

    def forward(self):
        pass
