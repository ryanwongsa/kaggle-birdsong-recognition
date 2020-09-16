import torch
import torch.nn as nn
import torchvision.models as models

class DenseNet161(nn.Module):
    def __init__(self, pretrained=True, num_classes=6):
        super().__init__()
        self.densenet = models.densenet161(pretrained=pretrained)
        self.densenet.classifier = torch.nn.Linear(2208, num_classes)
        
        self.num_classes = num_classes

    def forward(self, x):  # batch_size, 3, a, b
        bs, seq, c, h, w = x.shape
        x = x.reshape(bs*seq,c,h,w)
        x = self.densenet(x)
        x = x.reshape(bs, seq, self.num_classes)
        return x