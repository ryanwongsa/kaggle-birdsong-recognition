import torch
import torch.nn as nn
from torchvision.models import densenet161, densenet121
from collections import namedtuple

class SkipBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(SkipBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor

    def forward(self, x):
        if self.scale_factor >= 2:
            x = F.avg_pool2d(x, self.scale_factor)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
class AuxBlock(nn.Module):
    def __init__(self, last_fc, num_classes, base_size, dropout):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(base_size*8, base_size*last_fc),
            nn.PReLU(),
            nn.BatchNorm1d(base_size*last_fc),
            nn.Dropout(dropout/2),
            nn.Linear(base_size*last_fc, num_classes),
        )

    def forward(self, x):
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
# Source: https://github.com/luuuyi/CBAM.PyTorch
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


# Source: https://github.com/luuuyi/CBAM.PyTorch
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class ConvolutionalBlockAttentionModule(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(ConvolutionalBlockAttentionModule, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, input):
        out = self.ca(input) * input
        out = self.sa(out) * out
        return out


class AUXDense161(torch.nn.Module):
    def __init__(self, base_size=64, last_fc=2, num_classes=264, dropout=0.2, ratio=16, kernel_size=7, last_filters=8):
        super(AUXDense161, self).__init__()
        features = list(densenet161(pretrained = True).features)
        self.features = nn.ModuleList(features)
        self.skip1 = SkipBlock(in_channels=base_size*6, out_channels=base_size*8,
                               scale_factor=8)
        self.skip2 = SkipBlock(in_channels=base_size *6* 2, out_channels=base_size*8,
                               scale_factor=4)
        self.skip3 = SkipBlock(in_channels=base_size *33, out_channels=base_size*8,
                               scale_factor=2)
        
        self.skip4 = SkipBlock(in_channels=2208, out_channels=base_size*8,
                               scale_factor=1)
        
        self.aux1 = AuxBlock(last_fc, num_classes, base_size, dropout)
        self.aux2 = AuxBlock(last_fc, num_classes, base_size, dropout)
        self.aux3 = AuxBlock(last_fc, num_classes, base_size, dropout)
        
        self.attention = ConvolutionalBlockAttentionModule(base_size*8*4,
                                                           ratio=ratio,
                                                           kernel_size=kernel_size)
        self.merge = SkipBlock(base_size*8*4, base_size*last_filters, 1)

        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(base_size*last_filters, base_size*last_fc),
            nn.PReLU(),
            nn.BatchNorm1d(base_size*last_fc),
            nn.Dropout(dropout/2),
            nn.Linear(base_size*last_fc, num_classes),
        )
        self.num_classes = num_classes
        
        
    def forward(self, x):
        bs, s, c, h, w = x.shape
        x = x.reshape(bs*s, c, h,w)
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii == 4:
                skip1 = self.skip1(x)
                aux1 = self.aux1(skip1)
            elif ii == 6:
                skip2 = self.skip2(x)
                aux2 = self.aux2(skip2)
            elif ii == 8:
                skip3 = self.skip3(x)
                aux3 = self.aux3(skip3)
        x = self.skip4(x)
        
        x = torch.cat([x, skip1, skip2, skip3], dim=1)
        
        x = self.attention(x)
        x = self.merge(x)

        x = F.adaptive_avg_pool2d(x,1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = x.reshape(bs, s, self.num_classes)
        aux3 = aux3.reshape(bs, s, self.num_classes)
        aux2 = aux2.reshape(bs, s, self.num_classes)
        aux1 = aux1.reshape(bs, s, self.num_classes)
        return x, aux3, aux2, aux1