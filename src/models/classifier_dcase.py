import torch
import torch.nn as nn
import torch.nn.functional as F

def init_layer(layer, nonlinearity='leaky_relu'):
    """Initialize a Linear or Convolutional layer. """
    nn.init.kaiming_uniform_(layer.weight, nonlinearity=nonlinearity)

    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


def init_bn(bn):
    """Initialize a Batchnorm layer. """

    bn.bias.data.fill_(0.)
    bn.running_mean.data.fill_(0.)
    bn.weight.data.fill_(1.)
    bn.running_var.data.fill_(1.)

# Attention Layers
class SpatialAttention2d(nn.Module):
    def __init__(self, channel):
        super(SpatialAttention2d, self).__init__()
        self.squeeze = nn.Conv2d(channel, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        z = self.squeeze(x)
        z = self.sigmoid(z)
        return x * z

class GAB(nn.Module):
    def __init__(self, input_dim, reduction=4):
        super(GAB, self).__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(input_dim, input_dim // reduction, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(input_dim // reduction, input_dim, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        z = self.global_avgpool(x)
        z = self.relu(self.conv1(z))
        z = self.sigmoid(self.conv2(z))
        return x * z


class SCse(nn.Module):
    def __init__(self, dim):
        super(SCse, self).__init__()
        self.satt = SpatialAttention2d(dim)
        self.catt = GAB(dim)

    def forward(self, x):
        return self.satt(x) + self.catt(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=16):
        super(ConvBlock, self).__init__()
        # changed kernel size
        # we don't know why, but this is better than (3, 3) kernels.
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(2, 2),
                               stride=(1, 1),
                               padding=(1, 1),
                               bias=False)
        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=(2, 2),
                               stride=(1, 1),
                               padding=(1, 1),
                               bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Spatial and Channel Squeeze and Exitation
        self.scse = SCse(out_channels)

    #     self.init_weights()

    # def init_weights(self):
    #     init_layer(self.conv1)
    #     init_layer(self.conv2)
    #     init_bn(self.bn1)
    #     init_bn(self.bn2)

    def forward(self, inp, pool_size=(2, 2), pool_type="avg"):
        x = inp
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.scse(self.bn2(self.conv2(x))))
        if pool_type == "max":
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg":
            x = F.avg_pool2d(x, kernel_size=pool_size)
        # Added 'both' pool.
        elif pool_type == "both":
            x1 = F.max_pool2d(x, kernel_size=pool_size)
            x2 = F.avg_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            import pdb
            pdb.set_trace()
        return x


class Classifier_DCase(nn.Module):
    def __init__(self, num_classes=80):
        super(Classifier_DCase, self).__init__()
        # 5 ConvBlocks
        self.conv1 = ConvBlock(3, 32)
        self.conv2 = ConvBlock(32, 64)
        self.conv3 = ConvBlock(64, 128)
        self.conv4 = ConvBlock(128, 256)
        self.conv5 = ConvBlock(256, 512)

        
        self.bn1 = nn.BatchNorm1d(1536)
        self.drop1 = nn.Dropout(0.4)
        self.fc1 = nn.Linear(1536, 512)
        self.prelu = nn.PReLU()
        self.bn2 = nn.BatchNorm1d(512)
        self.drop2 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(512, num_classes)
        self.num_classes = num_classes

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # def init_weight(self):
    #     init_layer(self.fc1)
    #     init_layer(self.fc2)
    #     init_bn(self.bn1)
    #     init_bn(self.bn2)

    def forward(self, x):
        bs, seq, c, h, w = x.shape
        x = x.reshape(bs*seq,c,h,w)
        
        # Changed pooling policy.
        x = self.conv1(x, pool_size=(1, 1), pool_type="both")
        x = self.conv2(x, pool_size=(4, 1), pool_type="both")
        x = self.conv3(x, pool_size=(1, 3), pool_type="both")
        x = self.conv4(x, pool_size=(4, 1), pool_type="both")
        x = self.conv5(x, pool_size=(1, 3), pool_type="both")
        
        # Cutting the feature map to arbitrary size.
    
        x1_max = F.max_pool2d(x, (5, 8))
        x1_mean = F.avg_pool2d(x, (5, 8))
        x1 = F.adaptive_avg_pool2d((x1_max + x1_mean),1).view(x.size(0), -1)

        x2_max = F.max_pool2d(x, (2, 4))
        x2_mean = F.avg_pool2d(x, (2, 4))
        x2 = F.adaptive_avg_pool2d((x2_max + x2_mean),1).view(x.size(0), -1)
        
        x = torch.mean(x, dim=3)
        x, _ = torch.max(x, dim=2)
        
        x = torch.cat([x, x1, x2], dim=1)
        x = self.drop1(self.bn1(x))
        x = self.prelu(self.fc1(x))
        x = self.drop2(self.bn2(x))
        x = self.fc2(x)
        x = x.reshape(bs, seq,self.num_classes)
        return x