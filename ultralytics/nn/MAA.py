import torch
import torch.nn as nn


class MAA_Block(nn.Module):

    def __init__(self, in_channels):
        super(MAA_Block, self).__init__()
        self.in_channels = in_channels
        # Depthwise convolutions for multi-scale feature extraction
        self.dwc1 = nn.Conv2d(in_channels // 4, in_channels // 4, 1, padding=0, groups=in_channels // 4)
        self.dwc3 = nn.Conv2d(in_channels // 4, in_channels // 4, 3, padding=1, groups=in_channels // 4)
        self.dwc5 = nn.Conv2d(in_channels // 4, in_channels // 4, 5, padding=2, groups=in_channels // 4)
        self.bn = nn.BatchNorm2d(in_channels//4)
        self.bn1 = nn.BatchNorm2d(in_channels)
        # Pointwise convolution to fuse features
        self.pointwise_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.pointwise_conv1 = nn.Conv2d(in_channels, in_channels, 1)
        self.pointwise_conv2 = nn.Conv2d(in_channels * 2, in_channels, 1)

    def forward(self, x):
        x0 = self.pointwise_conv(x)
        x = self.pointwise_conv1(x)
        # Split the input feature map into two parts
        x1, x2, x3, x4 = torch.split(x, self.in_channels // 4, dim=1)

        # Apply depthwise convolutions
        x1 = self.dwc1(x1)
        x2 = self.dwc3(x2)
        x3 = self.dwc5(x3)

        # Concatenate the processed features
        x_concat = torch.cat([x1, x2, x3, x4], dim=1)

        # Concatenate with x2 and apply pointwise convolution
        x_out = torch.cat([x_concat, x0], dim=1)
        x_out = self.pointwise_conv2(x_out)

        return x_out


class MAA(nn.Module):
    def __init__(self, c1, c2):
        super(MAA, self).__init__()
        self.c1 = c1
        self.pointwise_conv = nn.Conv2d(c1, c1, 1)
        self.pointwise_conv1 = nn.Conv2d(c1, c2, 1)
        self.maa_block = MAA_Block(c1 // 2)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.pointwise_conv(x)
        x1, x2 = torch.split(x, self.c1 // 2, dim=1)
        x1 = self.maa_block(x1)
        x = torch.cat([x1, x2], dim=1)
        x = self.pointwise_conv1(x)

        return x


