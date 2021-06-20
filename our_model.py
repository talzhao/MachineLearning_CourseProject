
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv_layer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv_layer, self).__init__()
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.conv_l1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv_l2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        x = F.relu(self.conv_l1(x))
        x = self.batch_norm(x)
        x = F.relu(self.conv_l2(x))
        x = self.batch_norm(x)
        return x


class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')

        self.conv1 = Conv_layer(1, 32)
        self.conv2 = Conv_layer(32, 64)
        self.conv3 = Conv_layer(64, 128)
        self.conv4 = Conv_layer(128, 256)

        self.conv5 = Conv_layer(256, 512)

        self.conv6 = Conv_layer(768, 256)
        self.conv7 = Conv_layer(384, 128)
        self.conv8 = Conv_layer(192, 64)
        self.conv9 = Conv_layer(96, 32)

        self.conv10 = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        c1 = self.conv1(x)
        x = self.pool1(c1)
        c2 = self.conv2(x)
        x = self.pool2(c2)
        c3 = self.conv3(x)
        x = self.pool3(c3)
        c4 = self.conv4(x)
        x = self.pool4(c4)
        x = self.conv5(x)
        x = self.up1(x)
        x = torch.cat([x, c4], 1)
        x = self.conv6(x)
        x = self.up2(x)
        x = torch.cat([x, c3], 1)
        x = self.conv7(x)
        x = self.up3(x)
        x = torch.cat([x, c2], 1)
        x = self.conv8(x)
        x = self.up4(x)
        x = torch.cat([x, c1], 1)
        x = self.conv9(x)
        x = self.conv10(x)
        return x


# class Unet(nn.Module):
#     def __init__(self):
#         super(Unet, self).__init__()
#         self.pooling_l1 = nn.MaxPool2d(kernel_size=2)
#         self.pooling_l2 = nn.MaxPool2d(kernel_size=2)
#         self.pooling_l3 = nn.MaxPool2d(kernel_size=2)
#         self.pooling_l4 = nn.MaxPool2d(kernel_size=2)
#         self.pooling_l5 = nn.MaxPool2d(kernel_size=2)
#
#         self.up_l1 = nn.Upsample(scale_factor=2, mode='nearest')
#         self.up_l2 = nn.Upsample(scale_factor=2, mode='nearest')
#         self.up_l3 = nn.Upsample(scale_factor=2, mode='nearest')
#         self.up_l4 = nn.Upsample(scale_factor=2, mode='nearest')
#         self.up_l5 = nn.Upsample(scale_factor=2, mode='nearest')
#
#         self.conv_l1 = Conv_layer(1, 32)
#         self.conv_l2 = Conv_layer(32, 64)
#         self.conv_l3 = Conv_layer(64, 128)
#         self.conv_l4 = Conv_layer(128, 256)
#         self.conv_l5 = Conv_layer(256, 512)
#         self.conv_l6 = Conv_layer(512, 1024)
#
#         self.conv_l7 = Conv_layer(1536, 512)
#         self.conv_l8 = Conv_layer(768, 256)
#         self.conv_l9 = Conv_layer(384, 128)
#         self.conv_l10 = Conv_layer(192, 64)
#         self.conv_l11 = Conv_layer(96, 32)
#         self.conv_l12 = nn.Conv2d(32, 1, 1)
#         self.dropout = nn.Dropout(0.5)
#
#
#     def forward(self, x):
#         conv1 = self.conv_l1(x)
#         x = self.pooling_l1(conv1)
#         conv2 = self.conv_l2(x)
#         x = self.pooling_l2(conv2)
#         conv3 = self.conv_l3(x)
#         x = self.pooling_l3(conv3)
#         conv4 = self.conv_l4(x)
#         x = self.pooling_l4(conv4)
#         conv5 = self.conv_l5(x)
#         x = self.pooling_l5(conv5)
#         x = self.conv_l6(x)
#
#         # x = self.dropout(x)
#         ## upsampling process
#         x = self.up_l1(x)
#
#         x = torch.cat( [ x, conv5 ] , 1)
#         x = self.conv_l7(x)
#         x = self.up_l2(x)
#         x = torch.cat( [ x, conv4 ] , 1)
#         x = self.conv_l8(x)
#         x = self.up_l3(x)
#         x = torch.cat( [ x, conv3], 1)
#         x = self.conv_l9(x)
#         x = self.up_l4(x)
#         x = torch.cat([x, conv2], 1)
#         x = self.conv_l10(x)
#         x = self.up_l5(x)
#         x = torch.cat([x, conv1], 1)
#         x = self.conv_l11(x)
#         x = self.conv_l12(x)
#         # print( "self.conv_l10(x)", x.shape)
#         return x
#
