# -*- coding: utf-8 -*-
from tensorboardX import SummaryWriter
import torch.nn as nn
import torch
from PIL import Image, ImageFont, ImageDraw
from torchsummary import summary
import os
from lib.Config.movenet_config import config_loc as cfg
import torch.nn.functional as F

class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)

class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out

class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        score = self.channel_attention(x)
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out


# 我想通过zero padding获取heatmap的位置信息，并且学习，你觉得我应该怎么写这个zero padding模块，并且应该加在哪个地方
# 定义SE-Net注意力模块
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

# Unet的下采样模块，两次卷积
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, channel_reduce=False):  # 只是定义网络中需要用到的方法
        super(DoubleConv, self).__init__()

        # 通道减少的系数
        # True为2，False为1
        coefficient = 2 if channel_reduce else 1
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, coefficient * out_channels, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(coefficient * out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(coefficient * out_channels, out_channels, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # print("qqqqqqq")
        return self.double_conv(x)


# 上采样（转置卷积加残差链接）
class Up(nn.Module):

    # 千万注意输入，in_channels是要送入二次卷积的channel，out_channels是二次卷积之后的channel
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 先上采样特征图
        self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=4, stride=2, padding=1)
        self.conv = DoubleConv(in_channels, out_channels, channel_reduce=True)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # print("x1.shape: ",x1.shape)
        # print("x2.shape: ",x2.shape)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv(x)
        return x


# simple U-net模型
class U_net(nn.Module):

    def __init__(self, in_channels=16, feature_scale=2):  # 只是定义网络中需要用到的方法
        super(U_net, self).__init__()

        self.feature_scale = feature_scale

        filters = [32, 64, 128, 256, 512]
        filters = [int(x / self.feature_scale) for x in filters]

        self.double_conv1 = DoubleConv(in_channels, filters[0])
        self.double_conv2 = DoubleConv(filters[0], filters[1])
        self.double_conv3 = DoubleConv(filters[1], filters[2])
        self.double_conv4 = DoubleConv(filters[2], filters[3])
        self.double_conv5 = DoubleConv(filters[3], filters[3])

        self.cbam1 = CBAM(channel=filters[0])
        self.cbam2 = CBAM(channel=filters[1])
        self.cbam3 = CBAM(channel=filters[2])
        self.cbam4 = CBAM(channel=filters[3])

        # 上采样
        self.up1 = Up(filters[4], filters[2])
        self.up2 = Up(filters[3], filters[1])
        self.up3 = Up(filters[2], filters[0])
        self.up4 = Up(filters[1], 16)

        # self.feat_conv = nn.Linear(256*11*20, 1048)

        # # 添加SE-Net注意力模块
        # self.se1 = SEBlock(128)
        # self.se2 = SEBlock(64)
        # self.se3 = SEBlock(32)
        # self.se4 = SEBlock(16)

        self.init_weight()

        # 最后一层
        # self.out = nn.Conv2d(16, cfg['kpt_n'], kernel_size=(1, 1), padding=0)

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # m.weight.data.normal_(0, 0.01)
                torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
                # torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def save_feature_maps(self, feature_map, index, dir_name, name_prefix):
        output_dir = os.path.join('.', 'output', dir_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for i in range(feature_map.size(1)):
            img = feature_map[0, i].cpu().detach().numpy()  # Extract a single feature map
            img_path = os.path.join(output_dir, f"{name_prefix}_layer_{index}_channel_{i}.png")
            self.save_image(img, img_path)

    @staticmethod
    def save_image(image, img_path):
        # Normalize the image to range [0, 255]
        image = (image - image.min()) / (image.max() - image.min()) * 255
        image = image.astype('uint8')
        img = Image.fromarray(image)
        img.save(img_path)

    def init_weights(self, module):
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

    def forward(self, x, debug=False):
        # down
        c1 = self.double_conv1(x)  # [1, 32, i_h, i_w]
        c1 = self.cbam1(c1) + c1
        p1 = nn.MaxPool2d(2)(c1)   # [1, 32, i_h/2, i_w/2]
        # print("p1: ", p1.shape)
        c2 = self.double_conv2(p1) # [1, 64, i_h/2, i_w/2]
        c2 = self.cbam2(c2) + c2
        p2 = nn.MaxPool2d(2)(c2)   # [1, 64, i_h/4, i_w/4]
        # print("p2: ", p2.shape)
        c3 = self.double_conv3(p2)  # [1, 128, i_h/4, i_w/4]
        c3 = self.cbam3(c3) + c3
        p3 = nn.MaxPool2d(2)(c3)  # [1, 128, 45, 80]
        # print("p3: ", p3.shape)
        c4 = self.double_conv4(p3)  # [1, 128, 22, 40]
        # if return_score:
        #     temp, score = self.cbam4(c4, return_score=return_score)
        #     _, topk_indices = torch.topk(score, k=5, dim=1)
        # else:
        # temp = self.cbam4(c4)
        #
        # c4 = temp + c4
        p4 = nn.MaxPool2d(2)(c4)  # [1, 256, 11, 20]
        # print("p4: ", p4.shape)
        c5 = self.double_conv5(p4)  # [1, 256, 11, 20]
        # print("c6: ",c6.shape)
        # 最后一次卷积不做池化操作

        # # up
        u1 = self.up1(c5, c4)  # (,128,64,64)
        # u1 = self.se1(u1)  # 添加SE-Net注意力模块
        u2 = self.up2(u1, c3)  # (,64,128,128)
        # u2 = self.se2(u2)  # 添加SE-Net注意力模块
        u3 = self.up3(u2, c2)  # (,32,256,256)
        # u3 = self.se3(u3)  # 添加SE-Net注意力模块
        u4 = self.up4(u3, c1)  # (,16,512,512)
        # u4 = self.se4(u4)  # 添加SE-Net注意力模块

        # # up
        # u1 = self.up1(c5, c4)  # (,128,64,64)
        # # print("u1: ",u1.shape)
        # u2 = self.up2(u1, c3)  # (,64,128,128)
        # # print("u2: ", u2.shape)
        # u3 = self.up3(u2, c2)  # (,32,256,256)
        # # print("u3: ", u3.shape)
        # u4 = self.up4(u3, c1)  # (,16,512,512)
        # # print("u4: ", u4.shape)

        # 最后一层，隐射到16个特征图
        # out = self.out(u4)

        # print("out: ",out.shape)

        if debug:
            self.save_feature_maps(p1, 1, 'encoder', 'c1')
            self.save_feature_maps(p2, 2, 'encoder', 'c2')
            self.save_feature_maps(p3, 3, 'encoder', 'c3')
            self.save_feature_maps(p4, 4, 'encoder', 'c4')
            self.save_feature_maps(u1, 1, 'decoder', 'u1')
            self.save_feature_maps(u2, 2, 'decoder', 'u2')
            self.save_feature_maps(u3, 3, 'decoder', 'u3')
            self.save_feature_maps(u4, 4, 'decoder', 'u4')
            # self.save_feature_maps(out, 4, 'map', 'out')
        # 使用 topk_indices 来获取相应的5个通道的值
        # topk_channels = topk_indices.squeeze(2).squeeze(2)  # 去除不必要的维度
        # # 提取 c4 中相应通道的值
        # topk_c4_values = c4[0, topk_channels, :, :]
        # feat_x = topk_c4_values.view(c5.shape[0], -1)  # last channel 880
        # feat_x = c5.view(c5.shape[0], -1)
        feat_x = c5
        return u4, feat_x

    def summary(self, net):
        x = torch.rand(cfg['batch_size'], 3, cfg['input_h'], cfg['input_w'])  # torch.Size([1, 3, 180, 320])
        # 送入设备
        x = x.to(cfg['device'])
        # 输出y的shape
       #  print("net(x).shape",net(x).shape)

        # 展示网络结构
        summary(net, x)



