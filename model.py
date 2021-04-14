# -*- encoding: utf-8 -*-
'''
@File        :main.py
@Date        :2021/04/14 16:05
@Author      :Wentong Liao, Kai Hu
@Email       :liao@tnt.uni-hannover.de
@Version     :0.1
@Description : Implementation of SSA-GAN
'''
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict
from sync_batchnorm import SynchronizedBatchNorm2d

BatchNorm = SynchronizedBatchNorm2d


class NetG(nn.Module):
    def __init__(self, ngf=64, nz=100):
        super(NetG, self).__init__()
        self.ngf = ngf

        self.conv_mask = nn.Sequential(nn.Conv2d(8 * ngf, 100, 3, 1, 1),
                                       BatchNorm(100),
                                       nn.ReLU(),
                                       nn.Conv2d(100, 1, 1, 1, 0))

        # layer1输入的是一个100x1x1的随机噪声, 输出尺寸(ngf*8)x4x4
        self.fc = nn.Linear(nz, ngf * 8 * 4 * 4)
        self.block0 = G_Block(ngf * 8, ngf * 8)  # 4x4
        self.block1 = G_Block(ngf * 8, ngf * 8)  # 8x8
        self.block2 = G_Block(ngf * 8, ngf * 8)  # 16x16
        self.block3 = G_Block(ngf * 8, ngf * 8)  # 32x32
        self.block4 = G_Block(ngf * 8, ngf * 4)  # 64x64
        self.block5 = G_Block(ngf * 4, ngf * 2)  # 128x128
        self.block6 = G_Block(ngf * 2, ngf * 1, predict_mask=False)  # 256x256

        self.conv_img = nn.Sequential(
            BatchNorm(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, 3, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, x, c):

        out = self.fc(x)
        out = out.view(x.size(0), 8 * self.ngf, 4, 4)
        hh, ww = out.size(2), out.size(3)
        stage_mask = self.conv_mask(out)
        fusion_mask = torch.sigmoid(stage_mask)
        stage_mask_4 = fusion_mask
        out, stage_mask = self.block0(out, c, fusion_mask)

        out = F.interpolate(out, scale_factor=2)
        hh, ww = out.size(2), out.size(3)
        stage_mask = F.interpolate(stage_mask, size=(hh, ww), mode='bilinear', align_corners=True)
        fusion_mask = torch.sigmoid(stage_mask)
        stage_mask_8 = fusion_mask
        out, stage_mask = self.block1(out, c, fusion_mask)

        out = F.interpolate(out, scale_factor=2)
        hh, ww = out.size(2), out.size(3)
        stage_mask = F.interpolate(stage_mask, size=(hh, ww), mode='bilinear', align_corners=True)
        fusion_mask = torch.sigmoid(stage_mask)
        stage_mask_16 = fusion_mask
        out, stage_mask = self.block2(out, c, fusion_mask)

        out = F.interpolate(out, scale_factor=2)
        hh, ww = out.size(2), out.size(3)
        stage_mask = F.interpolate(stage_mask, size=(hh, ww), mode='bilinear', align_corners=True)
        fusion_mask = torch.sigmoid(stage_mask)
        stage_mask_32 = fusion_mask
        out, stage_mask = self.block3(out, c, fusion_mask)

        out = F.interpolate(out, scale_factor=2)
        hh, ww = out.size(2), out.size(3)
        stage_mask = F.interpolate(stage_mask, size=(hh, ww), mode='bilinear', align_corners=True)
        fusion_mask = torch.sigmoid(stage_mask)
        stage_mask_64 = fusion_mask
        out, stage_mask = self.block4(out, c, fusion_mask)

        out = F.interpolate(out, scale_factor=2)
        hh, ww = out.size(2), out.size(3)
        stage_mask = F.interpolate(stage_mask, size=(hh, ww), mode='bilinear', align_corners=True)
        fusion_mask = torch.sigmoid(stage_mask)
        stage_mask_128 = fusion_mask
        out, stage_mask = self.block5(out, c, fusion_mask)

        out = F.interpolate(out, scale_factor=2)
        hh, ww = out.size(2), out.size(3)
        stage_mask = F.interpolate(stage_mask, size=(hh, ww), mode='bilinear', align_corners=True)
        fusion_mask = torch.sigmoid(stage_mask)
        stage_mask_256 = fusion_mask
        out, _ = self.block6(out, c, fusion_mask)

        out = self.conv_img(out)

        # return out, fusion_mask
        return out, [stage_mask_4, stage_mask_8, stage_mask_16, stage_mask_32,
                     stage_mask_64, stage_mask_128, stage_mask_256]


class G_Block(nn.Module):

    def __init__(self, in_ch, out_ch, num_w=256, predict_mask=True):
        super(G_Block, self).__init__()

        self.learnable_sc = in_ch != out_ch
        self.predict_mask = predict_mask
        self.c1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.c2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.affine0 = affine(in_ch)
        #self.affine1 = affine(in_ch)
        self.affine2 = affine(out_ch)
        #self.affine3 = affine(out_ch)
        self.gamma = nn.Parameter(torch.zeros(1))
        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)

        if self.predict_mask:
            self.conv_mask = nn.Sequential(nn.Conv2d(out_ch, 100, 3, 1, 1),
                                           BatchNorm(100),
                                           nn.ReLU(),
                                           nn.Conv2d(100, 1, 1, 1, 0))

    def forward(self, x, y=None, fusion_mask=None):
        out = self.shortcut(x) + self.gamma * self.residual(x, y, fusion_mask)

        if self.predict_mask:
            mask = self.conv_mask(out)
        else:
            mask = None

        return out, mask

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
        return x

    def residual(self, x, y=None, fusion_mask=None):
        h = self.affine0(x, y, fusion_mask)
        h = nn.ReLU(inplace=True)(h)
        h = self.c1(h)

        h = self.affine2(h, y, fusion_mask)
        h = nn.ReLU(inplace=True)(h)
        return self.c2(h)


class affine(nn.Module):

    def __init__(self, num_features):
        super(affine, self).__init__()

        self.batch_norm2d = BatchNorm(num_features, affine=False)

        self.fc_gamma = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(256, 256)),
            ('relu1', nn.ReLU(inplace=True)),
            ('linear2', nn.Linear(256, num_features)),
        ]))
        self.fc_beta = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(256, 256)),
            ('relu1', nn.ReLU(inplace=True)),
            ('linear2', nn.Linear(256, num_features)),
        ]))
        self._initialize()

    def _initialize(self):
        nn.init.zeros_(self.fc_gamma.linear2.weight.data)
        nn.init.zeros_(self.fc_gamma.linear2.bias.data)
        nn.init.zeros_(self.fc_beta.linear2.weight.data)
        nn.init.zeros_(self.fc_beta.linear2.bias.data)

    def forward(self, x, y=None, fusion_mask=None):
        x = self.batch_norm2d(x)
        weight = self.fc_gamma(y)
        bias = self.fc_beta(y)

        if weight.dim() == 1:
            weight = weight.unsqueeze(0)
        if bias.dim() == 1:
            bias = bias.unsqueeze(0)

        size = x.size()
        weight = weight.unsqueeze(-1).unsqueeze(-1).expand(size)
        bias = bias.unsqueeze(-1).unsqueeze(-1).expand(size)
        weight = weight * fusion_mask + 1
        bias = bias * fusion_mask
        return weight * x + bias


class D_GET_LOGITS(nn.Module):
    def __init__(self, ndf):
        super(D_GET_LOGITS, self).__init__()
        self.df_dim = ndf

        self.joint_conv = nn.Sequential(
            nn.Conv2d(ndf * 16 + 256, ndf * 2, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, 1, 4, 1, 0, bias=False),
        )

    def forward(self, out, y):

        y = y.view(-1, 256, 1, 1)
        y = y.repeat(1, 1, 4, 4)
        h_c_code = torch.cat((out, y), 1)
        out = self.joint_conv(h_c_code)
        return out


# 定义鉴别器网络D
class NetD(nn.Module):
    def __init__(self, ndf):
        super(NetD, self).__init__()

        self.conv_img = nn.Conv2d(3, ndf, 3, 1, 1)  # 128
        self.block0 = resD(ndf * 1, ndf * 2)  # 64
        self.block1 = resD(ndf * 2, ndf * 4)  # 32
        self.block2 = resD(ndf * 4, ndf * 8)  # 16
        self.block3 = resD(ndf * 8, ndf * 16)  # 8
        self.block4 = resD(ndf * 16, ndf * 16)  # 4
        self.block5 = resD(ndf * 16, ndf * 16)  # 4

        self.COND_DNET = D_GET_LOGITS(ndf)

    def forward(self, x):

        out = self.conv_img(x)
        out = self.block0(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)

        return out


class resD(nn.Module):
    def __init__(self, fin, fout, downsample=True):
        super().__init__()
        self.downsample = downsample
        self.learned_shortcut = (fin != fout)
        self.conv_r = nn.Sequential(
            nn.Conv2d(fin, fout, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(fout, fout, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv_s = nn.Conv2d(fin, fout, 1, stride=1, padding=0)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, c=None):
        return self.shortcut(x) + self.gamma * self.residual(x)

    def shortcut(self, x):
        if self.learned_shortcut:
            x = self.conv_s(x)
        if self.downsample:
            return F.avg_pool2d(x, 2)
        return x

    def residual(self, x):
        return self.conv_r(x)


def conv2d(in_feat, out_feat, kernel_size=3, stride=1, padding=1, bias=True, spectral_norm=False):
    conv = nn.Conv2d(in_feat, out_feat, kernel_size, stride, padding, bias=bias)
    if spectral_norm:
        return nn.utils.spectral_norm(conv, eps=1e-4)
    else:
        return conv


def linear(in_feat, out_feat, bias=True, spectral_norm=False):
    lin = nn.Linear(in_feat, out_feat, bias=bias)
    if spectral_norm:
        return nn.utils.spectral_norm(lin)
    else:
        return lin
