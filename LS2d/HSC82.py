# full assembly of the sub-parts to form the complete net
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from PIL import Image
from torch.nn.functional import sigmoid


# 1X3 多层


class Conv_block(nn.Module):
    def __init__(self, in_ch, out_ch, size):
        super(Conv_block, self).__init__()
        self.padding = [(size[0] - 1) // 2, (size[1] - 1) // 2]
        self.conv = nn.Conv2d(in_ch, out_ch, size, padding=self.padding, stride=1)
        self.norm = nn.InstanceNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class Double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Double_conv, self).__init__()
        self.branchs = 6
        self.in_ch = in_ch
        self.mid_mid = out_ch // self.branchs
        self.out_ch = out_ch
        self.conv1x1_mid = Conv_block(self.in_ch, self.out_ch, [1, 1])
        self.conv1x1_2 = nn.Conv2d(self.out_ch, self.out_ch, 1)
        self.conv3x3_2_1 = Conv_block(self.mid_mid, self.mid_mid, [1, 3])
        self.conv3x3_1_1 = Conv_block(self.mid_mid, self.mid_mid, [1, 3])
        self.conv3x3_1 = Conv_block(self.mid_mid, self.mid_mid, [1, 3])
        self.conv3x1_1 = Conv_block(self.mid_mid, self.mid_mid, [1, 3])
        self.conv1x3_1 = Conv_block(self.mid_mid, self.mid_mid, [1, 3])
        self.conv3x3_1_2 = Conv_block(self.mid_mid, self.mid_mid, [3, 1])
        self.conv3x3_2_2 = Conv_block(self.mid_mid, self.mid_mid, [3, 1])
        self.conv3x3_2 = Conv_block(self.mid_mid, self.mid_mid, [3, 1])
        self.conv3x1_2 = Conv_block(self.mid_mid, self.mid_mid, [3, 1])
        self.conv1x3_2 = Conv_block(self.mid_mid, self.mid_mid, [3, 1])
        # self.conv1x1_2 = Conv_block(self.mid_mid, self.mid_mid, [1, 1])
        self.conv1x1_1 = nn.Conv2d(self.out_ch, self.out_ch, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.num = max(self.mid_mid // 2, 12)
        self.fc = nn.Linear(in_features=self.mid_mid, out_features=self.num)
        self.fcs = nn.ModuleList([])
        for i in range(self.branchs):
            self.fcs.append(nn.Linear(in_features=self.num, out_features=self.mid_mid))
        self.softmax = nn.Softmax(dim=1)

        self.rel = nn.ReLU(inplace=True)
        if self.in_ch > self.out_ch:
            self.short_connect = nn.Conv2d(in_ch, out_ch, 1, padding=0)

    def forward(self, x):
        short = x
        if self.in_ch > self.out_ch:
            short = self.short_connect(x)
        xxx = self.conv1x1_mid(x)
        x0 = xxx[:, 0:self.mid_mid, ...]
        x1 = xxx[:, self.mid_mid:self.mid_mid * 2, ...]
        x2 = xxx[:, self.mid_mid * 2:self.mid_mid * 3, ...]
        x3 = xxx[:, self.mid_mid * 3:self.mid_mid * 4, ...]
        x4 = xxx[:, self.mid_mid * 4:self.mid_mid * 5, ...]
        x5 = xxx[:, self.mid_mid * 5:self.mid_mid * 6, ...]
        x1 = self.conv1x3_1(x1+x0)
        x2 = self.conv3x1_1(x2 + x1)
        x3 = self.conv3x3_1(x3 + x2)
        x4 = self.conv3x3_1_1(x4 + x3)
        x5 = self.conv3x3_2_1(x5 + x4)
        xx = x0+x1+x2+x3+x4+x5
        sk1 = self.avg_pool(xx)
        sk1 = sk1.view(sk1.size(0),-1)
        #print(sk1.shape)
        sk2 = self.fc(sk1)
        for i, fc in enumerate(self.fcs):
            vector = fc(sk2).unsqueeze(1)
            if i == 0:
                attention_vector = vector
            else:
                attention_vector = torch.cat([attention_vector, vector], dim=1)
        attention_vector = self.softmax(attention_vector)
        attention_vector=attention_vector.unsqueeze(-1).unsqueeze(-1)
        #print(attention_vector[:,0,...].shape)
        x0 = x0 * attention_vector[:, 0, ...]
        x1 = x1 * attention_vector[:, 1, ...]
        x2 = x2 * attention_vector[:, 2, ...]
        x3 = x3 * attention_vector[:, 3, ...]
        x4 = x4 * attention_vector[:, 4, ...]
        x5 = x5 * attention_vector[:, 5, ...]
        xx = torch.cat((x0, x1, x2, x3, x4, x5), dim=1)
        xxx = self.conv1x1_1(xx)
        x0 = xxx[:, 0:self.mid_mid, ...]
        x1_2 = xxx[:, self.mid_mid:self.mid_mid * 2, ...]
        x2_2 = xxx[:, self.mid_mid * 2:self.mid_mid * 3, ...]
        x3_2 = xxx[:, self.mid_mid * 3:self.mid_mid * 4, ...]
        x4_2 = xxx[:, self.mid_mid * 4:self.mid_mid * 5, ...]
        x5_2 = xxx[:, self.mid_mid * 5:self.mid_mid * 6, ...]
        x1 = self.conv1x3_2(x1_2)
        x2 = self.conv3x1_2(x1 + x2_2)
        x3 = self.conv3x3_2(x2 + x3_2)
        x4 = self.conv3x3_1_2(x3 + x4_2)
        x5 = self.conv3x3_2_2(x4 + x5_2)
        xx = torch.cat((x0, x1, x2, x3, x4, x5), dim=1)
        xxx = self.conv1x1_2(xx)
        return self.rel(xxx + short)


class Conv_down_2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Conv_down_2, self).__init__()
        self.conv = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, stride=2, bias=True)
        self.norm = nn.InstanceNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        # self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class Conv_down(nn.Module):
    '''(conv => ReLU) * 2 => MaxPool2d'''

    def __init__(self, in_ch, out_ch, flage):
        """
        Args:
            in_ch(int) : input channel
            out_ch(int) : output channel
        """
        super(Conv_down, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.flage = flage
        if self.in_ch == 1:
            self.first = nn.Sequential(
                Conv_block(self.in_ch, self.out_ch, [3, 3]),
                Double_conv(self.out_ch, self.out_ch),
            )
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv_down = Conv_down_2(self.out_ch, self.out_ch)
        else:
            self.conv = Double_conv(self.in_ch, self.out_ch)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv_down = Conv_down_2(self.in_ch, self.out_ch)

    def forward(self, x):
        if self.in_ch == 1:
            x = self.first(x)
            pool_x = torch.cat((self.pool(x), self.conv_down(x)), dim=1)
        else:
            x = self.conv(x)
            if self.flage == True:
                pool_x = torch.cat((self.pool(x), self.conv_down(x)), dim=1)
            else:
                pool_x = None
        return pool_x, x


class side_output(nn.Module):
    def __init__(self, inChans, outChans, factor, padding):
        super(side_output, self).__init__()
        self.conv0 = nn.Conv2d(inChans, outChans, 3, 1, 1)
        self.transconv1 = nn.ConvTranspose2d(outChans, outChans, 2 * factor, factor, padding=padding)

    def forward(self, x):
        out = self.conv0(x)
        out = self.transconv1(out)
        return out


device1 = torch.device("cuda")


def hdc(image, num=2):
    x1 = torch.Tensor([]).to(device1)
    for i in range(num):
        for j in range(num):
            x3 = image[:, :, i::num, j::num]
            x1 = torch.cat((x1, x3), dim=1)
    return x1


class Conv_up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Conv_up, self).__init__()
        self.up = nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0, stride=1)
        self.conv = Double_conv(in_ch, out_ch)
        self.interp = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x1, x2):
        x1 = self.interp(x1)
        x1 = self.up(x1)
        x1 = torch.cat((x1, x2), dim=1)
        x1 = self.conv(x1)
        return x1


class CleanU_Net(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CleanU_Net, self).__init__()
        self.first = Conv_block(4, 72, [3, 3])
        self.Conv_down1 = Conv_down(72, 72, True)
        self.Conv_down2 = Conv_down(144, 144, True)
        self.Conv_down3 = Conv_down(288, 288, True)
        # self.Conv_down4 = Conv_down(512, 512,True)
        self.Conv_down5 = Conv_down(576, 576, False)

        # self.Conv_up1 = Conv_up(1024,280)
        self.Conv_up2 = Conv_up(576, 288)
        self.Conv_up3 = Conv_up(288, 144)
        self.Conv_up4 = Conv_up(144, 72)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.Conv_out = nn.Conv2d(72, out_channels, 1, padding=0, stride=1)

        # self.out1 = side_output(256, 2, 8, 4)
        # self.out2 = side_output(128, 2, 4, 2)
        # self.out3 = side_output(64, 2, 2, 1)

    def forward(self, x):
        x = hdc(x)
        x = self.first(x)
        x, conv1 = self.Conv_down1(x)
        # print("dConv1 => down1|", x.shape)
        x, conv2 = self.Conv_down2(x)
        # print("dConv2 => down2|", x.shape)
        x, conv3 = self.Conv_down3(x)
        # print("dConv3 => down3|", x.shape)
        # x, conv4 = self.Conv_down4(x)
        # print("dConv4 => down4|", x.shape)
        _, x = self.Conv_down5(x)
        # print("dConv5|", x.shape)
        # x = self.Conv_up1(x, conv4)
        # x1=self.out1(x)
        # print("up1 => uConv1|", x.shape)
        x = self.Conv_up2(x, conv3)
        # out1 = self.out1(x)
        # x2 = self.out2(x)
        # print("up2 => uConv2|", x.shape)
        x = self.Conv_up3(x, conv2)
        # out2 = self.out2(x)
        # x3= self.out3(x)
        # print("up3 => uConv3|", x.shape)
        x = self.Conv_up4(x, conv1)
        # out3 = self.out3(x)
        x = self.up(x)
        x = self.Conv_out(x)

        return x


def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count


if __name__ == "__main__":
    # A full forward pass
    from torchstat import stat
    # from torchsummary import summary
    # from thop import profile

    device = torch.device('cuda')
    # image_size = 128
    # out = None
    # x = torch.rand((1, 1, 64, 64), device=device)
    # print("x size: {}".format(x.size()))
    model = CleanU_Net(1, 2)
    # flops, params = profile(model, inputs=(x, out))
    # print("***********")
    # print(flops, params)
    # print("***********")
    print(count_param(model))
    # summary(model, input_size=(1, 256, 256), batch_size=1, device="cuda")
    stat(model, (1, 512, 512))
