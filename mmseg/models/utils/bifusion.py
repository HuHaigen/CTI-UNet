import torch
import torch.nn as nn


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat(
            (torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1
        )


class Conv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        bn=False,
        relu=True,
        bias=True,
    ):
        super(Conv, self).__init__()
        self.inp_dim = in_channels
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=(kernel_size - 1) // 2,
            bias=bias,
        )
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Residual(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Residual, self).__init__()
        mid_channels = out_channels // 2
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = Conv(in_channels, mid_channels, 1, relu=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv2 = Conv(mid_channels, mid_channels, 3, relu=False)
        self.bn3 = nn.BatchNorm2d(mid_channels)
        self.conv3 = Conv(mid_channels, out_channels, 1, relu=False)
        self.skip_layer = Conv(in_channels, out_channels, 1, relu=False)
        if in_channels == out_channels:
            self.need_skip = False
        else:
            self.need_skip = True

    def forward(self, x):
        if self.need_skip:
            residual = self.skip_layer(x)
        else:
            residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        return out


class BiFusion(nn.Module):
    def __init__(
        self,
        in_channels_trans,
        in_channels_cnn,
        rate,
        in_channels,
        out_channels,
        drop_rate=0.0,
    ):
        """
        Args:
            ch_1: Transformer 分支输入通道
            ch_2: CNN 分支输入通道
            r_2: Transformer 卷积比率
            ch_int, ch_out: Wg, Wx 的输入输出通道
        """
        super(BiFusion, self).__init__()

        # channel attention for F_g, use SE Block
        self.fc1 = nn.Conv2d(in_channels_cnn, in_channels_cnn // rate, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels_cnn // rate, in_channels_cnn, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        # spatial attention for F_l
        # self.compress = ChannelPool()
        self.spatial = Conv(2, 1, 7, bn=True, relu=False, bias=False)

        # bi-linear modelling for both
        self.W_g = Conv(in_channels_trans, in_channels, 1, bn=True, relu=False)
        self.W_x = Conv(in_channels_cnn, in_channels, 1, bn=True, relu=False)
        self.W = Conv(in_channels, in_channels, 3, bn=True, relu=True)

        self.relu = nn.ReLU(inplace=True)

        self.residual = Residual(
            in_channels_trans + in_channels_cnn + in_channels, out_channels
        )

        self.dropout = nn.Dropout2d(drop_rate)
        self.drop_rate = drop_rate

    def forward(self, g, x):
        # bilinear pooling
        W_g = self.W_g(g)
        W_x = self.W_x(x)
        bp = self.W(W_g * W_x)

        # spatial attention for cnn branch
        g_in = g
        # g = self.compress(g)
        g = torch.cat(
            (torch.max(g, 1)[0].unsqueeze(1), torch.mean(g, 1).unsqueeze(1)), dim=1
        )
        g = self.spatial(g)
        g = self.sigmoid(g) * g_in

        # channel attetion for transformer branch
        x_in = x
        x = x.mean((2, 3), keepdim=True)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x) * x_in
        fuse = self.residual(torch.cat([g, x, bp], 1))

        if self.drop_rate > 0:
            return self.dropout(fuse)
        else:
            return fuse


if __name__ == "__main__":
    x_t = torch.randn((1, 64, 16, 16))
    x_c = torch.randn((1, 64, 16, 16))

    net = BiFusion(64, 64, 1, 12, 64)
    y = net(x_t, x_c)
    print(y.shape)
