import torch
import torch.nn as nn
import torch.nn.functional as F
# from model.YNet.arch_util import LayerNorm2d
# from mmcv.ops import DeformConv2dPack as DCN


class DMR(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super(DMR, self).__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        # self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv11 = nn.Conv2d(mid_channels, mid_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.bn11 = nn.BatchNorm2d(mid_channels)
        self.relu11 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(in_channels, mid_channels, kernel_size=5, stride=1, padding=2, bias=True)
        self.bn3 = nn.BatchNorm2d(mid_channels)

        self.conv23 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn23 = nn.BatchNorm2d(mid_channels)
        self.relu23 = nn.ReLU(inplace=True)

        self.conv32 = nn.Conv2d(2 * mid_channels, mid_channels, kernel_size=5, stride=1, padding=2, bias=True)
        self.bn32 = nn.BatchNorm2d(mid_channels)
        self.relu32 = nn.ReLU(inplace=True)

        self.conv2233 = nn.Conv2d(2*mid_channels, mid_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv4 = nn.Conv2d(2*mid_channels, mid_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.relu4 = nn.ReLU(inplace=True)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):

        x2 = self.conv2(x)
        x2_1 = self.bn2(x2)

        x3 = self.conv3(x)
        x3_1 = self.bn3(x3)

        x11 = self.conv11(x)
        x11_2 = self.bn11(x11)
        x11_3 = self.relu11(x11_2)

        # x23 = torch.cat((x2_1, x3_1), dim=-3)
        x23_1 = self.conv23(x2_1+x3_1)
        x23_2 = self.bn23(x23_1)
        x23_3 = self.relu23(x23_2)

        x4 = torch.cat((x11_3, x23_3), dim=-3)

        x4 = self.conv2233(x4)

        return x4


# 自定义 GroupBatchnorm2d 类，实现分组批量归一化
class GroupBatchnorm2d(nn.Module):
    def __init__(self, c_num:int, group_num:int = 16, eps:float = 1e-10):
        super(GroupBatchnorm2d,self).__init__()  # 调用父类构造函数
        assert c_num >= group_num  # 断言 c_num 大于等于 group_num
        self.group_num  = group_num  # 设置分组数量
        self.gamma = nn.Parameter(torch.randn(c_num, 1, 1))  # 创建可训练参数 gamma
        self.beta = nn.Parameter(torch.zeros(c_num, 1, 1))  # 创建可训练参数 beta
        self.eps = eps  # 设置小的常数 eps 用于稳定计算

    def forward(self, x):
        N, C, H, W = x.size()  # 获取输入张量的尺寸
        x = x.view(N, self.group_num, -1)  # 将输入张量重新排列为指定的形状
        mean = x.mean(dim=2, keepdim=True)  # 计算每个组的均值
        std = x.std(dim=2, keepdim=True)  # 计算每个组的标准差
        x = (x - mean) / (std + self.eps)  # 应用批量归一化
        x = x.view(N, C, H, W)  # 恢复原始形状
        return x * self.gamma + self.beta  # 返回归一化后的张量


# 自定义 SRU（Spatial and Reconstruct Unit）类
class SRU(nn.Module):
    def __init__(self,
                 oup_channels:int,  # 输出通道数
                 group_num:int = 16,  # 分组数，默认为16
                 gate_treshold:float = 0.5,  # 门控阈值，默认为0.5
                 torch_gn:bool = False  # 是否使用PyTorch内置的GroupNorm，默认为False
                 ):
        super().__init__()  # 调用父类构造函数

         # 初始化 GroupNorm 层或自定义 GroupBatchnorm2d 层
        self.gn = nn.GroupNorm(num_channels=oup_channels, num_groups=group_num) if torch_gn else GroupBatchnorm2d(c_num=oup_channels, group_num=group_num)
        self.gate_treshold = gate_treshold  # 设置门控阈值
        self.sigomid = nn.Sigmoid()  # 创建 sigmoid 激活函数

    def forward(self, x):
        gn_x = self.gn(x)  # 应用分组批量归一化
        w_gamma = self.gn.gamma / sum(self.gn.gamma)  # 计算 gamma 权重
        reweights = self.sigomid(gn_x * w_gamma)  # 计算重要性权重

        # 门控机制
        info_mask = reweights >= self.gate_treshold  # 计算信息门控掩码
        noninfo_mask = reweights < self.gate_treshold  # 计算非信息门控掩码
        x_1 = info_mask * x  # 使用信息门控掩码
        x_2  = noninfo_mask * x  # 使用非信息门控掩码
        x = self.reconstruct(x_1, x_2)  # 重构特征
        return x

    def reconstruct(self, x_1, x_2):
        x_11, x_12 = torch.split(x_1, x_1.size(1) // 2, dim=1)  # 拆分特征为两部分
        x_21, x_22 = torch.split(x_2, x_2.size(1) // 2, dim=1)  # 拆分特征为两部分
        return torch.cat([x_11 + x_22, x_12 + x_21], dim=1)  # 重构特征并连接

# 自定义 CRU（Channel Reduction Unit）类
class CRU(nn.Module):
    def __init__(self, op_channel:int, alpha:float = 1/2, squeeze_radio:int = 2, group_size:int = 2, group_kernel_size:int = 3):
        super().__init__()  # 调用父类构造函数

        self.up_channel = up_channel = int(alpha * op_channel)  # 计算上层通道数
        self.low_channel = low_channel = op_channel - up_channel  # 计算下层通道数
        self.squeeze1 = nn.Conv2d(up_channel, up_channel // squeeze_radio, kernel_size=1, bias=False)  # 创建卷积层
        self.squeeze2 = nn.Conv2d(low_channel, low_channel // squeeze_radio, kernel_size=1, bias=False)  # 创建卷积层

        # 上层特征转换
        self.GWC = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=group_kernel_size, stride=1, padding=group_kernel_size // 2, groups=group_size)  # 创建卷积层
        self.PWC1 = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=1, bias=False)  # 创建卷积层

        # 下层特征转换
        self.PWC2 = nn.Conv2d(low_channel // squeeze_radio, op_channel - low_channel // squeeze_radio, kernel_size=1, bias=False)  # 创建卷积层
        self.advavg = nn.AdaptiveAvgPool2d(1)  # 创建自适应平均池化层

    def forward(self, x):
        # 分割输入特征
        up, low = torch.split(x, [self.up_channel, self.low_channel], dim=1)
        up, low = self.squeeze1(up), self.squeeze2(low)

        # 上层特征转换
        Y1 = self.GWC(up) + self.PWC1(up)

        # 下层特征转换
        Y2 = torch.cat([self.PWC2(low), low], dim=1)

        # 特征融合
        out = torch.cat([Y1, Y2], dim=1)
        out = F.softmax(self.advavg(out), dim=1) * out
        out1, out2 = torch.split(out, out.size(1) // 2, dim=1)
        return out1 + out2


# 自定义 ScConv（Squeeze and Channel Reduction Convolution）模型
class ScConv(nn.Module):
    def __init__(self, op_channel:int, group_num:int = 16, gate_treshold:float = 0.5, alpha:float = 1/2, squeeze_radio:int = 2, group_size:int = 2, group_kernel_size:int = 3):
        super().__init__()  # 调用父类构造函数

        self.SRU = SRU(op_channel, group_num=group_num, gate_treshold=gate_treshold)  # 创建 SRU 层
        self.CRU = CRU(op_channel, alpha=alpha, squeeze_radio=squeeze_radio, group_size=group_size, group_kernel_size=group_kernel_size)  # 创建 CRU 层

    def forward(self, x):
        x = self.SRU(x)  # 应用 SRU 层
        x = self.CRU(x)  # 应用 CRU 层
        return x


class SobelConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, requires_grad=True):
        assert kernel_size % 2 == 1, 'SobelConv2d\'s kernel_size must be odd.'
        assert out_channels % 4 == 0, 'SobelConv2d\'s out_channels must be a multiple of 4.'
        assert out_channels % groups == 0, 'SobelConv2d\'s out_channels must be a multiple of groups.'

        super(SobelConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        # In non-trainable case, it turns into normal Sobel operator with fixed weight and no bias.
        self.bias = bias if requires_grad else False

        if self.bias:
            self.bias = nn.Parameter(torch.zeros(size=(out_channels,), dtype=torch.float32), requires_grad=True)
        else:
            self.bias = None

        self.sobel_weight = nn.Parameter(torch.zeros(
            size=(out_channels, int(in_channels / groups), kernel_size, kernel_size)), requires_grad=False)

        # Initialize the Sobel kernal
        kernel_mid = kernel_size // 2
        for idx in range(out_channels):
            if idx % 4 == 0:
                self.sobel_weight[idx, :, 0, :] = -1
                self.sobel_weight[idx, :, 0, kernel_mid] = -2
                self.sobel_weight[idx, :, -1, :] = 1
                self.sobel_weight[idx, :, -1, kernel_mid] = 2
            elif idx % 4 == 1:
                self.sobel_weight[idx, :, :, 0] = -1
                self.sobel_weight[idx, :, kernel_mid, 0] = -2
                self.sobel_weight[idx, :, :, -1] = 1
                self.sobel_weight[idx, :, kernel_mid, -1] = 2
            elif idx % 4 == 2:
                self.sobel_weight[idx, :, 0, 0] = -2
                for i in range(0, kernel_mid + 1):
                    self.sobel_weight[idx, :, kernel_mid - i, i] = -1
                    self.sobel_weight[idx, :, kernel_size - 1 - i, kernel_mid + i] = 1
                self.sobel_weight[idx, :, -1, -1] = 2
            else:
                self.sobel_weight[idx, :, -1, 0] = -2
                for i in range(0, kernel_mid + 1):
                    self.sobel_weight[idx, :, kernel_mid + i, i] = -1
                    self.sobel_weight[idx, :, i, kernel_mid + i] = 1
                self.sobel_weight[idx, :, 0, -1] = 2

        # Define the trainable sobel factor
        if requires_grad:
            self.sobel_factor = nn.Parameter(torch.ones(size=(out_channels, 1, 1, 1), dtype=torch.float32),
                                             requires_grad=True)
        else:
            self.sobel_factor = nn.Parameter(torch.ones(size=(out_channels, 1, 1, 1), dtype=torch.float32),
                                             requires_grad=False)

    def forward(self, x):
        if torch.cuda.is_available():
            self.sobel_factor = self.sobel_factor.to(x.device)
            self.sobel_weight = nn.Parameter(self.sobel_weight.to(x.device))
            if self.bias is not None:
                self.bias = nn.Parameter(self.bias.to(x.device))
            x = x.to(x.device)  # 将输入张量 x 移动到相同的设备上

        sobel_weight = self.sobel_weight * self.sobel_factor

        out = F.conv2d(x, sobel_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        return out


class Up(nn.Module):

    def __init__(self, nc, bias):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels=nc, out_channels=nc, kernel_size=2, stride=2, bias=bias)

    def forward(self, x1, x):
        x2 = self.up(x1)

        diffY = x.size()[2] - x2.size()[2]
        diffX = x.size()[3] - x2.size()[3]
        x3 = F.pad(x2, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        return x3


class Up2(nn.Module):

    def __init__(self):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x1, x):
        x2 = self.up(x1)

        diffY = x.size()[2] - x2.size()[2]
        diffX = x.size()[3] - x2.size()[3]
        x3 = F.pad(x2, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        return x3


## Spatial Attention
class Basic(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, padding=0, bias=False):
        super(Basic, self).__init__()
        self.out_channels = out_planes
        groups = 1
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=padding, groups=groups, bias=bias)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


class ChannelPool(nn.Module):
    def __init__(self):
        super(ChannelPool, self).__init__()

    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SAB(nn.Module):
    def __init__(self):
        super(SAB, self).__init__()
        kernel_size = 5
        self.compress = ChannelPool()
        self.spatial = Basic(2, 1, kernel_size, padding=(kernel_size - 1) // 2, bias=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)
        return x * scale


## Channel Attention Layer
class CAB(nn.Module):
    def __init__(self, nc, reduction=8, bias=False):
        super(CAB, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
                nn.Conv2d(nc, nc // reduction, kernel_size=1, padding=0, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(nc // reduction, nc, kernel_size=1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.adaptive = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        s_hw = self.adaptive(x).sigmoid() * 2  # n, c, 1, 1
        x_hw = s_hw * x  # n, c, h, w

        x1 = x_hw.permute(0, 2, 1, 3)  # n, h, c, w
        s_cw = self.adaptive(x1).sigmoid() * 2  # n, h, 1, 1
        x_cw = (s_cw * x1).permute(0, 2, 1, 3)  # n, h, c, w ===> n, c, h, w

        x2 = x_hw.permute(0, 3, 2, 1)  # n, w, h, c
        s_ch = self.adaptive(x2).sigmoid() * 2  # n, w, 1, 1
        x_ch = (s_ch * x2).permute(0, 3, 2, 1)  # n, w, h, c ===> n, c, h, w

        return x_cw + x_hw + x_ch


class SEKG(nn.Module):
    def __init__(self, in_channels=64, kernel_size=3):
        super().__init__()
        self.conv_sa = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, dilation=1,
                                 groups=in_channels)
        # channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_ca = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)

    def forward(self, input_x):
        b, c, h, w = input_x.size()
        # spatial attention
        sa_x = self.conv_sa(input_x)
        # channel attention
        y = self.avg_pool(input_x)
        ca_x = self.conv_ca(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        out = sa_x + ca_x
        return out


# Adaptice Filter Generation
class AFG(nn.Module):
    def __init__(self, in_channels=64, kernel_size=3):
        super(AFG, self).__init__()
        self.kernel_size = kernel_size
        self.sekg = SEKG(in_channels, kernel_size)
        self.conv = nn.Conv2d(in_channels, in_channels * kernel_size * kernel_size, 1, 1, 0)

    def forward(self, input_x):
        b, c, h, w = input_x.size()
        x = self.sekg(input_x)
        x = self.conv(x)
        filter_x = x.reshape([b, c, self.kernel_size * self.kernel_size, h, w])

        return filter_x


# Dynamic convolution
class DyConv(nn.Module):
    def __init__(self, in_channels=64, kernel_size=3):
        super(DyConv, self).__init__()
        self.kernel_size = kernel_size
        self.afg = AFG(in_channels, kernel_size)
        self.unfold = nn.Unfold(kernel_size=3, dilation=1, padding=1, stride=1)

    def forward(self, input_x):
        b, c, h, w = input_x.size()
        filter_x = self.afg(input_x)
        unfold_x = self.unfold(input_x).reshape(b, c, -1, h, w)
        out = (unfold_x * filter_x).sum(2)

        return out


# Dynamic convolution block
class DCB(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.dyconv = DyConv(channels, 3)

        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        conv1 = self.lrelu(self.dyconv(self.conv1(x)))
        conv2 = self.conv2(conv1)

        out = x + conv2
        return out


# convolution block
class CB(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.conv3 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.lrelu(self.conv2(conv1))
        conv3 = self.conv3(conv2)
        out = x + conv3
        return out


class OffsetBlock(nn.Module):
    def __init__(self, in_channels=64, offset_channels=32):
        super().__init__()
        self.offset_conv1 = nn.Conv2d(in_channels, offset_channels, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        offset = self.lrelu(self.offset_conv1(x))
        return offset


class DepthDC(nn.Module):
    def __init__(self, in_x_channels, kernel_size):
        super(DepthDC, self).__init__()
        self.unfold = nn.Unfold(kernel_size=3, dilation=kernel_size, padding=kernel_size, stride=1)
        self.fuse = nn.Conv2d(in_x_channels, in_x_channels, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x, y):
        N, xC, xH, xW = x.size()
        kernel = y.reshape([N, xC, 3 ** 2, xH, xW])
        unfold_x = self.unfold(x).reshape([N, xC, -1, xH, xW])
        out = (unfold_x * kernel).sum(2)
        out = self.lrelu(self.fuse(out))
        return out


class ADCB(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, sobel_ch=32, nc=128, bias=True):
        super(ADCB, self).__init__()
        kernel_size = 3
        # self.conv1 = nn.Conv2d(in_channels, nc, kernel_size=kernel_size, bias=bias, padding=1)
        self.bn1 = nn.BatchNorm2d(nc)
        self.relu0 = nn.ReLU(inplace=True)
        self.dcb = DCB(nc)
        self.sc = ScConv(nc)

        self.bn2 = nn.BatchNorm2d(nc)
        self.relu1 = nn.ReLU(inplace=True)

        self.m_conv_tail = nn.Conv2d(nc, out_channels, kernel_size=kernel_size, bias=bias, padding=1)

    def forward(self, x):

        # xs = self.conv_p1(xs)
        x1 = self.sc(x)
        # print('conv1 output size:', x1.size())
        x1 = self.bn1(x1)
        x1 = self.relu0(x1)
        # x2 = self.sc(x1)
        # # print('conv2 output size:', x2.size())

        x3 = self.dcb(x1)
        # x3 = self.conv1(x3)
        # x3 = self.relu2(x3)
        x5 = self.m_conv_tail(x3+x)
        x5 = F.interpolate(x5, size=x.size()[2:], mode='nearest')
        x5_1 = x + x5
        # print('x5_1 output size:', x5_1.size())
        return x5_1


class MAB(nn.Module):
    def __init__(self, in_channels, sobel_ch, mid_channels):
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels

        self.conv_sobel = SobelConv2d(in_channels, sobel_ch, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_p1 = nn.Conv2d(in_channels + sobel_ch, mid_channels, kernel_size=1, stride=1, padding=0)
        self.conv_f1 = nn.Conv2d(in_channels + sobel_ch + mid_channels, mid_channels, kernel_size=3, stride=1, padding=1)

        self.conv_p2 = nn.Conv2d(in_channels + sobel_ch + mid_channels, mid_channels, kernel_size=1, stride=1, padding=0)
        self.conv_f2 = nn.Conv2d(in_channels + sobel_ch + mid_channels, mid_channels, kernel_size=3, stride=1, padding=1)

        self.conv_p3 = nn.Conv2d(in_channels + sobel_ch + mid_channels, mid_channels, kernel_size=1, stride=1, padding=0)
        self.conv_f3 = nn.Conv2d(in_channels + sobel_ch + mid_channels, mid_channels, kernel_size=3, stride=1, padding=1)

        self.c1 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, 1)
        # self.c1 = DepthwiseSeparableConv(mid_channels, mid_channels)
        self.c2 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 3, 3)
        self.c3 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 5, 5)
        self.fusion = nn.Conv2d(mid_channels * 4, in_channels, 1, 1, 0)

        self.conv1 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, 1)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 3, 3)
        self.conv3 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 5, 5)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.sc = ScConv(mid_channels)
        self.attention = Attention()
        self.bn = nn.BatchNorm2d(mid_channels)

    def forward(self, x):

        xs = self.conv_sobel(x)
        xs = torch.cat((x, xs), dim=-3)
        x0 = self.lrelu(self.conv_p1(xs))
        out_1 = torch.cat((xs, x0), dim=-3)
        out_1 = self.lrelu(self.conv_f1(out_1))
        # print('out_1 size:', out_1.size())

        x1 = self.lrelu(self.c1(out_1))

        out_2 = torch.cat((xs, x1), dim=-3)
        out_2 = self.lrelu(self.conv_f2(out_2))
        x2 = self.lrelu(self.c2(out_2))

        out_3 = torch.cat((xs, x2), dim=-3)
        out_3 = self.lrelu(self.conv_f3(out_3))
        x3 = self.lrelu(self.c3(out_3))

        out = self.fusion(self.attention(torch.cat([x0, x1, x2, x3], dim=1)))
        out = self.lrelu(self.bn(self.sc(x + out)))
        return out


class YNet(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nc=64, sobel_ch=32, bias=True):
        super(YNet, self).__init__()
        kernel_size = 3

        self.conv_sobel = SobelConv2d(in_nc, sobel_ch, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_p1 = nn.Conv2d(in_nc + sobel_ch, nc, kernel_size=1, stride=1, padding=0)
        self.conv_head = nn.Conv2d(in_nc, nc, kernel_size=kernel_size, padding=1, bias=bias)

        self.dmr = DMR(nc, nc)

        self.adcb = ADCB(nc, nc, sobel_ch, nc, bias)
        self.mab = MAB(nc, sobel_ch, nc)

        self.conv_tail = nn.Conv2d(nc, out_nc, kernel_size=kernel_size, padding=1, bias=bias)

        self.dual_tail = nn.Conv2d(2 * out_nc, out_nc, kernel_size=kernel_size, padding=1, bias=bias)

        self.down = nn.Conv2d(nc, nc, kernel_size=2, stride=2, bias=bias)

        self.up = Up(nc, bias)

    def forward(self, x):
        # y1 = self.conv_p1(xs)
        y1 = self.conv_head(x)

        y2 = self.mab(y1)
        y3 = self.mab(y2)
        y3_x = self.down(y3)

        y4 = self.mab(y3)
        y5 = self.mab(y4 + y3)
        y5_x = self.down(y5)
        y6 = self.mab(y5 + y2)
        y7 = self.conv_tail(y6 + y1)
        Y = x - y7

        x1 = self.conv_head(x)
        x2 = self.adcb(x1)
        x2_1 = self.down(x2)

        y3_xx = self.dmr(y3_x)
        x3 = self.adcb(x2_1+y3_xx)
        x3_1 = self.down(x3)

        x4 = self.adcb(x3_1)
        x4_1 = self.up(x4, x3)
        x4_2 = self.dmr(y5_x)
        x5 = self.adcb(x4_1 + x3 + x4_2)
        x5_1 = self.up(x5, x2)
        x6 = self.adcb(x5_1 + x2)
        x7 = self.conv_tail(x6 + x1)
        X = x - x7

        z1 = torch.cat([X, Y], dim=1)

        z = self.dual_tail(z1)

        Z = x - z
        return Z

def main():
    # 创建一个Net实例
    net = YNet()

    # 打印模型的结构
    print(net)

    # 随机生成一个输入张量，用于测试
    input_tensor = torch.randn(1, 1, 256, 256)  # 假设输入尺寸为 (batch_size, in_channels, height, width)

    # 使用模型进行前向传播
    output_tensor = net(input_tensor)

    # 打印输出张量的形状
    print("Output shape:", output_tensor.shape)

if __name__ == "__main__":
    main()