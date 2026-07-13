from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.layers import DropPath, trunc_normal_

from ultralytics.nn.Pconv import PConv
from ultralytics.nn.modules import DFL
from ultralytics.nn.modules.block import ABlock, C3k, C2f, A2C2f, RepC3k, C3, RepBottleneck, Bottleneck, GhostBottleneck
from ultralytics.nn.modules.conv import Conv, DWConv, GhostConv, space_to_depth
from einops import rearrange
from typing import Optional, Sequence
from mmengine.model import BaseModule
from ultralytics.nn.modules.conv import autopad
from torch.cuda.amp import autocast
from typing import Tuple
import warnings
from ultralytics.utils.tal import dist2bbox, make_anchors


class InceptionDWConv2d(nn.Module):
    """ Inception depthweise convolution
    """

    def __init__(self, in_channels, square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125):
        super().__init__()

        gc = int(in_channels * branch_ratio)  # channel numbers of a convolution branch
        self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc)
        self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2),
                                  groups=gc)
        self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0),
                                  groups=gc)
        self.split_indexes = (in_channels - 3 * gc, gc, gc, gc)

    def forward(self, x):
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        return torch.cat(
            (x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)),
            dim=1,
        )


class FRFN(nn.Module):
    def __init__(self, dim=32, hidden_dim=128, act_layer=nn.GELU, drop=0., use_eca=False):
        super().__init__()
        self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim * 2),
                                     act_layer())
        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, groups=hidden_dim, kernel_size=3, stride=1, padding=1),
            act_layer())
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))
        self.dim = dim
        self.hidden_dim = hidden_dim

        self.dim_conv = self.dim // 4
        self.dim_untouched = self.dim - self.dim_conv
        self.partial_conv3 = nn.Conv2d(self.dim_conv, self.dim_conv, 3, 1, 1, bias=False)

    def forward(self, x):
        # bs x hw x c
        c, bs, hh, hw = x.size()
        # hh = int(math.sqrt(hw))
        #
        # # spatial restore
        # x = rearrange(x, ' b (h w) (c) -> b c h w ', h=hh, w=hh)

        x1, x2, = torch.split(x, [self.dim_conv, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)

        # flaten
        x = rearrange(x, ' b c h w -> b (h w) c', h=hh, w=hw)

        x = self.linear1(x)
        # gate mechanism
        x_1, x_2 = x.chunk(2, dim=-1)

        x_1 = rearrange(x_1, ' b (h w) (c) -> b c h w ', h=hh, w=hw)
        x_1 = self.dwconv(x_1)
        x_1 = rearrange(x_1, ' b c h w -> b (h w) c', h=hh, w=hw)
        x = x_1 * x_2

        x = self.linear2(x)
        # x = self.eca(x)

        return rearrange(x, ' b (h w) (c) -> b c h w ', h=hh, w=hw)


class DMlp(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)
        self.conv_0 = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 3, 1, 1, groups=dim),
            nn.Conv2d(hidden_dim, hidden_dim, 1, 1, 0)
        )
        self.act = nn.GELU()
        self.conv_1 = nn.Conv2d(hidden_dim, dim, 1, 1, 0)

    def forward(self, x):
        x = self.conv_0(x)
        x = self.act(x)
        x = self.conv_1(x)
        return x


class SMFA(nn.Module):
    def __init__(self, dim=36):
        super(SMFA, self).__init__()
        self.linear_0 = nn.Conv2d(dim, dim * 2, 1, 1, 0)
        self.linear_1 = nn.Conv2d(dim, dim, 1, 1, 0)
        self.linear_2 = nn.Conv2d(dim, dim, 1, 1, 0)

        self.lde = DMlp(dim, 2)

        self.dw_conv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

        self.gelu = nn.GELU()
        self.down_scale = 8

        self.alpha = nn.Parameter(torch.ones((1, dim, 1, 1)))
        self.belt = nn.Parameter(torch.zeros((1, dim, 1, 1)))

    def forward(self, f):
        _, _, h, w = f.shape
        y, x = self.linear_0(f).chunk(2, dim=1)
        x_s = self.dw_conv(F.adaptive_max_pool2d(x, (h // self.down_scale, w // self.down_scale)))
        x_v = torch.var(x, dim=(-2, -1), keepdim=True)
        x_l = x * F.interpolate(self.gelu(self.linear_1(x_s * self.alpha + x_v * self.belt)), size=(h, w),
                                mode='nearest')
        y_d = self.lde(y)
        return self.linear_2(x_l + y_d)



class AKConv(nn.Module):
    def __init__(self, inc, outc, stride=1, num_param=5, bias=None):
        """
        初始化参数说明：
            inc: 输入通道数, outc: 输出通道数, num_param:（卷积核）参数量, stride = 1:卷积步长默认为1, bias = None：默认无偏执
            """
        super(AKConv, self).__init__()
        self.num_param = num_param
        self.stride = stride
        self.conv = Conv(inc, outc, k=(num_param, 1), s=(num_param, 1))
        self.p_conv = nn.Conv2d(inc, 2 * num_param, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_full_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        # N is num_param.
        offset = self.p_conv(x)
        dtype = offset.data.type()
        N = offset.size(1) // 2
        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2) - 1), torch.clamp(q_lt[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2) - 1), torch.clamp(q_rb[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2) - 1), torch.clamp(p[..., N:], 0, x.size(3) - 1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # resampling the features based on the modified coordinates.
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # bilinear
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        x_offset = self._reshape_x_offset(x_offset, self.num_param)
        out = self.conv(x_offset)

        return out

    # generating the inital sampled shapes for the AKConv with different sizes.
    def _get_p_n(self, N, dtype):
        base_int = round(math.sqrt(self.num_param))
        row_number = self.num_param // base_int
        mod_number = self.num_param % base_int
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(0, row_number),
            torch.arange(0, base_int), indexing='xy')
        p_n_x = torch.flatten(p_n_x)
        p_n_y = torch.flatten(p_n_y)
        if mod_number > 0:
            mod_p_n_x, mod_p_n_y = torch.meshgrid(
                torch.arange(row_number, row_number + 1),
                torch.arange(0, mod_number), indexing='xy')

            mod_p_n_x = torch.flatten(mod_p_n_x)
            mod_p_n_y = torch.flatten(mod_p_n_y)
            p_n_x, p_n_y = torch.cat((p_n_x, mod_p_n_x)), torch.cat((p_n_y, mod_p_n_y))
        p_n = torch.cat([p_n_x, p_n_y], 0)
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)
        return p_n

    # no zero-padding
    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(0, h * self.stride, self.stride),
            torch.arange(0, w * self.stride, self.stride), indexing='xy')

        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N] * padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    #  Stacking resampled features in the row direction.
    @staticmethod
    def _reshape_x_offset(x_offset, num_param):
        b, c, h, w, n = x_offset.size()
        x_offset = rearrange(x_offset, 'b c h w n -> b c (h n) w')
        return x_offset


class GroupBatchnorm2d(nn.Module):
    def __init__(self, c_num: int,
                 group_num: int = 16,
                 eps: float = 1e-10
                 ):
        super(GroupBatchnorm2d, self).__init__()
        assert c_num >= group_num
        self.group_num = group_num
        self.weight = nn.Parameter(torch.randn(c_num, 1, 1))
        self.bias = nn.Parameter(torch.zeros(c_num, 1, 1))
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, self.group_num, -1)
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)
        x = (x - mean) / (std + self.eps)
        x = x.view(N, C, H, W)
        return x * self.weight + self.bias


class SRU(nn.Module):
    def __init__(self,
                 oup_channels: int,
                 group_num: int = 16,
                 gate_treshold: float = 0.5,
                 torch_gn: bool = True
                 ):
        super().__init__()

        self.gn = nn.GroupNorm(num_channels=oup_channels, num_groups=group_num) if torch_gn else GroupBatchnorm2d(
            c_num=oup_channels, group_num=group_num)
        self.gate_treshold = gate_treshold
        self.sigomid = nn.Sigmoid()

    def forward(self, x):
        gn_x = self.gn(x)
        w_gamma = self.gn.weight / sum(self.gn.weight)
        w_gamma = w_gamma.view(1, -1, 1, 1)
        reweigts = self.sigomid(gn_x * w_gamma)
        # Gate
        w1 = torch.where(reweigts > self.gate_treshold, torch.ones_like(reweigts), reweigts)  # 大于门限值的设为1，否则保留原值
        w2 = torch.where(reweigts > self.gate_treshold, torch.zeros_like(reweigts), reweigts)  # 大于门限值的设为0，否则保留原值
        x_1 = w1 * x
        x_2 = w2 * x
        y = self.reconstruct(x_1, x_2)
        return y

    def reconstruct(self, x_1, x_2):
        x_11, x_12 = torch.split(x_1, x_1.size(1) // 2, dim=1)
        x_21, x_22 = torch.split(x_2, x_2.size(1) // 2, dim=1)
        return torch.cat([x_11 + x_22, x_12 + x_21], dim=1)


class CRU(nn.Module):
    '''
    alpha: 0<alpha<1
    '''

    def __init__(self,
                 op_channel: int,
                 alpha: float = 1 / 2,
                 squeeze_radio: int = 2,
                 group_size: int = 2,
                 group_kernel_size: int = 3,
                 ):
        super().__init__()
        self.up_channel = up_channel = int(alpha * op_channel)
        self.low_channel = low_channel = op_channel - up_channel
        self.squeeze1 = nn.Conv2d(up_channel, up_channel // squeeze_radio, kernel_size=1, bias=False)
        self.squeeze2 = nn.Conv2d(low_channel, low_channel // squeeze_radio, kernel_size=1, bias=False)
        # up
        self.GWC = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=group_kernel_size, stride=1,
                             padding=group_kernel_size // 2, groups=group_size)
        self.PWC1 = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=1, bias=False)
        # low
        self.PWC2 = nn.Conv2d(low_channel // squeeze_radio, op_channel - low_channel // squeeze_radio, kernel_size=1,
                              bias=False)
        self.advavg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # Split
        up, low = torch.split(x, [self.up_channel, self.low_channel], dim=1)
        up, low = self.squeeze1(up), self.squeeze2(low)
        # Transform
        Y1 = self.GWC(up) + self.PWC1(up)
        Y2 = torch.cat([self.PWC2(low), low], dim=1)
        # Fuse
        out = torch.cat([Y1, Y2], dim=1)
        out = F.softmax(self.advavg(out), dim=1) * out
        out1, out2 = torch.split(out, out.size(1) // 2, dim=1)
        return out1 + out2


class ScConv(nn.Module):
    def __init__(self,
                 op_channel: int,
                 group_num: int = 4,
                 gate_treshold: float = 0.5,
                 alpha: float = 1 / 2,
                 squeeze_radio: int = 2,
                 group_size: int = 2,
                 group_kernel_size: int = 3,
                 ):
        super().__init__()
        self.SRU = SRU(op_channel,
                       group_num=group_num,
                       gate_treshold=gate_treshold)
        self.CRU = CRU(op_channel,
                       alpha=alpha,
                       squeeze_radio=squeeze_radio,
                       group_size=group_size,
                       group_kernel_size=group_kernel_size)

    def forward(self, x):
        x = self.SRU(x)
        x = self.CRU(x)
        return x


class DSC(object):
    def __init__(self, input_shape, kernel_size, extend_scope, morph):
        self.num_points = kernel_size
        self.width = input_shape[2]
        self.height = input_shape[3]
        self.morph = morph
        self.extend_scope = extend_scope  # offset (-1 ~ 1) * extend_scope

        # define feature map shape
        """
        B: Batch size  C: Channel  W: Width  H: Height
        """
        self.num_batch = input_shape[0]
        self.num_channels = input_shape[1]

    """
    input: offset [B,2*K,W,H]  K: Kernel size (2*K: 2D image, deformation contains <x_offset> and <y_offset>)
    output_x: [B,1,W,K*H]   coordinate map
    output_y: [B,1,K*W,H]   coordinate map
    """

    def _coordinate_map_3D(self, offset, if_offset):
        device = offset.device
        # offset
        y_offset, x_offset = torch.split(offset, self.num_points, dim=1)

        y_center = torch.arange(0, self.width).repeat([self.height])
        y_center = y_center.reshape(self.height, self.width)
        y_center = y_center.permute(1, 0)
        y_center = y_center.reshape([-1, self.width, self.height])
        y_center = y_center.repeat([self.num_points, 1, 1]).float()
        y_center = y_center.unsqueeze(0)

        x_center = torch.arange(0, self.height).repeat([self.width])
        x_center = x_center.reshape(self.width, self.height)
        x_center = x_center.permute(0, 1)
        x_center = x_center.reshape([-1, self.width, self.height])
        x_center = x_center.repeat([self.num_points, 1, 1]).float()
        x_center = x_center.unsqueeze(0)

        if self.morph == 0:
            """
            Initialize the kernel and flatten the kernel
                y: only need 0
                x: -num_points//2 ~ num_points//2 (Determined by the kernel size)
                !!! The related PPT will be submitted later, and the PPT will contain the whole changes of each step
            """
            y = torch.linspace(0, 0, 1)
            x = torch.linspace(
                -int(self.num_points // 2),
                int(self.num_points // 2),
                int(self.num_points),
            )

            y, x = torch.meshgrid(y, x, indexing="ij")
            y_spread = y.reshape(-1, 1)
            x_spread = x.reshape(-1, 1)

            y_grid = y_spread.repeat([1, self.width * self.height])
            y_grid = y_grid.reshape([self.num_points, self.width, self.height])
            y_grid = y_grid.unsqueeze(0)  # [B*K*K, W,H]

            x_grid = x_spread.repeat([1, self.width * self.height])
            x_grid = x_grid.reshape([self.num_points, self.width, self.height])
            x_grid = x_grid.unsqueeze(0)  # [B*K*K, W,H]

            y_new = y_center + y_grid
            x_new = x_center + x_grid

            y_new = y_new.repeat(self.num_batch, 1, 1, 1).to(device)
            x_new = x_new.repeat(self.num_batch, 1, 1, 1).to(device)

            y_offset_new = y_offset.detach().clone()

            if if_offset:
                y_offset = y_offset.permute(1, 0, 2, 3)
                y_offset_new = y_offset_new.permute(1, 0, 2, 3)
                center = int(self.num_points // 2)

                # The center position remains unchanged and the rest of the positions begin to swing
                # This part is quite simple. The main idea is that "offset is an iterative process"
                y_offset_new[center] = 0
                for index in range(1, center):
                    y_offset_new[center + index] = (y_offset_new[center + index - 1] + y_offset[center + index])
                    y_offset_new[center - index] = (y_offset_new[center - index + 1] + y_offset[center - index])
                y_offset_new = y_offset_new.permute(1, 0, 2, 3).to(device)
                y_new = y_new.add(y_offset_new.mul(self.extend_scope))

            y_new = y_new.reshape(
                [self.num_batch, self.num_points, 1, self.width, self.height])
            y_new = y_new.permute(0, 3, 1, 4, 2)
            y_new = y_new.reshape([
                self.num_batch, self.num_points * self.width, 1 * self.height
            ])
            x_new = x_new.reshape(
                [self.num_batch, self.num_points, 1, self.width, self.height])
            x_new = x_new.permute(0, 3, 1, 4, 2)
            x_new = x_new.reshape([
                self.num_batch, self.num_points * self.width, 1 * self.height
            ])
            return y_new, x_new

        else:
            """
            Initialize the kernel and flatten the kernel
                y: -num_points//2 ~ num_points//2 (Determined by the kernel size)
                x: only need 0
            """
            y = torch.linspace(
                -int(self.num_points // 2),
                int(self.num_points // 2),
                int(self.num_points),
            )
            x = torch.linspace(0, 0, 1)

            y, x = torch.meshgrid(y, x, indexing="ij")
            y_spread = y.reshape(-1, 1)
            x_spread = x.reshape(-1, 1)

            y_grid = y_spread.repeat([1, self.width * self.height])
            y_grid = y_grid.reshape([self.num_points, self.width, self.height])
            y_grid = y_grid.unsqueeze(0)

            x_grid = x_spread.repeat([1, self.width * self.height])
            x_grid = x_grid.reshape([self.num_points, self.width, self.height])
            x_grid = x_grid.unsqueeze(0)

            y_new = y_center + y_grid
            x_new = x_center + x_grid

            y_new = y_new.repeat(self.num_batch, 1, 1, 1)
            x_new = x_new.repeat(self.num_batch, 1, 1, 1)

            y_new = y_new.to(device)
            x_new = x_new.to(device)
            x_offset_new = x_offset.detach().clone()

            if if_offset:
                x_offset = x_offset.permute(1, 0, 2, 3)
                x_offset_new = x_offset_new.permute(1, 0, 2, 3)
                center = int(self.num_points // 2)
                x_offset_new[center] = 0
                for index in range(1, center):
                    x_offset_new[center + index] = (x_offset_new[center + index - 1] + x_offset[center + index])
                    x_offset_new[center - index] = (x_offset_new[center - index + 1] + x_offset[center - index])
                x_offset_new = x_offset_new.permute(1, 0, 2, 3).to(device)
                x_new = x_new.add(x_offset_new.mul(self.extend_scope))

            y_new = y_new.reshape(
                [self.num_batch, 1, self.num_points, self.width, self.height])
            y_new = y_new.permute(0, 3, 1, 4, 2)
            y_new = y_new.reshape([
                self.num_batch, 1 * self.width, self.num_points * self.height
            ])
            x_new = x_new.reshape(
                [self.num_batch, 1, self.num_points, self.width, self.height])
            x_new = x_new.permute(0, 3, 1, 4, 2)
            x_new = x_new.reshape([
                self.num_batch, 1 * self.width, self.num_points * self.height
            ])
            return y_new, x_new

    """
    input: input feature map [N,C,D,W,H]；coordinate map [N,K*D,K*W,K*H] 
    output: [N,1,K*D,K*W,K*H]  deformed feature map
    """

    def _bilinear_interpolate_3D(self, input_feature, y, x):
        device = input_feature.device
        y = y.reshape([-1]).float()
        x = x.reshape([-1]).float()

        zero = torch.zeros([]).int()
        max_y = self.width - 1
        max_x = self.height - 1

        # find 8 grid locations
        y0 = torch.floor(y).int()
        y1 = y0 + 1
        x0 = torch.floor(x).int()
        x1 = x0 + 1

        # clip out coordinates exceeding feature map volume
        y0 = torch.clamp(y0, zero, max_y)
        y1 = torch.clamp(y1, zero, max_y)
        x0 = torch.clamp(x0, zero, max_x)
        x1 = torch.clamp(x1, zero, max_x)

        input_feature_flat = input_feature.flatten()
        input_feature_flat = input_feature_flat.reshape(
            self.num_batch, self.num_channels, self.width, self.height)
        input_feature_flat = input_feature_flat.permute(0, 2, 3, 1)
        input_feature_flat = input_feature_flat.reshape(-1, self.num_channels)
        dimension = self.height * self.width

        base = torch.arange(self.num_batch) * dimension
        base = base.reshape([-1, 1]).float()

        repeat = torch.ones([self.num_points * self.width * self.height
                             ]).unsqueeze(0)
        repeat = repeat.float()

        base = torch.matmul(base, repeat)
        base = base.reshape([-1])

        base = base.to(device)

        base_y0 = base + y0 * self.height
        base_y1 = base + y1 * self.height

        # top rectangle of the neighbourhood volume
        index_a0 = base_y0 - base + x0
        index_c0 = base_y0 - base + x1

        # bottom rectangle of the neighbourhood volume
        index_a1 = base_y1 - base + x0
        index_c1 = base_y1 - base + x1

        # get 8 grid values
        value_a0 = input_feature_flat[index_a0.type(torch.int64)].to(device)
        value_c0 = input_feature_flat[index_c0.type(torch.int64)].to(device)
        value_a1 = input_feature_flat[index_a1.type(torch.int64)].to(device)
        value_c1 = input_feature_flat[index_c1.type(torch.int64)].to(device)

        # find 8 grid locations
        y0 = torch.floor(y).int()
        y1 = y0 + 1
        x0 = torch.floor(x).int()
        x1 = x0 + 1

        # clip out coordinates exceeding feature map volume
        y0 = torch.clamp(y0, zero, max_y + 1)
        y1 = torch.clamp(y1, zero, max_y + 1)
        x0 = torch.clamp(x0, zero, max_x + 1)
        x1 = torch.clamp(x1, zero, max_x + 1)

        x0_float = x0.float()
        x1_float = x1.float()
        y0_float = y0.float()
        y1_float = y1.float()

        vol_a0 = ((y1_float - y) * (x1_float - x)).unsqueeze(-1).to(device)
        vol_c0 = ((y1_float - y) * (x - x0_float)).unsqueeze(-1).to(device)
        vol_a1 = ((y - y0_float) * (x1_float - x)).unsqueeze(-1).to(device)
        vol_c1 = ((y - y0_float) * (x - x0_float)).unsqueeze(-1).to(device)

        outputs = (value_a0 * vol_a0 + value_c0 * vol_c0 + value_a1 * vol_a1 +
                   value_c1 * vol_c1)

        if self.morph == 0:
            outputs = outputs.reshape([
                self.num_batch,
                self.num_points * self.width,
                1 * self.height,
                self.num_channels,
            ])
            outputs = outputs.permute(0, 3, 1, 2)
        else:
            outputs = outputs.reshape([
                self.num_batch,
                1 * self.width,
                self.num_points * self.height,
                self.num_channels,
            ])
            outputs = outputs.permute(0, 3, 1, 2)
        return outputs

    def deform_conv(self, input, offset, if_offset):
        y, x = self._coordinate_map_3D(offset, if_offset)
        deformed_feature = self._bilinear_interpolate_3D(input, y, x)
        return deformed_feature


class DSConv1(nn.Module):
    def __init__(self, in_ch, out_ch, morph, kernel_size=3, if_offset=True, extend_scope=1):
        """
        The Dynamic Snake Convolution
        :param in_ch: input channel
        :param out_ch: output channel
        :param kernel_size: the size of kernel
        :param extend_scope: the range to expand (default 1 for this method)
        :param morph: the morphology of the convolution kernel is mainly divided into two types
                        along the x-axis (0) and the y-axis (1) (see the paper for details)
        :param if_offset: whether deformation is required, if it is False, it is the standard convolution kernel
        """
        super(DSConv1, self).__init__()
        # use the <offset_conv> to learn the deformable offset
        self.offset_conv = nn.Conv2d(in_ch, 2 * kernel_size, 3, padding=1)
        self.bn = nn.BatchNorm2d(2 * kernel_size)
        self.kernel_size = kernel_size

        # two types of the DSConv (along x-axis and y-axis)
        self.dsc_conv_x = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=(kernel_size, 1),
            stride=(kernel_size, 1),
            padding=0,
        )
        self.dsc_conv_y = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=(1, kernel_size),
            stride=(1, kernel_size),
            padding=0,
        )

        self.gn = nn.GroupNorm([i + 1 for i in range(4) if out_ch % (i + 1) == 0][-1], out_ch)
        self.act = Conv.default_act

        self.extend_scope = extend_scope
        self.morph = morph
        self.if_offset = if_offset

    def forward(self, f):
        offset = self.offset_conv(f)
        offset = self.bn(offset)
        # We need a range of deformation between -1 and 1 to mimic the snake's swing
        offset = torch.tanh(offset)
        input_shape = f.shape
        dsc = DSC(input_shape, self.kernel_size, self.extend_scope, self.morph)
        deformed_feature = dsc.deform_conv(f, offset, self.if_offset)
        if self.morph == 0:
            x = self.dsc_conv_x(deformed_feature.type(f.dtype))
            x = self.gn(x)
            x = self.act(x)
            return x
        else:
            x = self.dsc_conv_y(deformed_feature.type(f.dtype))
            x = self.gn(x)
            x = self.act(x)
            return x


class DySnakeConv(nn.Module):
    def __init__(self, inc, ouc, k=5) -> None:
        super().__init__()
        c_ = ouc // 3
        self.conv_0 = Conv(inc, ouc - 2 * c_, k)
        self.conv_x = DSConv1(inc, c_, 0, k)
        self.conv_y = DSConv1(inc, c_, 1, k)

    def forward(self, x):
        return torch.cat([self.conv_0(x), self.conv_x(x), self.conv_y(x)], dim=1)


class SE(nn.Module):

    def __init__(self, channel=512, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class iRMB(nn.Module):
    def __init__(self, dim_in, dim_out, dim_head=32, norm_in=True, has_skip=True, exp_ratio=1.0,
                 act_layer='relu', v_proj=True, dw_ks=3, stride=1, dilation=1, se_ratio=0.0, window_size=7,
                 attn_s=True, attn_drop=0., drop=0., drop_path=0., v_group=False, attn_pre=False):
        super().__init__()
        self.norm = nn.BatchNorm2d(dim_in) if norm_in else nn.Identity()
        dim_mid = int(dim_in * exp_ratio)
        self.has_skip = (dim_in == dim_out and stride == 1) and has_skip
        self.attn_s = attn_s
        if self.attn_s:
            assert dim_in % dim_head == 0, 'dim should be divisible by num_heads'
            self.dim_head = dim_head
            self.window_size = window_size
            self.num_head = dim_in // dim_head
            self.scale = self.dim_head ** -0.5
            self.attn_pre = attn_pre
            self.qk = Conv(dim_in, int(dim_in * 2), k=1, act=False)
            self.v = Conv(dim_in, dim_mid, k=1, g=self.num_head if v_group else 1, act=False)
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            if v_proj:
                self.v = Conv(dim_in, dim_mid, k=1, act=act_layer)
            else:
                self.v = nn.Identity()
        self.conv_local = Conv(dim_mid, dim_mid, k=dw_ks, s=stride, d=dilation, g=dim_mid, )
        self.se = SE(dim_mid, reduction=se_ratio) if se_ratio > 0.0 else nn.Identity()

        self.proj_drop = nn.Dropout(drop)
        self.proj = Conv(dim_mid, dim_out, k=1, act=False)
        self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.norm(x)
        B, C, H, W = x.shape
        if self.attn_s:
            # padding
            if self.window_size <= 0:
                window_size_W, window_size_H = W, H
            else:
                window_size_W, window_size_H = self.window_size, self.window_size
            pad_l, pad_t = 0, 0
            pad_r = (window_size_W - W % window_size_W) % window_size_W
            pad_b = (window_size_H - H % window_size_H) % window_size_H
            x = F.pad(x, (pad_l, pad_r, pad_t, pad_b, 0, 0,))
            n1, n2 = (H + pad_b) // window_size_H, (W + pad_r) // window_size_W
            x = rearrange(x, 'b c (h1 n1) (w1 n2) -> (b n1 n2) c h1 w1', n1=n1, n2=n2).contiguous()
            # attention
            b, c, h, w = x.shape
            qk = self.qk(x)
            qk = rearrange(qk, 'b (qk heads dim_head) h w -> qk b heads (h w) dim_head', qk=2, heads=self.num_head,
                           dim_head=self.dim_head).contiguous()
            q, k = qk[0], qk[1]
            attn_spa = (q @ k.transpose(-2, -1)) * self.scale
            attn_spa = attn_spa.softmax(dim=-1)
            attn_spa = self.attn_drop(attn_spa)
            if self.attn_pre:
                x = rearrange(x, 'b (heads dim_head) h w -> b heads (h w) dim_head', heads=self.num_head).contiguous()
                x_spa = attn_spa @ x
                x_spa = rearrange(x_spa, 'b heads (h w) dim_head -> b (heads dim_head) h w', heads=self.num_head, h=h,
                                  w=w).contiguous()
                x_spa = self.v(x_spa)
            else:
                v = self.v(x)
                v = rearrange(v, 'b (heads dim_head) h w -> b heads (h w) dim_head', heads=self.num_head).contiguous()
                x_spa = attn_spa @ v
                x_spa = rearrange(x_spa, 'b heads (h w) dim_head -> b (heads dim_head) h w', heads=self.num_head, h=h,
                                  w=w).contiguous()
            # unpadding
            x = rearrange(x_spa, '(b n1 n2) c h1 w1 -> b c (h1 n1) (w1 n2)', n1=n1, n2=n2).contiguous()
            if pad_r > 0 or pad_b > 0:
                x = x[:, :, :H, :W].contiguous()
        else:
            x = self.v(x)

        x = x + self.se(self.conv_local(x)) if self.has_skip else self.se(self.conv_local(x))

        x = self.proj_drop(x)
        x = self.proj(x)

        x = (shortcut + self.drop_path(x)) if self.has_skip else x
        return x


class BasicConv_FFCA(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv_FFCA, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class FEM(nn.Module):
    def __init__(self, c1, c2, stride=1, scale=0.1, map_reduce=8):
        super(FEM, self).__init__()
        self.scale = scale
        self.out_channels = c2
        inter_planes = c1 // map_reduce
        self.branch0 = nn.Sequential(
            BasicConv_FFCA(c1, 2 * inter_planes, kernel_size=1, stride=stride),
            BasicConv_FFCA(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=1, relu=False)
        )
        self.branch1 = nn.Sequential(
            BasicConv_FFCA(c1, inter_planes, kernel_size=1, stride=1),
            BasicConv_FFCA(inter_planes, (inter_planes // 2) * 3, kernel_size=(1, 3), stride=stride, padding=(0, 1)),
            BasicConv_FFCA((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=(3, 1), stride=stride,
                           padding=(1, 0)),
            BasicConv_FFCA(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=5, dilation=5,
                           relu=False)
        )
        self.branch2 = nn.Sequential(
            BasicConv_FFCA(c1, inter_planes, kernel_size=1, stride=1),
            BasicConv_FFCA(inter_planes, (inter_planes // 2) * 3, kernel_size=(3, 1), stride=stride, padding=(1, 0)),
            BasicConv_FFCA((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=(1, 3), stride=stride,
                           padding=(0, 1)),
            BasicConv_FFCA(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=5, dilation=5,
                           relu=False)
        )

        self.ConvLinear = BasicConv_FFCA(6 * inter_planes, c2, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv_FFCA(c1, c2, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        out = torch.cat((x0, x1, x2), 1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out * self.scale + short
        out = self.relu(out)

        return out


class SKFusion(nn.Module):
    def __init__(self, dim, height=2, reduction=8):
        super(SKFusion, self).__init__()

        self.height = height
        d = max(int(dim / reduction), 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, d, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(d, dim * height, 1, bias=False)
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, in_feats):
        B, C, H, W = in_feats[0].shape

        in_feats = torch.cat(in_feats, dim=1)
        in_feats = in_feats.view(B, self.height, C, H, W)

        feats_sum = torch.sum(in_feats, dim=1)
        attn = self.mlp(self.avg_pool(feats_sum))
        attn = self.softmax(attn.view(B, self.height, C, 1, 1))

        out = torch.sum(in_feats * attn, dim=1)
        return out


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)
    else:
        return F.relu6(x + 3.) / 6.


class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None,
                 act_layer=nn.ReLU, gate_fn=hard_sigmoid, divisor=4, **_):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x


class ConvBnAct(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size,
                 stride=1, act_layer=nn.ReLU):
        super(ConvBnAct, self).__init__()
        self.conv = nn.Conv2d(in_chs, out_chs, kernel_size, stride, kernel_size // 2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_chs)
        self.act1 = act_layer(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.act1(x)
        return x


def gcd(a, b):
    if a < b:
        a, b = b, a
    while (a % b != 0):
        c = a % b
        a = b
        b = c
    return b


def MyNorm(dim):
    return nn.GroupNorm(1, dim)


class GhostModuleV3(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True, mode='ori', args=None):
        super(GhostModuleV3, self).__init__()
        # self.args=args
        # mode = 'ori_shortcut_mul_conv15'
        self.mode = mode
        self.gate_loc = 'before'

        self.inter_mode = 'nearest'
        self.scale = 1.0

        self.infer_mode = False
        self.num_conv_branches = 3
        self.dconv_scale = True
        self.gate_fn = nn.Sigmoid()

        # if args.gate_fn=='hard_sigmoid':
        #     self.gate_fn=hard_sigmoid
        # elif args.gate_fn=='sigmoid':
        #     self.gate_fn=nn.Sigmoid()
        # elif args.gate_fn=='relu':
        #     self.gate_fn=nn.ReLU()
        # elif args.gate_fn=='clip':
        #     self.gate_fn=myclip
        # elif args.gate_fn=='tanh':
        #     self.gate_fn=nn.Tanh()

        if self.mode in ['ori']:
            self.oup = oup
            init_channels = math.ceil(oup / ratio)
            new_channels = init_channels * (ratio - 1)
            if self.infer_mode:
                self.primary_conv = nn.Sequential(
                    nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
                    nn.BatchNorm2d(init_channels),
                    nn.ReLU(inplace=True) if relu else nn.Sequential(),
                )
                self.cheap_operation = nn.Sequential(
                    nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
                    nn.BatchNorm2d(new_channels),
                    nn.ReLU(inplace=True) if relu else nn.Sequential(),
                )
            else:
                self.primary_rpr_skip = nn.BatchNorm2d(inp) \
                    if inp == init_channels and stride == 1 else None
                primary_rpr_conv = list()
                for _ in range(self.num_conv_branches):
                    primary_rpr_conv.append(
                        self._conv_bn(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False))
                self.primary_rpr_conv = nn.ModuleList(primary_rpr_conv)
                # Re-parameterizable scale branch
                self.primary_rpr_scale = None
                if kernel_size > 1:
                    self.primary_rpr_scale = self._conv_bn(inp, init_channels, 1, 1, 0, bias=False)
                self.primary_activation = nn.ReLU(inplace=True) if relu else None

                self.cheap_rpr_skip = nn.BatchNorm2d(init_channels) \
                    if init_channels == new_channels else None
                cheap_rpr_conv = list()
                for _ in range(self.num_conv_branches):
                    cheap_rpr_conv.append(
                        self._conv_bn(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels,
                                      bias=False))
                self.cheap_rpr_conv = nn.ModuleList(cheap_rpr_conv)
                # Re-parameterizable scale branch
                self.cheap_rpr_scale = None
                if dw_size > 1:
                    self.cheap_rpr_scale = self._conv_bn(init_channels, new_channels, 1, 1, 0, groups=init_channels,
                                                         bias=False)
                self.cheap_activation = nn.ReLU(inplace=True) if relu else None
                self.in_channels = init_channels
                self.groups = init_channels
                self.kernel_size = dw_size

        elif self.mode in ['ori_shortcut_mul_conv15']:
            self.oup = oup
            init_channels = math.ceil(oup / ratio)
            new_channels = init_channels * (ratio - 1)
            self.short_conv = nn.Sequential(
                nn.Conv2d(inp, oup, kernel_size, stride, kernel_size // 2, bias=False),
                nn.BatchNorm2d(oup),
                nn.Conv2d(oup, oup, kernel_size=(1, 5), stride=1, padding=(0, 2), groups=oup, bias=False),
                nn.BatchNorm2d(oup),
                nn.Conv2d(oup, oup, kernel_size=(5, 1), stride=1, padding=(2, 0), groups=oup, bias=False),
                nn.BatchNorm2d(oup),
            )
            if self.infer_mode:
                self.primary_conv = nn.Sequential(
                    nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
                    nn.BatchNorm2d(init_channels),
                    nn.ReLU(inplace=True) if relu else nn.Sequential(),
                )
                self.cheap_operation = nn.Sequential(
                    nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
                    nn.BatchNorm2d(new_channels),
                    nn.ReLU(inplace=True) if relu else nn.Sequential(),
                )
            else:
                self.primary_rpr_skip = nn.BatchNorm2d(inp) \
                    if inp == init_channels and stride == 1 else None
                primary_rpr_conv = list()
                for _ in range(self.num_conv_branches):
                    primary_rpr_conv.append(
                        self._conv_bn(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False))
                self.primary_rpr_conv = nn.ModuleList(primary_rpr_conv)
                # Re-parameterizable scale branch
                self.primary_rpr_scale = None
                if kernel_size > 1:
                    self.primary_rpr_scale = self._conv_bn(inp, init_channels, 1, 1, 0, bias=False)
                self.primary_activation = nn.ReLU(inplace=True) if relu else None

                self.cheap_rpr_skip = nn.BatchNorm2d(init_channels) \
                    if init_channels == new_channels else None
                cheap_rpr_conv = list()
                for _ in range(self.num_conv_branches):
                    cheap_rpr_conv.append(
                        self._conv_bn(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels,
                                      bias=False))
                self.cheap_rpr_conv = nn.ModuleList(cheap_rpr_conv)
                # Re-parameterizable scale branch
                self.cheap_rpr_scale = None
                if dw_size > 1:
                    self.cheap_rpr_scale = self._conv_bn(init_channels, new_channels, 1, 1, 0, groups=init_channels,
                                                         bias=False)
                self.cheap_activation = nn.ReLU(inplace=True) if relu else None
                self.in_channels = init_channels
                self.groups = init_channels
                self.kernel_size = dw_size

    def forward(self, x):
        if self.mode in ['ori']:
            if self.infer_mode:
                x1 = self.primary_conv(x)
                x2 = self.cheap_operation(x1)
            else:
                identity_out = 0
                if self.primary_rpr_skip is not None:
                    identity_out = self.primary_rpr_skip(x)
                scale_out = 0
                if self.primary_rpr_scale is not None and self.dconv_scale:
                    scale_out = self.primary_rpr_scale(x)
                x1 = scale_out + identity_out
                for ix in range(self.num_conv_branches):
                    x1 += self.primary_rpr_conv[ix](x)
                if self.primary_activation is not None:
                    x1 = self.primary_activation(x1)

                cheap_identity_out = 0
                if self.cheap_rpr_skip is not None:
                    cheap_identity_out = self.cheap_rpr_skip(x1)
                cheap_scale_out = 0
                if self.cheap_rpr_scale is not None and self.dconv_scale:
                    cheap_scale_out = self.cheap_rpr_scale(x1)
                x2 = cheap_scale_out + cheap_identity_out
                for ix in range(self.num_conv_branches):
                    x2 += self.cheap_rpr_conv[ix](x1)
                if self.cheap_activation is not None:
                    x2 = self.cheap_activation(x2)

            out = torch.cat([x1, x2], dim=1)
            return out

        elif self.mode in ['ori_shortcut_mul_conv15']:
            res = self.short_conv(F.avg_pool2d(x, kernel_size=2, stride=2))

            if self.infer_mode:
                x1 = self.primary_conv(x)
                x2 = self.cheap_operation(x1)
            else:
                identity_out = 0
                if self.primary_rpr_skip is not None:
                    identity_out = self.primary_rpr_skip(x)
                scale_out = 0
                if self.primary_rpr_scale is not None and self.dconv_scale:
                    scale_out = self.primary_rpr_scale(x)
                x1 = scale_out + identity_out
                for ix in range(self.num_conv_branches):
                    x1 += self.primary_rpr_conv[ix](x)
                if self.primary_activation is not None:
                    x1 = self.primary_activation(x1)

                cheap_identity_out = 0
                if self.cheap_rpr_skip is not None:
                    cheap_identity_out = self.cheap_rpr_skip(x1)
                cheap_scale_out = 0
                if self.cheap_rpr_scale is not None and self.dconv_scale:
                    cheap_scale_out = self.cheap_rpr_scale(x1)
                x2 = cheap_scale_out + cheap_identity_out
                for ix in range(self.num_conv_branches):
                    x2 += self.cheap_rpr_conv[ix](x1)
                if self.cheap_activation is not None:
                    x2 = self.cheap_activation(x2)

            out = torch.cat([x1, x2], dim=1)

            if self.gate_loc == 'before':
                return out[:, :self.oup, :, :] * F.interpolate(self.gate_fn(res / self.scale), size=out.shape[-2:],
                                                               mode=self.inter_mode)  # 'nearest'
            #                 return out*F.interpolate(self.gate_fn(res/self.scale),size=out.shape[-1].item(),mode=self.inter_mode) # 'nearest'
            else:
                return out[:, :self.oup, :, :] * self.gate_fn(
                    F.interpolate(res, size=out.shape[-2:], mode=self.inter_mode))
            #                 return out*self.gate_fn(F.interpolate(res,size=out.shape[-1],mode=self.inter_mode))

    def reparameterize(self):
        """ Following works like `RepVGG: Making VGG-style ConvNets Great Again` -
        https://arxiv.org/pdf/2101.03697.pdf. We re-parameterize multi-branched
        architecture used at training time to obtain a plain CNN-like structure
        for inference.
        """
        if self.infer_mode:
            return
        primary_kernel, primary_bias = self._get_kernel_bias_primary()
        self.primary_conv = nn.Conv2d(in_channels=self.primary_rpr_conv[0].conv.in_channels,
                                      out_channels=self.primary_rpr_conv[0].conv.out_channels,
                                      kernel_size=self.primary_rpr_conv[0].conv.kernel_size,
                                      stride=self.primary_rpr_conv[0].conv.stride,
                                      padding=self.primary_rpr_conv[0].conv.padding,
                                      dilation=self.primary_rpr_conv[0].conv.dilation,
                                      groups=self.primary_rpr_conv[0].conv.groups,
                                      bias=True)
        self.primary_conv.weight.data = primary_kernel
        self.primary_conv.bias.data = primary_bias
        self.primary_conv = nn.Sequential(
            self.primary_conv,
            self.primary_activation if self.primary_activation is not None else nn.Sequential()
        )

        cheap_kernel, cheap_bias = self._get_kernel_bias_cheap()
        self.cheap_operation = nn.Conv2d(in_channels=self.cheap_rpr_conv[0].conv.in_channels,
                                         out_channels=self.cheap_rpr_conv[0].conv.out_channels,
                                         kernel_size=self.cheap_rpr_conv[0].conv.kernel_size,
                                         stride=self.cheap_rpr_conv[0].conv.stride,
                                         padding=self.cheap_rpr_conv[0].conv.padding,
                                         dilation=self.cheap_rpr_conv[0].conv.dilation,
                                         groups=self.cheap_rpr_conv[0].conv.groups,
                                         bias=True)
        self.cheap_operation.weight.data = cheap_kernel
        self.cheap_operation.bias.data = cheap_bias

        self.cheap_operation = nn.Sequential(
            self.cheap_operation,
            self.cheap_activation if self.cheap_activation is not None else nn.Sequential()
        )

        # Delete un-used branches
        for para in self.parameters():
            para.detach_()
        if hasattr(self, 'primary_rpr_conv'):
            self.__delattr__('primary_rpr_conv')
        if hasattr(self, 'primary_rpr_scale'):
            self.__delattr__('primary_rpr_scale')
        if hasattr(self, 'primary_rpr_skip'):
            self.__delattr__('primary_rpr_skip')

        if hasattr(self, 'cheap_rpr_conv'):
            self.__delattr__('cheap_rpr_conv')
        if hasattr(self, 'cheap_rpr_scale'):
            self.__delattr__('cheap_rpr_scale')
        if hasattr(self, 'cheap_rpr_skip'):
            self.__delattr__('cheap_rpr_skip')

        self.infer_mode = True

    def _get_kernel_bias_primary(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Method to obtain re-parameterized kernel and bias.
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L83
        :return: Tuple of (kernel, bias) after fusing branches.
        """
        # get weights and bias of scale branch
        kernel_scale = 0
        bias_scale = 0
        if self.primary_rpr_scale is not None:
            kernel_scale, bias_scale = self._fuse_bn_tensor(self.primary_rpr_scale)
            # Pad scale branch kernel to match conv branch kernel size.
            pad = self.kernel_size // 2
            kernel_scale = torch.nn.functional.pad(kernel_scale,
                                                   [pad, pad, pad, pad])

        # get weights and bias of skip branch
        kernel_identity = 0
        bias_identity = 0
        if self.primary_rpr_skip is not None:
            kernel_identity, bias_identity = self._fuse_bn_tensor(self.primary_rpr_skip)

        # get weights and bias of conv branches
        kernel_conv = 0
        bias_conv = 0
        for ix in range(self.num_conv_branches):
            _kernel, _bias = self._fuse_bn_tensor(self.primary_rpr_conv[ix])
            kernel_conv += _kernel
            bias_conv += _bias

        kernel_final = kernel_conv + kernel_scale + kernel_identity
        bias_final = bias_conv + bias_scale + bias_identity
        return kernel_final, bias_final

    def _get_kernel_bias_cheap(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Method to obtain re-parameterized kernel and bias.
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L83
        :return: Tuple of (kernel, bias) after fusing branches.
        """
        # get weights and bias of scale branch
        kernel_scale = 0
        bias_scale = 0
        if self.cheap_rpr_scale is not None:
            kernel_scale, bias_scale = self._fuse_bn_tensor(self.cheap_rpr_scale)
            # Pad scale branch kernel to match conv branch kernel size.
            pad = self.kernel_size // 2
            kernel_scale = torch.nn.functional.pad(kernel_scale,
                                                   [pad, pad, pad, pad])

        # get weights and bias of skip branch
        kernel_identity = 0
        bias_identity = 0
        if self.cheap_rpr_skip is not None:
            kernel_identity, bias_identity = self._fuse_bn_tensor(self.cheap_rpr_skip)

        # get weights and bias of conv branches
        kernel_conv = 0
        bias_conv = 0
        for ix in range(self.num_conv_branches):
            _kernel, _bias = self._fuse_bn_tensor(self.cheap_rpr_conv[ix])
            kernel_conv += _kernel
            bias_conv += _bias

        kernel_final = kernel_conv + kernel_scale + kernel_identity
        bias_final = bias_conv + bias_scale + bias_identity
        return kernel_final, bias_final

    def _fuse_bn_tensor(self, branch) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Method to fuse batchnorm layer with preceeding conv layer.
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L95
        :param branch:
        :return: Tuple of (kernel, bias) after fusing batchnorm.
        """
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = torch.zeros((self.in_channels,
                                            input_dim,
                                            self.kernel_size,
                                            self.kernel_size),
                                           dtype=branch.weight.dtype,
                                           device=branch.weight.device)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim,
                                    self.kernel_size // 2,
                                    self.kernel_size // 2] = 1
                self.id_tensor = kernel_value
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def _conv_bn(self, in_channels, out_channels, kernel_size, stride, padding, groups=1, bias=False):
        """ Helper method to construct conv-batchnorm layers.
        :param kernel_size: Size of the convolution kernel.
        :param padding: Zero-padding size.
        :return: Conv-BN module.
        """
        mod_list = nn.Sequential()
        mod_list.add_module('conv', nn.Conv2d(in_channels=in_channels,
                                              out_channels=out_channels,
                                              kernel_size=kernel_size,
                                              stride=stride,
                                              padding=padding,
                                              groups=groups,
                                              bias=bias))
        mod_list.add_module('bn', nn.BatchNorm2d(out_channels))
        return mod_list


class LSKA(nn.Module):
    # Large-Separable-Kernel-Attention
    # https://github.com/StevenLauHKHK/Large-Separable-Kernel-Attention/tree/main
    def __init__(self, dim, k_size=7):
        super().__init__()

        self.k_size = k_size

        if k_size == 7:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 3), stride=(1,1), padding=(0,(3-1)//2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(3, 1), stride=(1,1), padding=((3-1)//2,0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 3), stride=(1,1), padding=(0,2), groups=dim, dilation=2)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(3, 1), stride=(1,1), padding=(2,0), groups=dim, dilation=2)
        elif k_size == 11:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 3), stride=(1,1), padding=(0,(3-1)//2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(3, 1), stride=(1,1), padding=((3-1)//2,0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1,1), padding=(0,4), groups=dim, dilation=2)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1,1), padding=(4,0), groups=dim, dilation=2)
        elif k_size == 23:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1,1), padding=(0,(5-1)//2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1,1), padding=((5-1)//2,0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 7), stride=(1,1), padding=(0,9), groups=dim, dilation=3)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(7, 1), stride=(1,1), padding=(9,0), groups=dim, dilation=3)
        elif k_size == 35:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1,1), padding=(0,(5-1)//2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1,1), padding=((5-1)//2,0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 11), stride=(1,1), padding=(0,15), groups=dim, dilation=3)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(11, 1), stride=(1,1), padding=(15,0), groups=dim, dilation=3)
        elif k_size == 41:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1,1), padding=(0,(5-1)//2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1,1), padding=((5-1)//2,0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 13), stride=(1,1), padding=(0,18), groups=dim, dilation=3)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(13, 1), stride=(1,1), padding=(18,0), groups=dim, dilation=3)
        elif k_size == 53:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1,1), padding=(0,(5-1)//2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1,1), padding=((5-1)//2,0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 17), stride=(1,1), padding=(0,24), groups=dim, dilation=3)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(17, 1), stride=(1,1), padding=(24,0), groups=dim, dilation=3)

        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0h(x)
        attn = self.conv0v(attn)
        attn = self.conv_spatial_h(attn)
        attn = self.conv_spatial_v(attn)
        attn = self.conv1(attn)
        return u * attn

# =========================================================
# SPPF-LSKA
# =========================================================
class SPPF_LSKA(nn.Module):
    """
    SPPF-LSKA: SPPF + LSKA
    """
    def __init__(self, c1, c2, k=5, lk=7):  # c1 输入通道，c2 输出通道
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.lska = LSKA(c_ * 4, lk)

    def forward(self, x):
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        y = torch.cat(y, 1)
        y = self.lska(y)
        y = self.cv2(y)
        return y


# -----------------------------
# Scale-aware Attention
# -----------------------------
class ScaleAwareAttn(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, feats):
        B, L = feats[0].shape[0], len(feats)
        pooled = torch.stack([f.mean(dim=(1, 2, 3)) for f in feats], dim=1)  # [B, L]
        weights = torch.softmax(pooled, dim=1)  # [B, L]
        outs = []
        for i, f in enumerate(feats):
            w = weights[:, i].view(B, 1, 1, 1)
            outs.append(f * w)
        return outs


# -----------------------------
# Spatial-aware Attention
# -----------------------------
class SpatialAwareAttn(nn.Module):
    def __init__(self, in_channels, deformable=False):
        super().__init__()
        self.deformable = deformable
        if self.deformable:
            # 如果你实现了 ModulatedDeformConvPack 可以替换这里
            raise NotImplementedError("Deformable conv not implemented in Lite version")
        else:
            self.sampler = DWConv(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.att_conv = nn.Conv2d(in_channels, 1, 1, bias=True)
        self.sig = nn.Sigmoid()

    def forward(self, feats):
        outs = []
        for f in feats:
            x = self.sampler(f)
            att = self.sig(self.att_conv(x))
            outs.append(f * att)
        return outs


# -----------------------------
# Task-aware Attention
# -----------------------------
class TaskAwareAttn(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(1, channels // reduction)
        self.fc1 = nn.Linear(channels, mid)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(mid, channels)
        self.sig = nn.Sigmoid()

    def forward(self, feats):
        outs = []
        for f in feats:
            B, C, H, W = f.shape
            y = f.mean(dim=(2, 3)).view(B, C)  # GAP
            y = self.act(self.fc1(y))
            y = self.sig(self.fc2(y)).view(B, C, 1, 1)
            outs.append(f * y)
        return outs


# -----------------------------
# DyHead Block Lite
# -----------------------------
class DyHeadBlockLite(nn.Module):
    def __init__(self, in_channels, out_channels=None, deformable=False, reduction=4):
        super().__init__()
        self.out_channels = out_channels or in_channels
        self.proj = nn.Conv2d(in_channels, self.out_channels, 1, bias=False) \
            if in_channels != self.out_channels else nn.Identity()

        self.scale_attn = ScaleAwareAttn()
        self.spatial_attn = SpatialAwareAttn(self.out_channels, deformable=deformable)
        self.task_attn = TaskAwareAttn(self.out_channels, reduction=reduction)

        self.conv = DWConv(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, feats):
        feats = [self.proj(f) for f in feats]
        feats = self.scale_attn(feats)
        feats = self.spatial_attn(feats)
        feats = self.task_attn(feats)
        return [self.conv(f) for f in feats]


# -----------------------------
# DyHead Lite (整体结构)
# -----------------------------
class DyHead(nn.Module):
    dynamic = False
    export = False
    format = None
    end2end = False
    max_det = 300
    shape = None
    anchors = torch.empty(0)
    strides = torch.empty(0)
    legacy = False

    def __init__(self, nc=80, c_out=128, num_blocks=2, ch=(), reg_max=16, deformable=False, reduction=4):
        super().__init__()
        assert isinstance(ch, (list, tuple)), f"DyHeadLite expects ch as list/tuple, got {type(ch)}"

        self.nc = int(nc)
        self.c_out = int(c_out)
        self.num_blocks = int(num_blocks)
        self.reg_max = int(reg_max)

        self.ch = list(ch)
        self.nl = len(self.ch)
        self.no = self.nc + self.reg_max * 4
        self.stride = torch.zeros(self.nl)

        self.proj = nn.ModuleList([nn.Conv2d(ci, self.c_out, 1, bias=False) for ci in self.ch])

        self.blocks = nn.ModuleList([
            DyHeadBlockLite(self.c_out, self.c_out, deformable=deformable, reduction=reduction)
            for _ in range(self.num_blocks)
        ])

        self.cls_conv = nn.ModuleList([DWConv(self.c_out, self.c_out) for _ in range(self.nl)])
        self.reg_conv = nn.ModuleList([DWConv(self.c_out, self.c_out) for _ in range(self.nl)])
        self.cls_pred = nn.ModuleList([nn.Conv2d(self.c_out, self.nc, 1) for _ in range(self.nl)])
        self.box_pred = nn.ModuleList([nn.Conv2d(self.c_out, 4 * self.reg_max, 1) for _ in range(self.nl)])

        try:
            from ultralytics.nn.modules import DFL
            self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()
        except Exception:
            self.dfl = nn.Identity()

    def forward(self, feats):
        if not isinstance(feats, (list, tuple)):
            raise TypeError("DyHeadLite forward expects list/tuple of feature maps")

        feats = [proj(f) for proj, f in zip(self.proj, feats)]
        for block in self.blocks:
            feats = block(feats)

        box_outs, cls_outs = [], []
        for i, f in enumerate(feats):
            cls_feat = self.cls_conv[i](f)
            box_feat = self.reg_conv[i](f)
            cls_outs.append(self.cls_pred[i](cls_feat))
            box_outs.append(self.box_pred[i](box_feat))

        raw_preds = [torch.cat((b, c), dim=1) for b, c in zip(box_outs, cls_outs)]

        if self.training:
            return raw_preds

        shape = raw_preds[0].shape
        x_cat = torch.cat([rp.view(shape[0], self.no, -1) for rp in raw_preds], dim=2)
        if self.format != "imx" and (self.dynamic or self.shape != shape):
            from ultralytics.utils.tal import make_anchors, dist2bbox
            self.anchors, self.strides = (x.transpose(0,1) for x in make_anchors(raw_preds, self.stride, 0.5))
            self.shape = shape

        box, cls = x_cat.split((self.reg_max * 4, self.nc), dim=1)
        from ultralytics.utils.tal import dist2bbox
        dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides
        y = torch.cat((dbox, cls.sigmoid()), 1)
        return (y, raw_preds) if not self.export else y

    def bias_init(self):
        for bp, cp, s in zip(self.box_pred, self.cls_pred, self.stride):
            if hasattr(bp, "bias") and bp.bias is not None:
                nn.init.constant_(bp.bias, 1.0)
            if hasattr(cp, "bias") and cp.bias is not None and s > 0:
                cp.bias.data[: self.nc] = math.log(torch.tensor(5 / self.nc / (640 / s) ** 2))

    def decode_bboxes(self, bboxes, anchors, xywh=True):
        from ultralytics.utils.tal import dist2bbox
        return dist2bbox(bboxes, anchors, xywh=xywh and (not self.end2end), dim=1)


# -----------------------------
# Depthwise + Pointwise Conv
# -----------------------------
class DPConv(nn.Module):
    """Depthwise (k×k) + Pointwise (1×1)"""
    def __init__(self, c1, c2, k=3, s=1, act=True):
        super().__init__()
        self.dw = nn.Conv2d(c1, c1, k, s, padding=k // 2, groups=c1, bias=False)
        self.bn1 = nn.BatchNorm2d(c1)
        self.act1 = nn.SiLU(inplace=True) if act else nn.Identity()
        self.pw = nn.Conv2d(c1, c2, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)
        self.act2 = nn.SiLU(inplace=True) if act else nn.Identity()
        # self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        x = self.act1(self.bn1(self.dw(x)))
        x = self.act2(self.bn2(self.pw(x)))
        return x


class GAM_Attention(nn.Module):
    def __init__(self, in_channels, rate=4):
        super(GAM_Attention, self).__init__()

        self.channel_attention = nn.Sequential(
            nn.Linear(in_channels, int(in_channels / rate)),
            nn.ReLU(inplace=True),
            nn.Linear(int(in_channels / rate), in_channels)
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, int(in_channels / rate), kernel_size=7, padding=3),
            nn.BatchNorm2d(int(in_channels / rate)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(in_channels / rate), in_channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)
        x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)
        x_channel_att = x_att_permute.permute(0, 3, 1, 2).sigmoid()

        x = x * x_channel_att

        x_spatial_att = self.spatial_attention(x).sigmoid()
        out = x * x_spatial_att

        return out

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, reduction=32):
        super(CoordAtt, self).__init__()
        oup = inp
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out


class Bottleneck_SCConv(nn.Module):

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = ScConv(c_)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f_SCConv(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(
            Bottleneck_SCConv(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        x = self.cv1(x)
        x = x.chunk(2, 1)
        y = list(x)
        # y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))



class RepConv1(nn.Module):
    """Improved RepConv supporting arbitrary kernel size and stride (k, s).
       Compatible with Ultralytics YAML.
    """
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True, deploy=False):
        super().__init__()
        assert isinstance(k, int), "kernel size must be int"

        # 自动 padding：支持奇偶核
        p = autopad(k, p)

        self.deploy = deploy
        self.groups = g
        self.in_channels = c1
        self.out_channels = c2

        if deploy:
            # 部署时只保留一个卷积
            self.rbr_reparam = nn.Conv2d(
                c1, c2, k, s, p, groups=g, bias=True
            )
        else:
            # 3x3 conv branch
            self.rbr_dense = nn.Conv2d(
                c1, c2, k, s, p, groups=g, bias=False
            )
            self.rbr_dense_bn = nn.BatchNorm2d(c2)

            # 1x1 conv branch
            self.rbr_1x1 = nn.Conv2d(
                c1, c2, 1, s, autopad(1), groups=g, bias=False
            )
            self.rbr_1x1_bn = nn.BatchNorm2d(c2)

            # identity branch（仅当输入输出通道相等且 stride=1）
            if c1 == c2 and s == 1:
                self.rbr_identity = nn.BatchNorm2d(c1)
            else:
                self.rbr_identity = None

        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        if self.deploy:
            return self.act(self.rbr_reparam(x))
        out = self.rbr_dense_bn(self.rbr_dense(x)) + \
              self.rbr_1x1_bn(self.rbr_1x1(x))
        if self.rbr_identity is not None:
            out += self.rbr_identity(x)
        return self.act(out)


class ECA(nn.Module):           # Efficient Channel Attention module
    def __init__(self, c, b=1, gamma=2):
        super(ECA, self).__init__()
        t = int(abs((math.log(c, 2) + b) / gamma))
        k = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv1d(1, 1, kernel_size=k, padding=int(k/2), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.avg_pool(x)
        out = self.conv1(out.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        out = self.sigmoid(out)
        return out * x


class CBM(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    # default_act = nn.Hardtanh()  # default activation
    default_act = nn.Mish()  # default activation
    # default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        # self.bn = nn.LazyBatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Apply convolution and activation without batch normalization."""
        return self.act(self.conv(x))


class DWCBM(Conv):

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  # ch_in, ch_out, kernel, stride, dilation, activation
        """Initialize Depth-wise convolution with given parameters."""
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


# GSConvE official version released
class GSConvE2(nn.Module):
    '''
    GSConv enhancement for representation learning: generate various receptive-fields and
    texture-features only in one Conv module
    https://github.com/AlanLi1997/rethinking-fpn
    '''
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        c_ = c2 // 4
        self.cv1 = Conv(c1, c_, k, s, p, g, d, act)
        self.cv2 = Conv(c_, c_, 9, 1, 4, c_, d, act)
        self.cv3 = Conv(c_, c_, 13, 1, 6, c_, d, act)
        self.cv4 = Conv(c_, c_, 17, 1, 8, c_, d, act)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x1)
        x3 = self.cv3(x1)
        x4 = self.cv4(x1)

        y = torch.cat((x1, x2, x3, x4), dim=1)
        # shuffle
        y = y.reshape(y.shape[0], 2, y.shape[1] // 2, y.shape[2], y.shape[3])
        y = y.permute(0, 2, 1, 3, 4)
        return y.reshape(y.shape[0], -1, y.shape[3], y.shape[4])


class GSConvE3(nn.Module):
    '''
    GSConv enhancement for representation learning: generate various receptive-fields and
    texture-features only in one Conv module
    https://github.com/AlanLi1997/slim-neck-by-gsconv
    '''
    def __init__(self, c1, c2, k=1, s=1, g=1, d=1, act=True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s, None, g, d, act)
        self.cv2 = nn.Sequential(
            nn.Conv2d(c_, c_, 9, 1, 4, groups=2, bias=True),
            nn.GELU()
        )

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x1)
        y = torch.cat((x1, x2), dim=1)
        # shuffle
        y = y.reshape(y.shape[0], 2, y.shape[1] // 2, y.shape[2], y.shape[3])
        y = y.permute(0, 2, 1, 3, 4)
        return y.reshape(y.shape[0], -1, y.shape[3], y.shape[4])


class GSConvE(nn.Module):
    '''
    GSConv enhancement for representation learning: generate various receptive-fields and
    texture-features only in one Conv module
    # GSConvE1 https://github.com/AlanLi1997/rethinking-fpn
    '''
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s, p, g, d, act)
        self.cv2 = nn.Sequential(
            nn.Conv2d(c_, c_, 3, 1, 1, bias=False),
            nn.Conv2d(c_, c_, 3, 1, 1, groups=c_, bias=False),
            nn.GELU()
        )

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x1)
        y = torch.cat((x1, x2), dim=1)
        # shuffle
        y = y.reshape(y.shape[0], 2, y.shape[1] // 2, y.shape[2], y.shape[3])
        y = y.permute(0, 2, 1, 3, 4)
        return y.reshape(y.shape[0], -1, y.shape[3], y.shape[4])


class GSConv(nn.Module):
    # GSConv https://github.com/AlanLi1997/slim-neck-by-gsconv
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s, p, g, d, act)
        self.cv2 = Conv(c_, c_, 5, 1, 2, c_, d, act)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = torch.cat((x1, self.cv2(x1)), 1)
        # shuffle
        y = x2.reshape(x2.shape[0], 2, x2.shape[1] // 2, x2.shape[2], x2.shape[3])
        y = y.permute(0, 2, 1, 3, 4)
        return y.reshape(y.shape[0], -1, y.shape[3], y.shape[4])


class GSConvns(GSConv):
    # GSConv with a normative-shuffle https://github.com/AlanLi1997/slim-neck-by-gsconv
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__(c1, c2, k, s, p, g, d, act)
        c_ = c2 // 2
        self.shuf = nn.Conv2d(c_ * 2, c2, 1, 1, 0, bias=False)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = torch.cat((x1, self.cv2(x1)), 1)
        # normative-shuffle, TRT supported
        return self.shuf(x2).relu()


class GSBottleneck(nn.Module):
    # GS Bottleneck https://github.com/AlanLi1997/slim-neck-by-gsconv
    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()
        c_ = c2 // 2
        # for lighting
        self.conv_lighting = nn.Sequential(
            GSConv(c1, c_, 3, 1, 1),
            GSConv(c_, c2, 3, 1, 1, act=True))
        self.shortcut = Conv(c1, c2, 1, 1, act=False)

    def forward(self, x):
        return self.conv_lighting(x) + self.shortcut(x)


class GSBottleneck1(nn.Module):
    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()
        c_ = c2 // 2
        self.conv_lighting = nn.Sequential(
            GSConv(c1, c_, k, s, 0),       # stride=s
            GSConv(c_, c2, k, s, k//2, act=True)
        )
        self.shortcut = Conv(c1, c2, 1, s, act=False)  # stride=s

    def forward(self, x):
        return self.conv_lighting(x) + self.shortcut(x)



class GSBottleneckC(GSBottleneck):
    # cheap GS Bottleneck https://github.com/AlanLi1997/slim-neck-by-gsconv
    def __init__(self, c1, c2, k=3, s=1):
        super().__init__(c1, c2, k, s)
        self.shortcut = DWCBM(c1, c2, 3, 1, act=False)



class VoVGSCSP(nn.Module):
    # VoVGSCSP module with GSBottleneck
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        # self.gc1 = GSConv(c_, c_, 1, 1)
        # self.gc2 = GSConv(c_, c_, 1, 1)
        self.gsb = GSBottleneck(c_, c_, 3, 1)
        self.res = Conv(c_, c_, 3, 1, act=False)
        self.cv3 = Conv(2*c_, c2, 1)  #

    def forward(self, x):

        x1 = self.gsb(self.cv1(x))
        y = self.cv2(x)
        return self.cv3(torch.cat((y, x1), dim=1))


class VoVGSCSPC(VoVGSCSP):
    # cheap VoVGSCSP module with GSBottleneck
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, e)
        c_ = int(c2 * e)  # hidden channels
        self.gsb = GSBottleneckC(c_, c_, 3, 1)


# 假设GSConv, GSBottleneck已定义

class VoVGSCSP1(nn.Module):
    # VoVGSCSP module with GSBottleneck
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        # self.gc1 = GSConv(c_, c_, 1, 1)
        # self.gc2 = GSConv(c_, c_, 1, 1)
        self.gsb = GSBottleneck(c_, c_, 3, 1)
        self.res = Conv(c_, c_, 3, 1, act=False)
        self.cv3 = Conv(2*c_, c2, 1)  #

    def forward(self, x):

        x1 = self.gsb(self.cv1(x))
        y = self.cv2(x)
        return self.cv3(torch.cat((y, x1), dim=1))

class VoVGSCSP2(nn.Module):
    # VoVGSCSP module with GSBottleneck
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        # self.gc1 = GSConv(c_, c_, 1, 1)
        # self.gc2 = GSConv(c_, c_, 1, 1)
        self.gsb = GSBottleneck(c_, c_, 3, 1)
        self.res = Conv(c_, c_, 3, 1, act=False)
        self.cv3 = Conv(3*c_, c2, 1)  #

    def forward(self, x):

        x1 = self.gsb(self.cv1(x))
        y1 = self.res(x1)
        y = self.cv2(x)
        return self.cv3(torch.cat((y, x1, y1), dim=1))

class VoVGSCSP3(nn.Module):
    # VoVGSCSP module with GSBottleneck
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        # self.gc1 = GSConv(c_, c_, 1, 1)
        # self.gc2 = GSConv(c_, c_, 1, 1)
        self.gsb = GSBottleneck(c_, c_, 3, 1)
        self.res = Conv(c_, 2*c_, 3, 1, act=False)
        self.cv3 = Conv(2*c_, c2, 1)  #

    def forward(self, x):

        x1 = self.gsb(self.cv1(x))
        y1 = self.res(self.cv1(x))
        y = self.cv2(x)
        return self.cv3(torch.cat((y, x1), dim=1)) + y1

class Infos(nn.Module):
    """Concatenate a list of tensors along dimension."""

    def __init__(self, c1, c2):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        # self.d = 1
        self.liner = nn.Conv2d(c2, c2, 1, 1, bias=False)
        self.merge = nn.Conv2d(c1, c2, 1, 1, bias=False)

    def forward(self, x):
        x1 = self.merge(torch.cat(x, 1))
        alpha = self.liner(x1).sigmoid()

        return alpha * x1


class SNIr(nn.Module):
    # local feature aligned refine
    def __init__(self, c1=0, c2=0):
        super().__init__()
        self.us = nn.Upsample(None, 2, 'nearest')
        # self.alpha = 1/(up_f**2)
        # self.smooth = nn.AdaptiveAvgPool2d(1)
        self.liner = nn.Conv2d(c1, c2, 1, 1, 0, bias=True)

    def forward(self, x):
        x1 = 0.25 * self.us(x)
        # a = self.smooth(x1)
        # alpha = self.liner(0.25 * x1)
        return x1 * self.liner(x1).sigmoid()


class SNI(nn.Module):
    # local feature aligned refine
    def __init__(self, c1=0, c2=0):
        super().__init__()
        self.us = nn.Upsample(None, 2, 'nearest')
        # self.alpha = 0.25
        # self.smooth = nn.AdaptiveAvgPool2d(1)
        # self.liner = nn.Conv2d(c1, c2, 1, 1, 0, bias=False)

    def forward(self, x):
        # x1 = self.us(x)
        # alpha = self.liner(0.25 * x1).sigmoid()
        return self.us(0.25*x)


class SNIl(nn.Module):

    def __init__(self, c1=0, c2=0):
        super().__init__()
        self.us = nn.Upsample(None, 2, 'nearest')

        self.alpha = nn.Parameter(torch.tensor(0.25), requires_grad=False)

        # self.register_buffer('alpha_min', torch.tensor(0.1))
        # self.register_buffer('alpha_max', torch.tensor(1.0))

    def forward(self, x):

        self.alpha.data = torch.clamp(self.alpha.data, 0.1, 1.0)

        # print(self.alpha)
        return self.alpha * self.us(x)


class LAPM1(nn.Module):
    def __init__(self, c1=3, c2=1, th=0.02):
        super().__init__()
        self.th = th
        self.ds = nn.MaxPool2d(2, 2)
        # self.pfactor = 10
        # self.pfactor_param = nn.Parameter(torch.tensor(6.906), requires_grad=True) # 6.906=10
        self.cv = Conv(1, c2, 1, 1)

    def forward(self, x):
        # x = (0.1 + torch.sigmoid(self.pfactor_param)) * 10 * self.ds(x)
        x = 10 * self.ds(x)
        # gray_tensor = (0.299 * x[:, 0, :, :] + 0.587 * x[:, 1, :, :] + 0.114 * x[:, 2, :, :])
        # binary_tensor = torch.where(gray_tensor > self.th, x.new_ones(gray_tensor.size()),
        #                             x.new_zeros(gray_tensor.size())).unsqueeze(0)
        weights = torch.tensor([0.299, 0.587, 0.114], dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
        gray_tensor = torch.sum(x * weights, dim=1, keepdim=True)
        binary_tensor = torch.where(gray_tensor > self.th, x.new_ones(gray_tensor.size()),
                                    x.new_zeros(gray_tensor.size()))
        return self.cv(binary_tensor)


class LAPM2(nn.Module):
    def __init__(self, c1=1, c2=1, th=0.02):
        super().__init__()
        self.th = th
        self.ds = nn.MaxPool2d(2, 2)
        # self.cv = nn.Conv2d(2*c1, c2, 1, 1, bias=False)
        self.cv = Conv(2 * c1, c2, 1, 1)

    def forward(self, x):
        gray_tensor = self.ds(x)
        binary_tensor = torch.where(gray_tensor > self.th,
                                    x.new_ones(gray_tensor.size()),
                                    x.new_zeros(gray_tensor.size()))
        # binary_tensor = torch.where(gray_tensor > self.th, 1.0, 0.0)
        return self.cv(torch.cat((binary_tensor, gray_tensor), 1))


class SLConv(nn.Module):
    '''
    Separable learning strategy for deeplearning model
    '''
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s, p, g, d, act)
        self.cv2 = Conv(c_, c_, 3, 1, 1, g, d, act)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x1)
        # y = torch.cat((x1, x2), dim=1)
        # # shuffle
        # y = y.reshape(y.shape[0], 2, y.shape[1] // 2, y.shape[2], y.shape[3])
        # y = y.permute(0, 2, 1, 3, 4)
        # return y.reshape(y.shape[0], -1, y.shape[3], y.shape[4])
        return torch.cat((x1, x2), dim=1)


class ESD(nn.Module):
    '''
    https://github.com/AlanLi1997/rethinking-fpn
    Extended spatial window for down-sampling
    lightweight fusion
    '''
    def __init__(self, c1, c2, k=3, s=2, g=1, d=1, act=True):
        super().__init__()
        self.out_c = c2
        self.dense_feature = Conv(c1, c2, k, s, k // 2, g, d, act)  # window_dense_f
        self.global_feature = nn.AvgPool2d(4, 2, 1)  # window_global_f
        self.local_feature = nn.MaxPool2d(4, 2, 1)  # window_local_f

    def forward(self, x):
        if self.out_c == x.shape[1]:
            return self.global_feature(x) + self.local_feature(x) + self.dense_feature(x)
        else:
            return torch.cat((self.global_feature(x), self.local_feature(x)), dim=1) + self.dense_feature(x)


class ESD2(nn.Module):
    '''
    https://github.com/AlanLi1997/rethinking-fpn
    Extended spatial window for down-sampling
    learnable linearly fusion
    '''
    def __init__(self, c1, c2, k=3, s=2, g=1, d=1, act=True):
        super().__init__()
        self.dense_feature = Conv(c1, c2, k, s, None, g, d, act)  # window_dense_f
        self.global_feature = nn.AvgPool2d(4, 2, 1)  # window_global_f
        self.local_feature = nn.MaxPool2d(4, 2, 1)  # window_local_f
        self.fuse = nn.Conv2d(2*c2, c2, 1, 1, bias=False)

    def forward(self, x):
        return self.fuse(torch.cat((self.global_feature(x), self.local_feature(x), self.dense_feature(x)), dim=1))


class ConvNeXtBlock(nn.Module):
    """
    Lightweight ConvNeXt-style block for YOLO backbone
    Uses depthwise conv + GELU + pointwise linear layers
    """
    def __init__(self, c1, k=7, mlp_ratio=4):
        super().__init__()
        hidden_dim = int(c1 * mlp_ratio)
        self.dwconv = nn.Conv2d(c1, c1, k, 1, k // 2, groups=c1)
        self.norm = nn.LayerNorm(c1, eps=1e-6)
        self.pwconv1 = nn.Linear(c1, hidden_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(hidden_dim, c1)
        self.gamma = nn.Parameter(torch.ones((c1)), requires_grad=True)

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        return shortcut + x


class UIBBlock(nn.Module):
    """
    Unified Inverted Bottleneck Block for MobileNetV4-Conv-S.
    Supports SE-Lite and MobileMQA attention.
    """
    def __init__(self, c1, c2, expansion=4, out_channels=None, stride=1, attn=None):
        super().__init__()
        out_channels = out_channels or c2
        mid_channels = int(c1 * expansion)

        # ✅ expand conv, 输入通道来自上层输出
        self.expand = Conv(c1, mid_channels, k=1, s=1, act=True)

        # ✅ depthwise conv
        self.dwconv = Conv(mid_channels, mid_channels, k=3, s=stride, g=mid_channels, act=True)

        # ✅ attention
        if attn == "se":
            self.attn = SELite(mid_channels)
        elif attn == "mqa":
            self.attn = MobileMQA(mid_channels)
        else:
            self.attn = nn.Identity()

        # ✅ projection conv
        self.project = Conv(mid_channels, out_channels, k=1, s=1, act=False)

        # ✅ shortcut
        self.add = c1 == out_channels and stride == 1

    def forward(self, x):
        y = self.project(self.attn(self.dwconv(self.expand(x))))
        if self.add:
            y = y + x
        return y



class SELite(nn.Module):
    """Squeeze-and-Excitation Lite (轻量SE)"""
    def __init__(self, c, reduction=4):
        super().__init__()
        r = max(c // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(c, r, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(r, c, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(self.pool(x))


class MobileMQA(nn.Module):
    """
    Official Mobile Multi-Query Attention (MQA)
    Based on: MobileNetV4 (Google Research 2024)
    """
    def __init__(self, dim, num_heads=4, qkv_bias=True, expansion=2):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.q = nn.Conv2d(dim, dim, 1, bias=qkv_bias)
        self.kv = nn.Conv2d(dim, dim * 2, 1, bias=qkv_bias)
        self.proj = nn.Conv2d(dim, dim, 1, bias=False)
        self.dwconv_q = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False)
        self.dwconv_kv = nn.Conv2d(dim * 2, dim * 2, 3, 1, 1, groups=dim * 2, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.dwconv_q(self.q(x)).reshape(B, self.num_heads, C // self.num_heads, H * W)
        kv = self.dwconv_kv(self.kv(x)).reshape(B, 2, self.num_heads, C // self.num_heads, H * W)
        k, v = kv[:, 0], kv[:, 1]
        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        out = (v @ attn.transpose(-2, -1)).reshape(B, C, H, W)
        return self.proj(out)


class Upsample1(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(Upsample1, self).__init__()

        self.upsample = nn.Sequential(
            Conv(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=scale_factor, mode='bilinear')
        )

        # carafe
        # from mmcv.ops import CARAFEPack
        # self.upsample = nn.Sequential(
        #     BasicConv(in_channels, out_channels, 1),
        #     CARAFEPack(out_channels, scale_factor=scale_factor)
        # )

    def forward(self, x):
        x = self.upsample(x)

        return x


class Downsample1(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(Downsample1, self).__init__()

        self.downsample = nn.Sequential(
            Conv(in_channels, out_channels, scale_factor, scale_factor, 0)
        )

    def forward(self, x):
        x = self.downsample(x)

        return x


class ASFF_2(nn.Module):
    def __init__(self, inter_dim=512, level=0, channel=[64, 128]):
        super(ASFF_2, self).__init__()

        self.inter_dim = inter_dim
        compress_c = 8

        self.weight_level_1 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = Conv(self.inter_dim, compress_c, 1, 1)

        self.weight_levels = nn.Conv2d(compress_c * 2, 2, kernel_size=1, stride=1, padding=0)

        self.conv = DWConv(self.inter_dim, self.inter_dim, 3, 1)
        self.upsample = Upsample1(channel[1], channel[0])
        self.downsample = Downsample1(channel[0], channel[1])
        self.level = level

    def forward(self, x):
        input1, input2 = x
        if self.level == 0:
            input2 = self.upsample(input2)
        elif self.level == 1:
            input1 = self.downsample(input1)

        level_1_weight_v = self.weight_level_1(input1)
        level_2_weight_v = self.weight_level_2(input2)

        levels_weight_v = torch.cat((level_1_weight_v, level_2_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = input1 * levels_weight[:, 0:1, :, :] + \
                            input2 * levels_weight[:, 1:2, :, :]

        out = self.conv(fused_out_reduced)

        return out


class ASFF_3(nn.Module):
    def __init__(self, inter_dim=512, level=0, channel=[64, 128, 256]):
        super(ASFF_3, self).__init__()

        self.inter_dim = inter_dim
        compress_c = 8

        self.weight_level_1 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_3 = Conv(self.inter_dim, compress_c, 1, 1)

        self.weight_levels = nn.Conv2d(compress_c * 3, 3, kernel_size=1, stride=1, padding=0)

        self.conv = DWConv(self.inter_dim, self.inter_dim, 3, 1)

        self.level = level
        if self.level == 0:
            self.upsample4x = Upsample1(channel[2], channel[0], scale_factor=4)
            self.upsample2x = Upsample1(channel[1], channel[0], scale_factor=2)
        elif self.level == 1:
            self.upsample2x1 = Upsample1(channel[2], channel[1], scale_factor=2)
            self.downsample2x1 = Downsample1(channel[0], channel[1], scale_factor=2)
        elif self.level == 2:
            self.downsample2x = Downsample1(channel[1], channel[2], scale_factor=2)
            self.downsample4x = Downsample1(channel[0], channel[2], scale_factor=4)

    def forward(self, x):
        input1, input2, input3 = x
        if self.level == 0:
            input2 = self.upsample2x(input2)
            input3 = self.upsample4x(input3)
        elif self.level == 1:
            input3 = self.upsample2x1(input3)
            input1 = self.downsample2x1(input1)
        elif self.level == 2:
            input1 = self.downsample4x(input1)
            input2 = self.downsample2x(input2)
        level_1_weight_v = self.weight_level_1(input1)
        level_2_weight_v = self.weight_level_2(input2)
        level_3_weight_v = self.weight_level_3(input3)

        levels_weight_v = torch.cat((level_1_weight_v, level_2_weight_v, level_3_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = input1 * levels_weight[:, 0:1, :, :] + \
                            input2 * levels_weight[:, 1:2, :, :] + \
                            input3 * levels_weight[:, 2:, :, :]

        out = self.conv(fused_out_reduced)

        return out


class Upsample(nn.Module):
    """Applies convolution followed by upsampling."""

    def __init__(self, c1, c2, scale_factor=2):
        super().__init__()
        if scale_factor == 2:
            self.cv1 = nn.ConvTranspose2d(c1, c2, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        elif scale_factor == 4:
            self.cv1 = nn.ConvTranspose2d(c1, c2, 4, 4, 0, bias=True)  # nn.Upsample(scale_factor=4, mode='nearest')

    def forward(self, x):
        # return self.upsample(self.cv1(x))
        return self.cv1(x)


class ASFF2(nn.Module):
    """ASFF2 module for YOLO AFPN head https://arxiv.org/abs/2306.15988"""

    def __init__(self, c1, c2, level=0):
        super().__init__()
        c1_l, c1_h = c1[0], c1[1]
        self.level = level
        self.dim = c1_l, c1_h
        self.inter_dim = self.dim[self.level]
        compress_c = 8

        if level == 0:
            self.stride_level_1 = Upsample(c1_h, self.inter_dim)
        if level == 1:
            self.stride_level_0 = Conv(c1_l, self.inter_dim, 2, 2, 0)  # downsample 2x

        self.weight_level_0 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = Conv(self.inter_dim, compress_c, 1, 1)

        self.weights_levels = nn.Conv2d(compress_c * 2, 2, kernel_size=1, stride=1, padding=0)
        self.conv = Conv(self.inter_dim, self.inter_dim, 3, 1)

    def forward(self, x):
        x_level_0, x_level_1 = x[0], x[1]

        if self.level == 0:
            level_0_resized = x_level_0
            level_1_resized = self.stride_level_1(x_level_1)
        elif self.level == 1:
            level_0_resized = self.stride_level_0(x_level_0)
            level_1_resized = x_level_1

        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v), 1)
        levels_weight = self.weights_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = level_0_resized * levels_weight[:, 0:1] + level_1_resized * levels_weight[:, 1:2]
        return self.conv(fused_out_reduced)


class ASFF3(nn.Module):
    """ASFF3 module for YOLO AFPN head https://arxiv.org/abs/2306.15988"""

    def __init__(self, c1, c2, level=0):
        super().__init__()
        c1_l, c1_m, c1_h = c1[0], c1[1], c1[2]
        self.level = level
        self.dim = c1_l, c1_m, c1_h
        self.inter_dim = self.dim[self.level]
        compress_c = 8

        if level == 0:
            self.stride_level_1 = Upsample(c1_m, self.inter_dim)
            self.stride_level_2 = Upsample(c1_h, self.inter_dim, scale_factor=4)

        if level == 1:
            self.stride_level_0 = Conv(c1_l, self.inter_dim, 2, 2, 0)  # downsample 2x
            self.stride_level_2 = Upsample(c1_h, self.inter_dim)

        if level == 2:
            self.stride_level_0 = Conv(c1_l, self.inter_dim, 4, 4, 0)  # downsample 4x
            self.stride_level_1 = Conv(c1_m, self.inter_dim, 2, 2, 0)  # downsample 2x

        self.weight_level_0 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = Conv(self.inter_dim, compress_c, 1, 1)

        self.weights_levels = nn.Conv2d(compress_c * 3, 3, kernel_size=1, stride=1, padding=0)
        self.conv = Conv(self.inter_dim, self.inter_dim, 3, 1)

    def forward(self, x):
        x_level_0, x_level_1, x_level_2 = x[0], x[1], x[2]

        if self.level == 0:
            level_0_resized = x_level_0
            level_1_resized = self.stride_level_1(x_level_1)
            level_2_resized = self.stride_level_2(x_level_2)

        elif self.level == 1:
            level_0_resized = self.stride_level_0(x_level_0)
            level_1_resized = x_level_1
            level_2_resized = self.stride_level_2(x_level_2)

        elif self.level == 2:
            level_0_resized = self.stride_level_0(x_level_0)
            level_1_resized = self.stride_level_1(x_level_1)
            level_2_resized = x_level_2

        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)

        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v), 1)
        w = self.weights_levels(levels_weight_v)
        w = F.softmax(w, dim=1)

        fused_out_reduced = level_0_resized * w[:, :1] + level_1_resized * w[:, 1:2] + level_2_resized * w[:, 2:]
        return self.conv(fused_out_reduced)


class DWConvPW1(nn.Module):
    """Depthwise(3x3) + Pointwise(1x1)"""
    def __init__(self, in_ch, out_ch, k=3, s=2, p=1, bias=False):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size=k, stride=s,
                            padding=p, groups=in_ch, bias=bias)
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=bias)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)
        # self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        return self.act(x)

class CBR(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    # default_act = nn.Hardtanh()  # default activation
    # default_act = nn.Mish()  # default activation
    default_act = nn.ReLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, bias=False):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=bias)
        self.bn = nn.BatchNorm2d(c2)
        # self.bn = nn.LazyBatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Apply convolution and activation without batch normalization."""
        return self.act(self.conv(x))

class DWBottleneck(Bottleneck):
    """Rep bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a RepBottleneck module with customizable in/out channels, shortcuts, groups and expansion."""
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = DWConv(c1, c_, k[0], 1)


class DWC3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(DWBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class DWC3k2(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            DWC3k(self.c, self.c, 2, shortcut, g) if c3k else DWBottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )


class DWA2C2f(A2C2f):

    def __init__(self, c1, c2, n=1, a2=True, area=1, residual=False, mlp_ratio=2.0, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, a2, area, residual, mlp_ratio, e, g, shortcut)
        c_ = int(c2 * e)  # hidden channels
        assert c_ % 32 == 0, "Dimension of ABlock be a multiple of 32."

        # num_heads = c_ // 64 if c_ // 64 >= 2 else c_ // 32
        num_heads = c_ // 32

        self.cv1 = DWConvPW1(c1, c_, 1, 1)
        self.cv2 = DWConvPW1((1 + n) * c_, c2, 1)  # optional act=FReLU(c2)

        init_values = 0.01  # or smaller
        self.gamma = nn.Parameter(init_values * torch.ones((c2)), requires_grad=True) if a2 and residual else None

        self.m = nn.ModuleList(
            nn.Sequential(*(ABlock(c_, num_heads, mlp_ratio, area) for _ in range(2))) if a2 else DWC3k(c_, c_, 2,
                                                                                                         shortcut, g)
            for _ in range(n)
        )


class EMA(nn.Module):  # 定义一个继承自 nn.Module 的 EMA 类
    def __init__(self, channels, c2=None, factor=32):  # 构造函数，初始化对象
        super(EMA, self).__init__()  # 调用父类的构造函数
        self.groups = factor  # 定义组的数量为 factor，默认值为 32
        assert channels // self.groups > 0  # 确保通道数可以被组数整除
        self.softmax = nn.Softmax(-1)  # 定义 softmax 层，用于最后一个维度
        self.agp = nn.AdaptiveAvgPool2d((1, 1))  # 定义自适应平均池化层，输出大小为 1x1
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # 定义自适应平均池化层，只在宽度上池化
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # 定义自适应平均池化层，只在高度上池化
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)  # 定义组归一化层
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1,
                                 padding=0)  # 定义 1x1 卷积层
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1,
                                 padding=1)  # 定义 3x3 卷积层

    def forward(self, x):  # 定义前向传播函数
        b, c, h, w = x.size()  # 获取输入张量的大小：批次、通道、高度和宽度
        group_x = x.reshape(b * self.groups, -1, h, w)  # 将输入张量重新形状为 (b * 组数, c // 组数, 高度, 宽度)
        x_h = self.pool_h(group_x)  # 在高度上进行池化
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)  # 在宽度上进行池化并交换维度
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))  # 将池化结果拼接并通过 1x1 卷积层
        x_h, x_w = torch.split(hw, [h, w], dim=2)  # 将卷积结果按高度和宽度分割
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())  # 进行组归一化，并结合高度和宽度的激活结果
        x2 = self.conv3x3(group_x)  # 通过 3x3 卷积层
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))  # 对 x1 进行池化、形状变换、并应用 softmax
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # 将 x2 重新形状为 (b * 组数, c // 组数, 高度 * 宽度)
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))  # 对 x2 进行池化、形状变换、并应用 softmax
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # 将 x1 重新形状为 (b * 组数, c // 组数, 高度 * 宽度)
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)  # 计算权重
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)  # 应用权重并将形状恢复为原始大小


class EFC(BaseModule):
    def __init__(self, c1, c2):
        super().__init__()
        self.conv1 = nn.Conv2d(c1, c2, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(c2, c2, kernel_size=1, stride=1)
        self.conv4 = nn.Conv2d(c2, c2, kernel_size=1, stride=1)
        self.bn = nn.BatchNorm2d(c2)
        self.sigomid = nn.Sigmoid()
        self.group_num = 16
        self.eps = 1e-10
        self.gamma = nn.Parameter(torch.randn(c2, 1, 1))
        self.beta = nn.Parameter(torch.zeros(c2, 1, 1))
        self.gate_genator = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(c2, c2, 1, 1),
            nn.ReLU(True),
            nn.Softmax(dim=1),
        )
        self.dwconv = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1, groups=c2)
        self.conv3 = nn.Conv2d(c2, c2, kernel_size=1, stride=1)
        self.Apt = nn.AdaptiveAvgPool2d(1)
        self.one = c2
        self.two = c2
        self.conv4_gobal = nn.Conv2d(c2, 1, kernel_size=1, stride=1)
        for group_id in range(0, 4):
            self.interact = nn.Conv2d(c2 // 4, c2 // 4, 1, 1, )

    def forward(self, x):
        x1, x2 = x
        global_conv1 = self.conv1(x1)
        bn_x = self.bn(global_conv1)
        weight_1 = self.sigomid(bn_x)
        global_conv2 = self.conv2(x2)
        bn_x2 = self.bn(global_conv2)
        weight_2 = self.sigomid(bn_x2)
        X_GOBAL = global_conv1 + global_conv2
        x_conv4 = self.conv4_gobal(X_GOBAL)
        X_4_sigmoid = self.sigomid(x_conv4)
        X_ = X_4_sigmoid * X_GOBAL
        X_ = X_.chunk(4, dim=1)
        out = []
        for group_id in range(0, 4):
            out_1 = self.interact(X_[group_id])
            N, C, H, W = out_1.size()
            x_1_map = out_1.reshape(N, 1, -1)
            mean_1 = x_1_map.mean(dim=2, keepdim=True)
            x_1_av = x_1_map / mean_1
            x_2_2 = F.softmax(x_1_av, dim=-1)
            x1 = x_2_2.reshape(N, C, H, W)
            x1 = X_[group_id] * x1
            out.append(x1)
        out = torch.cat([out[0], out[1], out[2], out[3]], dim=1)
        N, C, H, W = out.size()
        x_add_1 = out.reshape(N, self.group_num, -1)
        N, C, H, W = X_GOBAL.size()
        x_shape_1 = X_GOBAL.reshape(N, self.group_num, -1)
        mean_1 = x_shape_1.mean(dim=2, keepdim=True)
        std_1 = x_shape_1.std(dim=2, keepdim=True)
        x_guiyi = (x_add_1 - mean_1) / (std_1 + self.eps)
        x_guiyi_1 = x_guiyi.reshape(N, C, H, W)
        x_gui = (x_guiyi_1 * self.gamma + self.beta)
        weight_x3 = self.Apt(X_GOBAL)
        reweights = self.sigomid(weight_x3)
        x_up_1 = reweights >= weight_1
        x_low_1 = reweights < weight_1
        x_up_2 = reweights >= weight_2
        x_low_2 = reweights < weight_2
        x_up = x_up_1 * X_GOBAL + x_up_2 * X_GOBAL
        x_low = x_low_1 * X_GOBAL + x_low_2 * X_GOBAL
        x11_up_dwc = self.dwconv(x_low)
        x11_up_dwc = self.conv3(x11_up_dwc)
        x_so = self.gate_genator(x_low)
        x11_up_dwc = x11_up_dwc * x_so
        x22_low_pw = self.conv4(x_up)
        xL = x11_up_dwc + x22_low_pw
        xL = xL + x_gui

        return xL


class Maxpoll(BaseModule):  # 串联
    def __init__(self, dim, dim_out):
        super().__init__()
        self.conv1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=3 // 2)
        self.conv3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=3 // 2)
        self.conv2 = nn.Conv2d(2 * dim, dim_out, 1, 1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv3(x)
        x3 = torch.cat([x1, x2], dim=1)
        x4 = self.conv2(x3)
        return x4


# https://arxiv.org/pdf/1807.03247
# ------coordconv-----------------------------------------------------
class AddCoords(nn.Module):
    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """
        batch_size, _, x_dim, y_dim = input_tensor.size()

        xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)

        xx_channel = xx_channel.float() / (x_dim - 1)
        yy_channel = yy_channel.float() / (y_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

        ret = torch.cat([
            input_tensor,
            xx_channel.type_as(input_tensor),
            yy_channel.type_as(input_tensor)], dim=1)

        if self.with_r:
            rr = torch.sqrt(
                torch.pow(xx_channel.type_as(input_tensor) - 0.5, 2) + torch.pow(yy_channel.type_as(input_tensor) - 0.5,
                                                                                 2))
            ret = torch.cat([ret, rr], dim=1)

        return ret


class CoordConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, with_r=False):
        super().__init__()
        self.addcoords = AddCoords(with_r=with_r)
        in_channels += 2
        if with_r:
            in_channels += 1
        self.conv = Conv(in_channels, out_channels, k=kernel_size, s=stride)

    def forward(self, x):
        x = self.addcoords(x)
        x = self.conv(x)
        return x

# -----------------------------------------------------------

class GSC3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GSBottleneck1(c_, c_) for _ in range(n)))


class GSC3k2(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            GSC3k(self.c, self.c, 2, shortcut, g) if c3k else GSBottleneck1(self.c, self.c) for _ in range(n)
        )

class GSA2C2f(A2C2f):

    def __init__(self, c1, c2, n=1, a2=True, area=1, residual=False, mlp_ratio=2.0, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, a2, area, residual, mlp_ratio, e, g, shortcut)
        c_ = int(c2 * e)  # hidden channels
        assert c_ % 32 == 0, "Dimension of ABlock be a multiple of 32."

        # num_heads = c_ // 64 if c_ // 64 >= 2 else c_ // 32
        num_heads = c_ // 32

        self.cv1 = GSConv(c1, c_, 1, 1)
        self.cv2 = GSConv((1 + n) * c_, c2, 1)  # optional act=FReLU(c2)

        init_values = 0.01  # or smaller
        self.gamma = nn.Parameter(init_values * torch.ones((c2)), requires_grad=True) if a2 and residual else None

        self.m = nn.ModuleList(
            nn.Sequential(*(ABlock(c_, num_heads, mlp_ratio, area) for _ in range(2))) if a2 else GSC3k(c_, c_, 2,
                                                                                                         shortcut, g)
            for _ in range(n)
        )

class CABottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.ca = CoordAtt(c2)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.ca(self.cv2(self.cv1(x))) if self.add else self.ca(self.cv2(self.cv1(x)))

class CAC3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(CABottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

class CAC3k2(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            CAC3k(self.c, self.c, 2, shortcut, g) if c3k else CABottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )

class CAA2C2f(A2C2f):

    def __init__(self, c1, c2, n=1, a2=True, area=1, residual=False, mlp_ratio=2.0, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, a2, area, residual, mlp_ratio, e, g, shortcut)
        c_ = int(c2 * e)  # hidden channels
        assert c_ % 32 == 0, "Dimension of ABlock be a multiple of 32."

        # num_heads = c_ // 64 if c_ // 64 >= 2 else c_ // 32
        num_heads = c_ // 32

        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv((1 + n) * c_, c2, 1)  # optional act=FReLU(c2)

        init_values = 0.01  # or smaller
        self.gamma = nn.Parameter(init_values * torch.ones((c2)), requires_grad=True) if a2 and residual else None

        self.m = nn.ModuleList(
            nn.Sequential(*(ABlock(c_, num_heads, mlp_ratio, area) for _ in range(2))) if a2 else CAC3k(c_, c_,
                                                                                                        2, shortcut, g)
            for _ in range(n)
        )


class DPBottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = DPConv(c1, c_, k[0], 1)
        self.cv2 = DPConv(c_, c2, k[1], 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class DPC3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(DPBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))


class DPC3k2(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            DPC3k(self.c, self.c, 2, shortcut, g) if c3k else DPBottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )


class DPA2C2f(A2C2f):

    def __init__(self, c1, c2, n=1, a2=True, area=1, residual=False, mlp_ratio=2.0, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, a2, area, residual, mlp_ratio, e, g, shortcut)
        c_ = int(c2 * e)  # hidden channels
        assert c_ % 32 == 0, "Dimension of ABlock be a multiple of 32."

        # num_heads = c_ // 64 if c_ // 64 >= 2 else c_ // 32
        num_heads = c_ // 32

        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv((1 + n) * c_, c2, 1)  # optional act=FReLU(c2)

        init_values = 0.01  # or smaller
        self.gamma = nn.Parameter(init_values * torch.ones((c2)), requires_grad=True) if a2 and residual else None

        self.m = nn.ModuleList(
            nn.Sequential(*(ABlock(c_, num_heads, mlp_ratio, area) for _ in range(2))) if a2 else DPC3k(c_, c_,
                                                                                                        2, shortcut, g)
            for _ in range(n)
        )


from pytorch_wavelets import DWTForward

class Down_wt(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down_wt, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_ch * 4, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        yL, yH = self.wt(x)
        y_HL = yH[0][:, :, 0, ::]
        y_LH = yH[0][:, :, 1, ::]
        y_HH = yH[0][:, :, 2, ::]
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        x = self.conv_bn_relu(x)
        return x

class DPR(nn.Module):
    """Depthwise(3x3) + Pointwise(1x1)"""
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, bias=False):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size=k, stride=s,
                            padding=p, groups=in_ch, bias=bias)
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=bias)
        self.bn = nn.BatchNorm2d(out_ch)
        # self.act = nn.SiLU()
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        return self.act(x)


class ASFF(nn.Module):
    """
    Lightweight ASFF (Adaptive Spatial Feature Fusion with DPR).
    Args:
        level (int): target level (0,1,2...) defines which scale is the output reference.
        in_channels (list[int]): channels of each input level, e.g. [256,512,1024]
        out_channels (int): output channels (default = in_channels[0])
    """

    def __init__(self, level: int, in_channels: list, out_channels: int = None):
        super().__init__()
        assert isinstance(in_channels, (list, tuple)), "in_channels must be list"
        self.level = int(level)
        self.in_channels = list(in_channels)
        self.num_levels = len(self.in_channels)
        self.out_channels = out_channels or self.in_channels[0]

        # per-level reduction conv (lightweight)
        self.reduce_convs = nn.ModuleList([
            DPR(c, self.out_channels) for c in self.in_channels
        ])

        # weight generation conv (lightweight)
        self.weight_conv = nn.Sequential(
            DPR(self.out_channels * self.num_levels, self.num_levels, k=1, p=0, bias=True)
        )

        # final output conv (lightweight)
        self.out_conv = DPR(self.out_channels, self.out_channels, k=3, p=1)

    def forward(self, feats):
        """
        feats: list of tensors, length == num_levels
        returns: fused feature [B, out_channels, H_ref, W_ref]
        """
        assert isinstance(feats, (list, tuple)), "ASFF expects list of multi-scale features"
        assert len(feats) == self.num_levels, f"ASFF expects {self.num_levels} levels, got {len(feats)}"

        ref = feats[self.level]
        ref_h, ref_w = ref.shape[-2], ref.shape[-1]
        reduced = []
        for i, f in enumerate(feats):
            x = self.reduce_convs[i](f)
            if x.shape[-2:] != (ref_h, ref_w):
                x = F.interpolate(x, size=(ref_h, ref_w), mode="bilinear", align_corners=False)
            reduced.append(x)

        concat = torch.cat(reduced, dim=1)  # [B, out_ch * L, H, W]
        w_logits = self.weight_conv(concat)  # [B, L, H, W]
        w = F.softmax(w_logits, dim=1)

        out = sum(reduced[i] * w[:, i:i + 1, :, :] for i in range(self.num_levels))
        out = self.out_conv(out)
        return out


class L_ASFF(nn.Module):
    """
    Lightweight ASFF (Adaptive Spatial Feature Fusion with DPR).
    Args:
        level (int): target level (0,1,2...) defines which scale is the output reference.
        in_channels (list[int]): channels of each input level, e.g. [256,512,1024]
        out_channels (int): output channels (default = in_channels[0])
    """

    def __init__(self, level: int, in_channels: list, out_channels: int = None):
        super().__init__()
        assert isinstance(in_channels, (list, tuple)), "in_channels must be list"
        self.level = int(level)
        self.in_channels = list(in_channels)
        self.num_levels = len(self.in_channels)
        self.out_channels = out_channels or self.in_channels[0]

        # per-level reduction conv (lightweight)
        self.reduce_convs = nn.ModuleList([
            DPR(c, self.out_channels) for c in self.in_channels
        ])

        # weight generation conv (lightweight)
        self.weight_conv = nn.Sequential(
            DPR(self.out_channels * self.num_levels, self.num_levels, k=1, p=0, bias=True)
        )

        # final output conv (lightweight)
        self.out_conv = DPR(self.out_channels, self.out_channels, k=3, p=1)

    def forward(self, feats):
        """
        feats: list of tensors, length == num_levels
        returns: fused feature [B, out_channels, H_ref, W_ref]
        """
        assert isinstance(feats, (list, tuple)), "ASFF expects list of multi-scale features"
        assert len(feats) == self.num_levels, f"ASFF expects {self.num_levels} levels, got {len(feats)}"

        ref = feats[self.level]
        ref_h, ref_w = ref.shape[-2], ref.shape[-1]
        reduced = []
        for i, f in enumerate(feats):
            x = self.reduce_convs[i](f)
            if x.shape[-2:] != (ref_h, ref_w):
                x = F.interpolate(x, size=(ref_h, ref_w), mode="bilinear", align_corners=False)
            reduced.append(x)

        concat = torch.cat(reduced, dim=1)  # [B, out_ch * L, H, W]
        w_logits = self.weight_conv(concat)  # [B, L, H, W]
        w = F.softmax(w_logits, dim=1)

        out = sum(reduced[i] * w[:, i:i + 1, :, :] for i in range(self.num_levels))
        out = self.out_conv(out)
        return out


class LASFF(nn.Module):
    """
    Lightweight ASFF (Adaptive Spatial Feature Fusion with DPR).
    Args:
        level (int): target level (0,1,2...) defines which scale is the output reference.
        in_channels (list[int]): channels of each input level, e.g. [256,512,1024]
        out_channels (int): output channels (default = in_channels[0])
    """
    def __init__(self, level:int, in_channels:list, out_channels:int=None):
        super().__init__()
        assert isinstance(in_channels, (list, tuple)), "in_channels must be list"
        self.level = int(level)
        self.in_channels = list(in_channels)
        self.num_levels = len(self.in_channels)
        self.out_channels = out_channels or self.in_channels[0]

        # per-level reduction conv (lightweight)
        self.reduce_convs = nn.ModuleList([
            DPR(c, self.out_channels) for c in self.in_channels
        ])

        # weight generation conv (lightweight)
        self.weight_conv = nn.Sequential(
            DPR(self.out_channels * self.num_levels, self.num_levels, k=1, p=0, bias=True)
        )

        # final output conv (lightweight)
        self.out_conv = DPR(self.out_channels, self.out_channels, k=3, p=1)

    def forward(self, feats):
        """
        feats: list of tensors, length == num_levels
        returns: fused feature [B, out_channels, H_ref, W_ref]
        """
        assert isinstance(feats, (list, tuple)), "ASFF expects list of multi-scale features"
        assert len(feats) == self.num_levels, f"ASFF expects {self.num_levels} levels, got {len(feats)}"

        ref = feats[self.level]
        ref_h, ref_w = ref.shape[-2], ref.shape[-1]
        reduced = []
        for i, f in enumerate(feats):
            x = self.reduce_convs[i](f)
            if x.shape[-2:] != (ref_h, ref_w):
                x = F.interpolate(x, size=(ref_h, ref_w), mode="bilinear", align_corners=False)
            reduced.append(x)

        concat = torch.cat(reduced, dim=1)  # [B, out_ch * L, H, W]
        w_logits = self.weight_conv(concat)  # [B, L, H, W]
        w = F.softmax(w_logits, dim=1)

        out = sum(reduced[i] * w[:, i:i+1, :, :] for i in range(self.num_levels))
        out = self.out_conv(out)
        return out



class DP_Detect(nn.Module):
    """
    DP_Detect head (lightweight: DWConv + PWConv).
    Args:
        nc: number of classes
        in_channels: list of channels for each input level [P3, P4, P5]
        c_out: unified channel for head convs (default = in_channels[0])
        reg_max: DFL bins (default 16)
    """
    dynamic = False
    export = False
    format = None
    end2end = False
    max_det = 300
    shape = None
    anchors = torch.empty(0)
    strides = torch.empty(0)
    legacy = False

    def __init__(self, nc:int, in_channels:list, c_out:int=None, reg_max:int=16):
        super().__init__()
        assert isinstance(in_channels, (list, tuple)), "in_channels must be list"
        self.nc = int(nc)
        self.in_channels = list(in_channels)
        self.nl = len(self.in_channels)
        self.c_out = int(c_out) if c_out is not None else int(self.in_channels[0])
        self.reg_max = int(reg_max)
        self.no = self.nc + self.reg_max * 4
        self.stride = torch.zeros(self.nl)

        # per-level projection to c_out (1x1 conv if needed)
        self.proj = nn.ModuleList([
            nn.Conv2d(ch, self.c_out, 1, bias=False) if ch != self.c_out else nn.Identity()
            for ch in self.in_channels
        ])

        # per-level decoupled convs (use DPR)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.box_preds = nn.ModuleList()

        for _ in range(self.nl):
            # 使用轻量化 DWConv + PWConv
            self.cls_convs.append(DPR(self.c_out, self.c_out))
            # self.cls_convs.append(nn.Sequential(DPConv(self.c_out, self.c_out), DPConv(self.c_out, self.c_out)))
            # self.reg_convs.append(DPR(self.c_out, self.c_out))
            self.reg_convs.append(nn.Sequential(DPR(self.c_out, self.c_out), DPR(self.c_out, self.c_out)))
            # self.reg_convs.append(nn.Sequential(DPConv(self.c_out, self.c_out), DPConv(self.c_out, self.c_out)))


            self.cls_preds.append(nn.Conv2d(self.c_out, self.nc, 1))
            self.box_preds.append(nn.Conv2d(self.c_out, 4 * self.reg_max, 1))

        # DFL
        try:
            from ultralytics.nn.modules.head import DFL
            self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()
        except Exception:
            self.dfl = nn.Identity()

    def forward(self, feats):
        if isinstance(feats, torch.Tensor):
            feats = [feats]

        assert isinstance(feats, (list, tuple)), "DecoupledHead expects list of feature maps"
        assert len(feats) == self.nl, f"Expected {self.nl} levels, got {len(feats)}"

        # project
        feats = [p(f) for p, f in zip(self.proj, feats)]

        box_outs, cls_outs = [], []
        for i, f in enumerate(feats):
            cls_f = self.cls_convs[i](f)
            reg_f = self.reg_convs[i](f)
            cls_outs.append(self.cls_preds[i](cls_f))    # [B, nc, H, W]
            box_outs.append(self.box_preds[i](reg_f))    # [B, 4*reg_max, H, W]

        raw_preds = [torch.cat((b, c), dim=1) for b, c in zip(box_outs, cls_outs)]  # per-level

        if self.training:
            return raw_preds

        # inference
        shape = raw_preds[0].shape
        x_cat = torch.cat([rp.view(shape[0], self.no, -1) for rp in raw_preds], dim=2)

        if self.format != "imx" and (self.dynamic or self.shape != shape):
            try:
                from ultralytics.utils.tal import make_anchors
            except Exception:
                make_anchors = None
            if make_anchors is not None:
                self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(raw_preds, self.stride, 0.5))
            self.shape = shape

        box, cls = x_cat.split((self.reg_max * 4, self.nc), dim=1)
        try:
            from ultralytics.utils.tal import dist2bbox
            dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides
        except Exception:
            dbox = box
        y = torch.cat((dbox, cls.sigmoid()), 1)
        return (y, raw_preds) if not self.export else y

    def bias_init(self):
        for bp, cp, s in zip(self.box_preds, self.cls_preds, self.stride):
            if hasattr(bp, "bias") and bp.bias is not None:
                nn.init.constant_(bp.bias, 1.0)
            if hasattr(cp, "bias") and cp.bias is not None and s > 0:
                cp.bias.data[: self.nc] = math.log(torch.tensor(5 / self.nc / (640 / s) ** 2))

    def decode_bboxes(self, bboxes, anchors, xywh=True):
        try:
            from ultralytics.utils.tal import dist2bbox
            return dist2bbox(bboxes, anchors, xywh=xywh and (not self.end2end), dim=1)
        except Exception:
            return bboxes


class DBSPPF(nn.Module):

    def __init__(self, c1, c2, k=5, e=0.5, dk=3):
        super().__init__()
        c_ = max(1, int(c2 * e))
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 1, 1)
        self.p = nn.MaxPool2d(kernel_size=k, stride=1, padding=k//2)
        self.dp1 = DPConv(c_, c_, k=dk, s=1)
        self.dp2 = DPConv(c_, c_, k=dk, s=1)
        self.dp3 = DPConv(c_, c_, k=dk, s=1)
        self.cv4 = Conv(c_*4, c_, 1, 1)
        self.cv5 = Conv(c_*2, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        p1 = self.p(x2)
        p2 = self.p(p1)
        p3 = self.p(p2)
        a1 = x1 + p1
        d1 = self.dp1(a1)
        a2 = d1 + a1 +p2
        d2 = self.dp2(a2)
        a3 = d2 + a2 + p3
        d3 = self.dp3(a3)
        x1 = self.cv3(d3*x1)
        x2 = torch.cat([x2, p1, p2, p3], dim=1)
        x2 = self.cv4(x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.cv5(x)
        return x


class LAE(nn.Module):
    # Light-weight Adaptive Extraction
    def __init__(self, ch, group=16) -> None:
        super().__init__()

        self.softmax = nn.Softmax(dim=-1)
        self.attention = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            Conv(ch, ch, k=1)
        )

        self.ds_conv = Conv(ch, ch * 4, k=3, s=2, g=(ch // group))

    def forward(self, x):
        # bs, ch, 2*h, 2*w => bs, ch, h, w, 4
        att = rearrange(self.attention(x), 'bs ch (s1 h) (s2 w) -> bs ch h w (s1 s2)', s1=2, s2=2)
        att = self.softmax(att)

        # bs, 4 * ch, h, w => bs, ch, h, w, 4
        x = rearrange(self.ds_conv(x), 'bs (s ch) h w -> bs ch h w s', s=4)
        x = torch.sum(x * att, dim=-1)
        return x


class MatchNeck_Inner(nn.Module):
    def __init__(self, channels) -> None:
        super().__init__()

        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            Conv(channels, channels)
        )
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv_hw = Conv(channels, channels, (3, 1))
        self.conv_pool_hw = Conv(channels, channels, 1)

    def forward(self, x):
        _, _, h, w = x.size()
        x_pool_h, x_pool_w, x_pool_ch = self.pool_h(x), self.pool_w(x).permute(0, 1, 3, 2), self.gap(x)
        x_pool_hw = torch.cat([x_pool_h, x_pool_w], dim=2)
        x_pool_h, x_pool_w = torch.split(x_pool_hw, [h, w], dim=2)
        x_pool_hw_weight = x_pool_hw.sigmoid()
        x_pool_h_weight, x_pool_w_weight = torch.split(x_pool_hw_weight, [h, w], dim=2)
        x_pool_h, x_pool_w = x_pool_h * x_pool_h_weight, x_pool_w * x_pool_w_weight
        x_pool_ch = x_pool_ch * torch.mean(x_pool_hw_weight, dim=2, keepdim=True)
        return x * x_pool_h.sigmoid() * x_pool_w.permute(0, 1, 3, 2).sigmoid() * x_pool_ch.sigmoid()


class MatchNeck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2
        self.MN = MatchNeck_Inner(c2)

    def forward(self, x):
        return x + self.MN(self.cv2(self.cv1(x))) if self.add else self.MN(self.cv2(self.cv1(x)))


class MSFM(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(MatchNeck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class RFEMBranch(nn.Module):
    def __init__(self, c1, c_, d=3):
        super().__init__()
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c_, 3, 1, d=d)

    def forward(self, x):
        x = self.cv1(x)
        x = self.cv2(x)
        return x


class RFEM(nn.Module):
    def __init__(self, c1, c2, ds=(3, 5, 7), e=0.5, shortcut=True, act=True):
        super().__init__()
        assert len(ds) >= 1
        self.shortcut = shortcut and (c1 == c2)
        self.post_act = nn.SiLU(inplace=True)
        self.cv_shortcut = Conv(c1, c2, 1, 1, act=False)
        hidden_tatal = max(1, int(c2*e))
        branch_num = len(ds)
        branch_c = math.ceil(hidden_tatal/branch_num)
        self.branchs = nn.ModuleList([RFEMBranch(c1, branch_c, d=d) for d in ds])
        self.cv_fuse = Conv(branch_c*branch_num, c2, 1, 1, act=False)
        self.cv_identity = Conv(c1, c2, 1, 1, act=False)  if(c1 != c2 and not self.shortcut) else nn.Identity()

    def forward(self, x):
        identity = self.cv_shortcut(x) if self.shortcut else self.cv_identity(x)
        ys = [branch(x) for branch in self.branchs]
        y = torch.cat(ys, dim=1)
        y = self.cv_fuse(y)
        y = y + identity
        y = self.post_act(y)
        return y


class SPD_DPConv(nn.Module):

    def __init__(self, c1, c2, k=3, act=True):
        super().__init__()
        self.spd = space_to_depth()
        c_ = 4 * c1
        self.cv = DPConv(c_, c2, 3, 1, act=act)

    def forward(self, x):
        x = self.spd(x)
        x = self.cv(x)
        return x


class AsymPadConv(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, mode='h', pad=(0, 0, 0, 0), act=True):
        super().__init__()
        assert mode in ('h', 'v')
        self.pad = nn.ZeroPad2d(pad)
        if mode == 'h':
            kernel = (1, k)
        else:
            kernel = (k, 1)
        self.conv = Conv(c1, c2, k=kernel, s=s, p=0, act=act)
    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        return x


class PiConv(nn.Module):
    """风车卷积 — 对输入的部分通道执行3×3卷积。"""
    def __init__(self, c1, c2, k=3, s=1, act=True):
        super().__init__()
        branch_c = math.ceil(c2 / 4)
        self.branch_C = branch_c
        self.s = s
        self.k = k
        self.b1 = AsymPadConv(c1, branch_c, k=k, s=s, mode='h', pad=(0, k, 1, 0), act=act)
        self.b2 = AsymPadConv(c1, branch_c, k=k, s=s, mode='v', pad=(0, 1, 1, k), act=act)
        self.b3 = AsymPadConv(c1, branch_c, k=k, s=s, mode='h', pad=(k, 0, 0, 1), act=act)
        self.b4 = AsymPadConv(c1, branch_c, k=k, s=s, mode='v', pad=(1, 0, k, 0), act=act)
        self.fuse = Conv(branch_c * 4, c2, k=2, s=1, p=0, act=act)

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        y = torch.cat([y1, y2, y3, y4], dim=1)
        y = self.fuse(y)
        return y


class SK(nn.Module):

    def __init__(self, channel=512, kernels=[1, 3, 5, 7], reduction=16, group=1, L=32):
        super().__init__()
        self.d = max(L, channel // reduction)
        self.convs = nn.ModuleList([])
        for k in kernels:
            self.convs.append(
                nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(channel, channel, kernel_size=k, padding=k // 2, groups=group)),
                    ('bn', nn.BatchNorm2d(channel)),
                    ('relu', nn.ReLU())
                ]))
            )
        self.fc = nn.Linear(channel, self.d)
        self.fcs = nn.ModuleList([])
        for i in range(len(kernels)):
            self.fcs.append(nn.Linear(self.d, channel))
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        bs, c, _, _ = x.size()
        conv_outs = []
        ### split
        for conv in self.convs:
            conv_outs.append(conv(x))
        feats = torch.stack(conv_outs, 0)  # k,bs,channel,h,w

        ### fuse
        U = sum(conv_outs)  # bs,c,h,w

        ### reduction channel
        S = U.mean(-1).mean(-1)  # bs,c
        Z = self.fc(S)  # bs,d

        ### calculate attention weight
        weights = []
        for fc in self.fcs:
            weight = fc(Z)
            weights.append(weight.view(bs, c, 1, 1))  # bs,channel
        attention_weughts = torch.stack(weights, 0)  # k,bs,channel,1,1
        attention_weughts = self.softmax(attention_weughts)  # k,bs,channel,1,1

        ### fuse
        V = (attention_weughts * feats).sum(0)
        return V
    

class ELA(nn.Module):
    def __init__(self, channels: int, k: int=7, gn_groups: int=8, use_residual: bool=False):
        super().__init__()
        assert k % 2 == 1
        self.C = channels
        self.k = k
        self.use_residual = use_residual
        g = int(channels / gn_groups)
        self.gn = nn.GroupNorm(num_groups=g, num_channels=channels)
        self.cv_h = nn.Conv1d(channels, channels, kernel_size=k, padding=k // 2, groups=1, bias=True)
        self.cv_w = nn.Conv1d(channels, channels, kernel_size=k, padding=k // 2, groups=1, bias=True)
        self.sigmod = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_gn = self.gn(x)
        x_h = x_gn.mean(dim=3, keepdim=True)
        x_w = x_gn.mean(dim=2, keepdim=True)
        x_h_1d = x_h.squeeze(3)
        x_w_1d = x_w.squeeze(2)
        a_h = self.sigmod(self.cv_h(x_h_1d))
        a_w = self.sigmod(self.cv_w(x_w_1d))
        a_h = a_h.squeeze(3)
        a_w = a_w.squeeze(2)
        out = x * a_h * a_w
        if self.use_residual:
            out = out + x
        return x


class ELA72(nn.Module):
    def __init__(self,  channel, ks=7):
        super(ELA72, self).__init__()

        p = ks // 2
        self.conv = nn.Conv1d(channel, channel, kernel_size=ks, padding=p, groups=channel//8, bias=False)
        self.gn = nn.GroupNorm(16, channel)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()

        x_h = torch.mean(x, dim=3, keepdim=True).view(b, c, h)
        x_w = torch.mean(x, dim=2, keepdim=True).view(b, c, w)

        x_h = self.sig(self.gn(self.conv(x_h))).view(b, c, h, 1)
        x_w = self.sig(self.gn(self.conv(x_w))).view(b, c, 1, w)

        return x * x_h * x_w


class SimAM(torch.nn.Module):
    def __init__(self, c,e_lambda=1e-4):
        super(SimAM, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)


class PiBottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = PiConv(c_, c2, k[1], 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class PiC3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(PiBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))


class PiC3k2(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            PiC3k(self.c, self.c, 2, shortcut, g) if c3k else PiBottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )


class PBottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = PConv(c_, c2, )
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class PC3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(PBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))


class PC3k2(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            PC3k(self.c, self.c, 2, shortcut, g) if c3k else PBottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )


class MLCA(nn.Module):
    def __init__(self, in_size, local_size=5, gamma = 2, b = 1,local_weight=0.5):
        super(MLCA, self).__init__()

        # ECA 计算方法
        self.local_size=local_size
        self.gamma = gamma
        self.b = b
        t = int(abs(math.log(in_size, 2) + self.b) / self.gamma)   # eca  gamma=2
        k = t if t % 2 else t + 1

        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.conv_local = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)

        self.local_weight=local_weight

        self.local_arv_pool = nn.AdaptiveAvgPool2d(local_size)
        self.global_arv_pool=nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        local_arv=self.local_arv_pool(x)
        global_arv=self.global_arv_pool(local_arv)

        b,c,m,n = x.shape
        b_local, c_local, m_local, n_local = local_arv.shape

        # (b,c,local_size,local_size) -> (b,c,local_size*local_size) -> (b,local_size*local_size,c) -> (b,1,local_size*local_size*c)
        temp_local= local_arv.view(b, c_local, -1).transpose(-1, -2).reshape(b, 1, -1)
        # (b,c,1,1) -> (b,c,1) -> (b,1,c)
        temp_global = global_arv.view(b, c, -1).transpose(-1, -2)

        y_local = self.conv_local(temp_local)
        y_global = self.conv(temp_global)

        # (b,c,local_size,local_size) <- (b,c,local_size*local_size)<-(b,local_size*local_size,c) <- (b,1,local_size*local_size*c)
        y_local_transpose=y_local.reshape(b, self.local_size * self.local_size,c).transpose(-1,-2).view(b, c, self.local_size , self.local_size)
        # (b,1,c) -> (b,c,1) -> (b,c,1,1)
        y_global_transpose = y_global.transpose(-1,-2).unsqueeze(-1)

        # 反池化
        att_local = y_local_transpose.sigmoid()
        att_global = F.adaptive_avg_pool2d(y_global_transpose.sigmoid(),[self.local_size, self.local_size])
        att_all = F.adaptive_avg_pool2d(att_global*(1-self.local_weight)+(att_local*self.local_weight), [m, n])

        x = x * att_all
        return x


class MHSA(nn.Module):
    def __init__(self, n_dims, width=14, height=14, heads=4, pos_emb=False):
        super(MHSA, self).__init__()

        self.heads = heads
        self.query = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.key = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.value = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.pos = pos_emb
        if self.pos:
            self.rel_h_weight = nn.Parameter(torch.randn([1, heads, (n_dims) // heads, 1, int(height)]),
                                             requires_grad=True)
            self.rel_w_weight = nn.Parameter(torch.randn([1, heads, (n_dims) // heads, int(width), 1]),
                                             requires_grad=True)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        n_batch, C, width, height = x.size()
        q = self.query(x).view(n_batch, self.heads, C // self.heads, -1)
        k = self.key(x).view(n_batch, self.heads, C // self.heads, -1)
        v = self.value(x).view(n_batch, self.heads, C // self.heads, -1)
        content_content = torch.matmul(q.permute(0, 1, 3, 2), k)  # 1,C,h*w,h*w
        c1, c2, c3, c4 = content_content.size()
        if self.pos:
            content_position = (self.rel_h_weight + self.rel_w_weight).view(1, self.heads, C // self.heads, -1).permute(
                0, 1, 3, 2)  # 1,4,1024,64

            content_position = torch.matmul(content_position, q)  # ([1, 4, 1024, 256])
            content_position = content_position if (
                    content_content.shape == content_position.shape) else content_position[:, :, :c3, ]
            assert (content_content.shape == content_position.shape)
            energy = content_content + content_position
        else:
            energy = content_content
        attention = self.softmax(energy)
        out = torch.matmul(v, attention.permute(0, 1, 3, 2))  # 1,4,256,64
        out = out.view(n_batch, C, width, height)
        return out


class PsConv(nn.Module):
    ''' Pinwheel-shaped Convolution using the Asymmetric Padding method. '''

    def __init__(self, c1, c2, k, s):
        super().__init__()

        # self.k = k
        p = [(k, 0, 1, 0), (0, k, 0, 1), (0, 1, k, 0), (1, 0, 0, k)]
        self.pad = [nn.ZeroPad2d(padding=(p[g])) for g in range(4)]
        self.cw = Conv(c1, c2 // 4, (1, k), s=s, p=0)
        self.ch = Conv(c1, c2 // 4, (k, 1), s=s, p=0)
        self.cat = Conv(c2, c2, 2, s=1, p=0)

    def forward(self, x):
        yw0 = self.cw(self.pad[0](x))
        yw1 = self.cw(self.pad[1](x))
        yh0 = self.ch(self.pad[2](x))
        yh1 = self.ch(self.pad[3](x))
        return self.cat(torch.cat([yw0, yw1, yh0, yh1], dim=1))


class APC2f(nn.Module):
    """Faster Implementation of APCSP Bottleneck with Asymmetric Padding convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, P=True, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        if P:
            self.m = nn.ModuleList(
                PsBottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        else:
            self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through APC2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class PsBottleneck(nn.Module):
    """Asymmetric Padding bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        p = [(2, 0, 2, 0), (0, 2, 0, 2), (0, 2, 2, 0), (2, 0, 0, 2)]
        self.pad = [nn.ZeroPad2d(padding=(p[g])) for g in range(4)]
        self.cv1 = Conv(c1, c_ // 4, k[0], 1, p=0)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2((torch.cat([self.cv1(self.pad[g](x)) for g in range(4)], 1))) if self.add else self.cv2(
            (torch.cat([self.cv1(self.pad[g](x)) for g in range(4)], 1)))


class PsC3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(PsBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))


class PsC3k2(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            PsC3k(self.c, self.c, 2, shortcut, g) if c3k else PsBottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )


import copy

class PDetect(nn.Module):
    """YOLO Detect head for detection models."""

    dynamic = False  # force grid reconstruction
    export = False  # export mode
    format = None  # export format
    end2end = False  # end2end
    max_det = 300  # max_det
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init
    legacy = False  # backward compatibility for v3/v5/v8/v9 models
    light = False
    def __init__(self, nc=80, ch=()):
        """Initializes the YOLO detection layer with specified number of classes and channels."""
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        if self.light:
            self.cv2 = nn.ModuleList(
                nn.Sequential(Conv(x, c2, 3), PConv(c2, c2,), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
            )
        else:
            self.cv2 = nn.ModuleList(
                nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
            )

        if self.light:
            self.cv3 = (
                nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
                if self.legacy
                else nn.ModuleList(
                    nn.Sequential(
                        nn.Sequential(PConv(x, x,), Conv(x, c3, 1)),
                        nn.Sequential(PConv(c3, c3,), Conv(c3, c3, 1)),
                        nn.Conv2d(c3, self.nc, 1),
                    )
                    for x in ch
                )
            )
        else:
            self.cv3 = (
                nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
                if self.legacy
                else nn.ModuleList(
                    nn.Sequential(
                        nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                        nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                        nn.Conv2d(c3, self.nc, 1),
                    )
                    for x in ch
                )
            )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        if self.end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        if self.end2end:
            return self.forward_end2end(x)

        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:  # Training path
            return x
        y = self._inference(x)
        return y if self.export else (y, x)

    def forward_end2end(self, x):
        """
        Performs forward pass of the v10Detect module.

        Args:
            x (tensor): Input tensor.

        Returns:
            (dict, tensor): If not in training mode, returns a dictionary containing the outputs of both one2many and one2one detections.
                           If in training mode, returns a dictionary containing the outputs of one2many and one2one detections separately.
        """
        x_detach = [xi.detach() for xi in x]
        one2one = [
            torch.cat((self.one2one_cv2[i](x_detach[i]), self.one2one_cv3[i](x_detach[i])), 1) for i in range(self.nl)
        ]
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:  # Training path
            return {"one2many": x, "one2one": one2one}

        y = self._inference(one2one)
        y = self.postprocess(y.permute(0, 2, 1), self.max_det, self.nc)
        return y if self.export else (y, {"one2many": x, "one2one": one2one})

    def _inference(self, x):
        """Decode predicted bounding boxes and class probabilities based on multiple-level feature maps."""
        # Inference path
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.format != "imx" and (self.dynamic or self.shape != shape):
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.export and self.format in {"saved_model", "pb", "tflite", "edgetpu", "tfjs"}:  # avoid TF FlexSplitV ops
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        if self.export and self.format in {"tflite", "edgetpu"}:
            # Precompute normalization factor to increase numerical stability
            # See https://github.com/ultralytics/ultralytics/issues/7371
            grid_h = shape[2]
            grid_w = shape[3]
            grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * grid_size)
            dbox = self.decode_bboxes(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2])
        elif self.export and self.format == "imx":
            dbox = self.decode_bboxes(
                self.dfl(box) * self.strides, self.anchors.unsqueeze(0) * self.strides, xywh=False
            )
            return dbox.transpose(1, 2), cls.sigmoid().permute(0, 2, 1)
        else:
            dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

        return torch.cat((dbox, cls.sigmoid()), 1)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)
        if self.end2end:
            for a, b, s in zip(m.one2one_cv2, m.one2one_cv3, m.stride):  # from
                a[-1].bias.data[:] = 1.0  # box
                b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)

    def decode_bboxes(self, bboxes, anchors, xywh=True):
        """Decode bounding boxes."""
        return dist2bbox(bboxes, anchors, xywh=xywh and (not self.end2end), dim=1)

    @staticmethod
    def postprocess(preds: torch.Tensor, max_det: int, nc: int = 80):
        """
        Post-processes YOLO model predictions.

        Args:
            preds (torch.Tensor): Raw predictions with shape (batch_size, num_anchors, 4 + nc) with last dimension
                format [x, y, w, h, class_probs].
            max_det (int): Maximum detections per image.
            nc (int, optional): Number of classes. Default: 80.

        Returns:
            (torch.Tensor): Processed predictions with shape (batch_size, min(max_det, num_anchors), 6) and last
                dimension format [x, y, w, h, max_class_prob, class_index].
        """
        batch_size, anchors, _ = preds.shape  # i.e. shape(16,8400,84)
        boxes, scores = preds.split([4, nc], dim=-1)
        index = scores.amax(dim=-1).topk(min(max_det, anchors))[1].unsqueeze(-1)
        boxes = boxes.gather(dim=1, index=index.repeat(1, 1, 4))
        scores = scores.gather(dim=1, index=index.repeat(1, 1, nc))
        scores, index = scores.flatten(1).topk(min(max_det, anchors))
        i = torch.arange(batch_size)[..., None]  # batch indices
        return torch.cat([boxes[i, index // nc], scores[..., None], (index % nc)[..., None].float()], dim=-1)
