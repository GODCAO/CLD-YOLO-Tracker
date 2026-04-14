import math
from typing import List

import torch
from timm.layers import DropPath
from torch import Tensor, nn

from ultralytics.nn.modules import Conv


class PConv(nn.Module):
    def __init__(self, dim, ouc, n_div=4, forward='split_cat'):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)
        self.conv = Conv(dim, ouc, k=1)

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x):
        # only for inference
        x = x.clone()   # !!! Keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])
        x = self.conv(x)
        return x

    def forward_split_cat(self, x):
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)
        x = self.conv(x)
        return x


class FasterNetBlock(nn.Module):
    """FasterNet Block: 结合 PConv + Conv1x1 + BN + ReLU + Conv1x1 + 残差"""
    def __init__(self,
                 dim,
                 n_div,
                 drop_path=0.,
                 layer_scale_init_value=0,
                 act_layer=nn.ReLU,
                 norm_layer=nn.BatchNorm2d,
                 pconv_fw_type='split_cat',
                 mlp_ratio=2,
                 ):

        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.drop_path =  nn.Identity() #DropPath(drop_path) if drop_path > 0. else
        self.n_div = n_div

        mlp_hidden_dim = int(dim * mlp_ratio)

        mlp_layer: List[nn.Module] = [
            nn.Conv2d(dim, mlp_hidden_dim, 1, bias=False),
            norm_layer(mlp_hidden_dim),
            act_layer(),
            nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False)
        ]

        self.mlp = nn.Sequential(*mlp_layer)

        self.spatial_mixing = PConv(
            dim,
            n_div,
            pconv_fw_type
        )

        if layer_scale_init_value > 0:
            self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.forward = self.forward_layer_scale
        else:
            self.forward = self.forward

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(self.mlp(x))
        return x

    def forward_layer_scale(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(
            self.layer_scale.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
        return x



class PatchEmbed(nn.Module):

    def __init__(self, in_chans, embed_dim):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=4, stride=4, bias=False)
        self.norm = nn.BatchNorm2d(embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(self.proj(x))
        return x



class PatchMerging(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.reduction = nn.Conv2d(dim, 2 * dim, kernel_size=2, stride=2, bias=False)
        self.norm = nn.BatchNorm2d(2 * dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(self.reduction(x))
        return x


