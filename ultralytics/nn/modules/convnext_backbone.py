# filename: convnext_backbone.py
# Usage:
#   from convnext_backbone import ConvNeXt
#   backbone = ConvNeXt('tiny', pretrained=True, out_indices=(1, 2, 3), out_channels=(256, 512, 1024))
#
# 在 YAML 里（示例）:
#   backbone:
#     - [-1, 1, ConvNeXt, ['tiny', True, [1,2,3], [256,512,1024]]]

from typing import Iterable, List, Tuple, Union, Optional
import torch
import torch.nn as nn

try:
    from timm import create_model
except Exception as e:
    create_model = None


def _to_list3(x: Union[int, Iterable[int], None], ref: List[int]) -> Optional[List[int]]:
    """将 int 或列表规范为与 ref 等长的列表；None 则返回 None。"""
    if x is None:
        return None
    if isinstance(x, int):
        return [x] * len(ref)
    x = list(x)
    assert len(x) == len(ref), f"Expected len(out_channels)=={len(ref)}, but got {len(x)}"
    return x


class ConvNeXt(nn.Module):
    """
    ConvNeXt backbone 适配器（timm 后端），用于 Ultralytics 风格模型图解析。
    - 支持 variant: 'tiny' | 'small' | 'base' | 'large' | 'xlarge'（timm 里存在的名字都可）
    - 默认输出 P3/P4/P5（stride 8/16/32），可通过 out_indices 更改
    - 可选用 1×1 conv 将通道映射到指定维度（out_channels）

    Args:
        variant: ConvNeXt 类型字符串，例如 'tiny', 'small', 'base', 'large'
        pretrained: 是否加载 ImageNet 预训练
        out_indices: 输出 stage 索引（基于 timm 的 features_only 索引）
                     对 convnext 系列通常：
                       0->P2/4, 1->P3/8, 2->P4/16, 3->P5/32
                     因此 (1,2,3) 对应 P3/P4/P5
        out_channels: 输出映射通道数；可以是单个 int（对三个尺度相同），或长度与 out_indices 相同的列表。
                      若为 None 则不做通道对齐，直接返回 timm 的原通道。
        in_chans: 输入图像通道数（默认 3）
        fuse_bn: 是否在 1x1 映射上融合 BN（导出/部署时可选）
        act: 1x1 映射后的激活，默认 None（不加激活）
    """

    def __init__(
        self,
        variant: str = "tiny",
        pretrained: bool = True,
        out_indices: Tuple[int, ...] = (1, 2, 3),
        out_channels: Union[int, Iterable[int], None] = None,
        in_chans: int = 3,
        fuse_bn: bool = False,
        act: Optional[str] = None,
    ):
        super().__init__()
        if create_model is None:
            raise ImportError(
                "timm 未安装。请先 `pip install timm` 再使用 ConvNeXt 适配器。"
            )

        # 构建 timm ConvNeXt 主干
        model_name = f"convnext_{variant}"
        self.backbone = create_model(
            model_name,
            features_only=True,        # 输出多尺度特征
            out_indices=out_indices,   # 只要这些 level
            pretrained=pretrained,
            in_chans=in_chans,
        )

        # timm 的每个输出层通道数
        feat_info = self.backbone.feature_info
        self.native_channels: List[int] = [feat_info[i]["num_chs"] for i in range(len(feat_info))]
        self.out_indices = tuple(out_indices)
        self.export_channels = _to_list3(out_channels, list(out_indices))

        # 可选的 1×1 通道映射头
        self.proj_heads = nn.ModuleList()
        if self.export_channels is None:
            # 不做投影
            for _ in self.out_indices:
                self.proj_heads.append(nn.Identity())
            self.out_channels: List[int] = [self.native_channels[i] for i in self.out_indices]
        else:
            self.out_channels = self.export_channels
            for want_c, idx in zip(self.export_channels, self.out_indices):
                in_c = self.native_channels[idx]
                head = nn.Sequential(
                    nn.Conv2d(in_c, want_c, kernel_size=1, bias=False),
                    nn.BatchNorm2d(want_c),
                    _make_activation(act) if act else nn.Identity(),
                )
                self.proj_heads.append(head)

        # 可选 BN 融合（仅对 1x1 映射）
        if fuse_bn and self.export_channels is not None:
            self._fuse_all_1x1_bn()

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        返回 [P3, P4, P5]（或根据 out_indices 指定的顺序/层数）。
        Ultralytics Detect/Segment/Pose 头可直接消费本列表。
        """
        feats = self.backbone(x)   # list[Tensor], 顺序与 out_indices 对齐
        assert len(feats) == len(self.out_indices), "timm features_only 输出与 out_indices 数量不一致"

        outs: List[torch.Tensor] = []
        for f, head in zip(feats, self.proj_heads):
            outs.append(head(f))
        return outs  # e.g., [P3, P4, P5]

    # --------------------- helpers ---------------------

    def _fuse_all_1x1_bn(self):
        """将 1×1 Conv 和后续 BN 融合为一个卷积（导出友好）。"""
        for i, head in enumerate(self.proj_heads):
            if isinstance(head, nn.Sequential) and len(head) >= 2:
                conv = head[0]
                bn = head[1]
                if isinstance(conv, nn.Conv2d) and conv.kernel_size == (1, 1) and isinstance(bn, nn.BatchNorm2d):
                    fused = fuse_conv_bn(conv, bn)
                    # 替换前两层
                    new_seq = [fused]
                    # 保留后续激活（若有）
                    for layer in head[2:]:
                        new_seq.append(layer)
                    self.proj_heads[i] = nn.Sequential(*new_seq)


def _make_activation(name: Optional[str]) -> nn.Module:
    if name is None:
        return nn.Identity()
    name = name.lower()
    if name in ("relu", "relu6"):
        return nn.ReLU(inplace=True)
    if name in ("silu", "swish"):
        return nn.SiLU(inplace=True)
    if name in ("gelu",):
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {name}")


def fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """参考常见的 Conv+BN 融合，实现 1×1 卷积与 BN 的融合。"""
    with torch.no_grad():
        # 准备权重/偏置
        w = conv.weight.clone().view(conv.out_channels, -1)
        if conv.bias is None:
            bias = torch.zeros(conv.out_channels, device=w.device, dtype=w.dtype)
        else:
            bias = conv.bias.clone()

        # BN 参数
        gamma = bn.weight
        beta = bn.bias
        mean = bn.running_mean
        var = bn.running_var
        eps = bn.eps

        # 融合
        std = torch.sqrt(var + eps)
        w_fused = (gamma / std).reshape(-1, 1) * w
        b_fused = beta + (bias - mean) * gamma / std

        # 回填到新的 conv
        fused_conv = nn.Conv2d(
            conv.in_channels, conv.out_channels, kernel_size=conv.kernel_size,
            stride=conv.stride, padding=conv.padding, dilation=conv.dilation,
            groups=conv.groups, bias=True, padding_mode=conv.padding_mode,
        )
        fused_conv.weight.copy_(w_fused.view_as(conv.weight))
        fused_conv.bias.copy_(b_fused)
        return fused_conv

