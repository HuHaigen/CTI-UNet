
from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_activation_layer, build_norm_layer
from mmcv.runner import ModuleList
from torch.nn.modules.utils import _pair as to_2tuple

from mmseg.ops import resize
from ..builder import BACKBONES
from ..utils import PatchEmbed
from .unet import UNet
from .vit import TransformerEncoderLayer


@BACKBONES.register_module()
class CTIUNet(UNet):

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 pre_num_layers=6,
                 num_layers=1,
                 interacted=(2, 3, 4),
                 embed_dims=384,
                 num_heads=6,
                 mlp_ratio=4,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 num_fcs=2,
                 qkv_bias=True,
                 trans_act_cfg=dict(type='GELU'),
                 trans_norm_cfg=dict(type='LN'),
                 with_cp=False,
                 aux_head=False,
                 pos_emb=False,
                 ca_cfg=None,
                 sa_cfg=None,
                 **kwargs):
        super(CTIUNet, self).__init__(**kwargs)

        assert ca_cfg is None or isinstance(ca_cfg, dict)
        assert sa_cfg is None or isinstance(sa_cfg, dict)

        if isinstance(img_size, int):
            img_size = to_2tuple(img_size)
        elif isinstance(img_size, tuple):
            if len(img_size) == 1:
                img_size = to_2tuple(img_size[0])
            assert len(img_size) == 2, \
                f'The size of image should have length 1 or 2, ' \
                f'but got {len(img_size)}'

        self.aux_head = aux_head

        num_stages = kwargs['num_stages']
        strides = kwargs['strides']
        downsamples = kwargs['downsamples']
        in_channels = kwargs['in_channels']
        base_channels = kwargs['base_channels']

        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            embed_dims=embed_dims,
            conv_type='Conv2d',
            kernel_size=patch_size,
            stride=patch_size,
            padding='corner',
            norm_cfg=trans_norm_cfg if False else None,
            init_cfg=None,
        )

        num_patches = (img_size[0] // patch_size) * \
            (img_size[1] // patch_size)

        self.pos_emb = pos_emb

        if self.pos_emb:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, embed_dims))
            self.drop_after_pos = nn.Dropout(p=drop_rate)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, pre_num_layers)
        ]  # stochastic depth decay rule
        self.trans_layers = ModuleList()
        for i in range(pre_num_layers):
            self.trans_layers.append(
                TransformerEncoderLayer(
                    embed_dims=embed_dims,
                    num_heads=num_heads,
                    feedforward_channels=mlp_ratio * embed_dims,
                    attn_drop_rate=attn_drop_rate,
                    drop_rate=drop_rate,
                    drop_path_rate=dpr[i],
                    num_fcs=num_fcs,
                    qkv_bias=qkv_bias,
                    act_cfg=trans_act_cfg,
                    norm_cfg=trans_norm_cfg,
                    with_cp=with_cp,
                    batch_first=True))

        self.encoder = nn.ModuleList()
        for i in range(num_stages):
            self.encoder.append(ConvTransBlock(
                in_channels=in_channels,
                out_channels=base_channels * 2**i,
                dw_stride=224 // 14 // 2**(i-1) if i != 0 else 16,
                stride=strides[i],
                down_sampled=(i != 0 and downsamples[i-1]),
                dilation=kwargs['enc_dilations'][i],
                conv_cfg=kwargs['conv_cfg'],
                norm_cfg=kwargs['norm_cfg'],
                act_cfg=kwargs['act_cfg'],
                interacted=i in interacted,
                embed_dims=embed_dims,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                attn_drop_rate=attn_drop_rate,
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate,
                num_fcs=num_fcs,
                qkv_bias=qkv_bias,
                with_cp=with_cp,
                trans_norm_cfg=trans_norm_cfg,
                trans_act_cfg=trans_act_cfg,
                ca_cfg=ca_cfg,
                sa_cfg=sa_cfg,
                num_layers=num_layers
            ))
            in_channels = base_channels * 2**i

    def forward(self, x):
        self._check_input_divisible(x)

        x_t, hw_shape = self.patch_embed(x)

        if self.pos_emb:
            x_t = self._pos_embeding(x_t, hw_shape, self.pos_embed)

        for layer in self.trans_layers:
            x_t = layer(x_t)

        enc_outs = []
        if self.aux_head:
            for i, enc in enumerate(self.encoder):
                if i < len(self.encoder) - 1:
                    x, x_t = enc(x, x_t)
                else:
                    x = enc(x, x_t, return_x_t=False)
                enc_outs.append(x)
        else:
            for i, enc in enumerate(self.encoder):
                x, x_t = enc(x, x_t)
                enc_outs.append(x)

        dec_outs = [x]
        for i in reversed(range(len(self.decoder))):
            x = self.decoder[i](enc_outs[i], x)
            dec_outs.append(x)

        if self.aux_head:
            dec_outs.append(x_t)
        return dec_outs

    def _pos_embeding(self, patched_img, hw_shape, pos_embed):
        assert patched_img.ndim == 3 and pos_embed.ndim == 3, \
            'the shapes of patched_img and pos_embed must be [B, L, C]'
        x_len, pos_len = patched_img.shape[1], pos_embed.shape[1]
        if x_len != pos_len:
            if pos_len == (self.img_size[0] // self.patch_size) * (
                    self.img_size[1] // self.patch_size):
                pos_h = self.img_size[0] // self.patch_size
                pos_w = self.img_size[1] // self.patch_size
            else:
                raise ValueError(
                    'Unexpected shape of pos_embed, got {}.'.format(
                        pos_embed.shape))
            pos_embed = self.resize_pos_embed(pos_embed, hw_shape,
                                              (pos_h, pos_w),
                                              self.interpolate_mode)
        return self.drop_after_pos(patched_img + pos_embed)

    @staticmethod
    def resize_pos_embed(pos_embed, input_shpae, pos_shape, mode):
        assert pos_embed.ndim == 3, 'shape of pos_embed must be [B, L, C]'
        pos_h, pos_w = pos_shape

        pos_embed_weight = pos_embed
        pos_embed_weight = pos_embed_weight.reshape(
            1, pos_h, pos_w, pos_embed.shape[2]).permute(0, 3, 1, 2)
        pos_embed_weight = resize(
            pos_embed_weight, size=input_shpae, align_corners=False, mode=mode)
        pos_embed_weight = torch.flatten(pos_embed_weight, 2).transpose(1, 2)
        return pos_embed_weight


class FuseDown(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 dw_stride,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN')):
        super(FuseDown, self).__init__()
        self.dw_stride = dw_stride

        self.conv_project = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.avg_pooling = nn.AvgPool2d(
            kernel_size=dw_stride, stride=dw_stride)

        self.ln = build_norm_layer(norm_cfg, out_channels)[1]
        self.act = build_activation_layer(act_cfg)

    def forward(self, x):
        x = self.conv_project(x)

        x = self.avg_pooling(x).flatten(2).transpose(1, 2).contiguous()
        x = self.ln(x)
        x = self.act(x)

        return x


class FuseUp(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 up_stride,
                 act_cfg=dict(type='ReLU'),
                 norm_cfg=dict(type='BN')):
        super(FuseUp, self).__init__()

        self.up_stride = up_stride
        self.conv_project = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn = build_norm_layer(norm_cfg, out_channels)[1]
        self.act = build_activation_layer(act_cfg)

    def forward(self, x, H, W):
        B, _, C = x.shape
        x_r = x.transpose(1, 2).contiguous().reshape(B, C, 14, 14)
        x_r = self.act(self.bn(self.conv_project(x_r)))

        return F.interpolate(x_r, size=(H // 2, W // 2))


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(
            2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ConvTransBlock(nn.Module):
    """
    Basic module for ConvTransformer, keep feature maps for CNN block and patch embeddings for transformer encoder block
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 dw_stride,
                 stride=1,
                 down_sampled=False,
                 dilation=1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 dcn=None,
                 plugins=None,
                 interacted=True,
                 embed_dims=384,
                 num_heads=12,
                 mlp_ratio=4,
                 attn_drop_rate=0.,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 num_fcs=1,
                 qkv_bias=True,
                 with_cp=False,
                 trans_norm_cfg=dict(type='LN'),
                 trans_act_cfg=dict(type='GELU'),
                 num_layers=1,
                 ca_cfg=None,
                 sa_cfg=None):

        super(ConvTransBlock, self).__init__()
        assert dcn is None, 'Not implemented yet.'
        assert plugins is None, 'Not implemented yet.'

        self.interacted = interacted
        self.down_sampled = down_sampled
        self.with_ca = ca_cfg is not None and interacted
        self.with_sa = sa_cfg is not None and interacted

        if self.with_ca:
            self.ca = _build_feat_ca(
                ca_cfg,
                in_channels=embed_dims
            )
        if self.with_sa:
            self.sa = _build_feat_sa(sa_cfg)

        if self.down_sampled:
            self.max_pooling = nn.MaxPool2d(kernel_size=2)

        if self.interacted:
            self.trans_block = nn.Sequential(
                *[TransformerEncoderLayer(
                    embed_dims=embed_dims,
                    num_heads=num_heads,
                    feedforward_channels=mlp_ratio * embed_dims,
                    attn_drop_rate=attn_drop_rate,
                    drop_rate=drop_rate,
                    drop_path_rate=torch.linspace(0, drop_path_rate, 1).item(),
                    num_fcs=num_fcs,
                    qkv_bias=qkv_bias,
                    act_cfg=trans_act_cfg,
                    norm_cfg=trans_norm_cfg,
                    with_cp=with_cp,
                    batch_first=True) for _ in range(num_layers)]
            )

            self.fcu_down = FuseDown(
                in_channels=in_channels, out_channels=embed_dims, dw_stride=dw_stride)
            self.fcu_up = FuseUp(
                in_channels=embed_dims, out_channels=in_channels * 2, up_stride=dw_stride)

        self.conv1 = ConvModule(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                                stride=stride, dilation=1, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv2 = ConvModule(in_channels=out_channels, out_channels=out_channels, kernel_size=3,
                                stride=1, dilation=dilation, padding=dilation, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, x, x_t, return_x_t=True):

        if not self.interacted:
            if self.down_sampled:
                return self.conv2(self.conv1(self.max_pooling(x))), x_t
            return self.conv2(self.conv1(x)), x_t

        _, _, H, W = x.shape

        x_down = self.fcu_down(x)

        x_t = x_down + x_t
        if self.with_ca:
            B, N, C = x_t.shape
            x_t = x_t.permute(0, 2, 1).contiguous().view(B, C, 14, 14)
            x_t = (self.ca(x_t) * x_t).view(B, C, -1).permute(0, 2, 1)

        x_t = self.trans_block(x_t)

        x_up = self.fcu_up(x_t, H, W)

        if self.down_sampled:
            x = self.conv1(self.max_pooling(x))
        else:
            x = self.conv1(x)

        x = x + x_up
        if self.with_sa:
            x = self.sa(x) * x

        x = self.conv2(x)

        if return_x_t:
            return x, x_t

        return x


def _build_feat_ca(cfg: Optional[Dict], *args, **kwargs):
    if cfg is None:
        cfg_ = dict(type='CBAM_CA')
    cfg_ = cfg.copy()
    fuse_type = cfg_.pop('type')
    if fuse_type == 'CBAM_CA':
        return ChannelAttention(*args, **kwargs, **cfg_)
    return None


def _build_feat_sa(cfg: Optional[Dict], *args, **kwargs):
    if cfg is None:
        cfg_ = dict(type='CBAM_SA')
    cfg_ = cfg.copy()
    fuse_type = cfg_.pop('type')
    if fuse_type == 'CBAM_SA':
        return SpatialAttention(*args, **kwargs, **cfg_)
    return None
