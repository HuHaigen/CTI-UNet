
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks import (ConvModule, build_activation_layer,
                             build_conv_layer)
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmcv.runner import ModuleList
from mmcv.utils import Registry

from mmseg.ops import resize
from ..backbones.vit import TransformerEncoderLayer
from ..builder import HEADS, MODELS
from ..utils import PatchEmbed
from .decode_head import BaseDecodeHead
from .fcn_head import FCNHead

STRUCT_TOKEN = Registry('struct_token')


@HEADS.register_module()
class StructTokenHead(BaseDecodeHead):
    """StructToken

    Args:
        inter_module (str): interaction modules
        num_layers (int): Number of StructToken blocks.
        embed_dims (int): Number of channels of feature map, which
            is also the dimension of tokens in Transformer.
        h_s (int): Height of StructTokens.
        w_s (int): Width of StructTokens.
        feedforward_channels (int): The hidden dimension of FFNs, i.e. HW in the paper.
        norm_cfg (dict): Config dict for normalization layer.
        patch_norm (bool): Whether to apply normalization.
        inter_module_cfg (dict): The config of interaction module.

        # conv_block_cfg (dict): None.
    """

    def __init__(self,
                 embed_dims,
                 struct_token_size=(14, 14),
                 hidden_channels=256,
                 decoder_cfg=None,
                 **kwargs):

        super(StructTokenHead, self).__init__(**kwargs)

        self.patch_embed = PatchEmbed(
            in_channels=self.in_channels,
            embed_dims=embed_dims,
            conv_type='Conv2d',
            kernel_size=1,
            stride=1,
            padding='corner',
            norm_cfg=dict(type='LN')
        )

        # define the struct token corresponding the number of classes
        self.struct_tokens = nn.Parameter(
            torch.zeros(1, self.out_channels, *struct_token_size))

        if 'embed_dims' not in decoder_cfg:
            decoder_cfg['embed_dims'] = embed_dims
        self.st_decoder = StructTokenDecoder(**decoder_cfg)

        self.conv_seg = ConvBlock(
            in_channels=self.channels,
            hidden_channels=hidden_channels,
            out_channels=self.out_channels
        )

    def forward(self, inputs):

        assert isinstance(inputs, (list, tuple, torch.Tensor))

        if isinstance(inputs, (list, tuple)):
            inputs = inputs[0]

        B, _, H, W = inputs.shape  # (b,1024,H,W)

        # obtain the last feature map, (B,in_channels,H,W)
        x, hw_shape = self.patch_embed(inputs)
        x = x.transpose(1, 2)  # (B,C,HW), C is embed_dims

        # (B,K,HW), K is num_classes (i.e. out_channels)
        struct_tokens = resize(
            self.struct_tokens.expand(
                B, -1, -1, -1),
            size=inputs.shape[-2:],
            mode='bilinear'
        ).view(B, self.out_channels, -1)

        struct_tokens = self.st_decoder(x, struct_tokens)\
            .view(B, self.out_channels, H, W)

        # conv_seg is ConvBlock
        output = self.cls_seg(struct_tokens)

        return output


class StructTokenDecoder(nn.Module):
    """
    Args:
        embed_dims (int): Dimension of tokens (i.e. HW in the paper).
        num_layers (int): Number of blocks in Decoder.
        ffn_feat_cfg  (dice): Config of FFN of feature map.
        ffn_st_cfg (dice): Config of FFN of StructToken.
        inter_module_cfg (dice): Config of interaction module.
    """

    def __init__(self,
                 embed_dims,
                 num_layers,
                 ffn_feat_cfg,
                 ffn_st_cfg,
                 inter_module_cfg):
        super(StructTokenDecoder, self).__init__()

        if 'embed_dims' not in ffn_feat_cfg:
            ffn_feat_cfg['embed_dims'] = embed_dims
        if 'embed_dims' not in ffn_st_cfg:
            ffn_st_cfg['embed_dims'] = embed_dims
        if 'embed_dims' not in inter_module_cfg:
            inter_module_cfg['embed_dims'] = embed_dims

        self.layers = ModuleList()
        for i in range(num_layers):
            self.layers.append(
                StructBlock(
                    ffn_feat_cfg=ffn_feat_cfg,
                    ffn_st_cfg=ffn_st_cfg,
                    inter_module_cfg=inter_module_cfg,
                    final_block=(i == num_layers - 1)
                )
            )

    def forward(self, feat, struct_tokens):
        for i, layer in enumerate(self.layers):
            if i == len(self.layers) - 1:
                break
            feat, struct_tokens = layer(feat, struct_tokens)
        return self.layers[-1](feat, struct_tokens)


class StructBlock(nn.Module):
    """
    Args:
        ffn_feat_cfg (dict): Config of FFN of feature map.
        ffn_st_cfg (dict): Config of FFN of StructToken.
            - in_channels (int): in_channels, the dim of the Transformer tokens.
            - feedforward_channels (int): The hidden dimension of FFNs.
            - drop_rate (float): Probability of an element to be zeroed.
            - drop_path_rate (float): stochastic depth rate.
            - act_cfg (dict): Config of activation. Default dict(type='GELU')
        inter_module_cfg (dict): Config of interaction module.
        final_block (bool)
    """

    def __init__(self,
                 ffn_feat_cfg=None,
                 ffn_st_cfg=None,
                 inter_module_cfg=None,
                 final_block=False):
        super(StructBlock, self).__init__()

        self.inter_module = self._build_interaction_module(inter_module_cfg)

        self.final_block = final_block

        if not self.final_block:
            self.ffn_feat = FFN(**ffn_feat_cfg)

        self.ffn_st = FFN(**ffn_st_cfg)

    def forward(self, feat, struct_tokens):
        feat, struct_tokens = self.inter_module(feat, struct_tokens)

        # If this is the final block, the only struct_tokens would be returned.
        struct_tokens = self.ffn_st(struct_tokens)

        if self.final_block:
            return struct_tokens

        feat = self.ffn_feat(feat)
        return feat, struct_tokens

    def _build_interaction_module(self, cfg, *args, **kwargs):
        if cfg is None:
            cfg = dict(type='CSE')
        else:
            if not isinstance(cfg, dict):
                raise TypeError('cfg must be a dict')
            if 'type' not in cfg:
                raise KeyError('the cfg dict must contain the key "type"')
            cfg_ = cfg.copy()
        inter_type = cfg_.pop('type')
        if inter_type not in STRUCT_TOKEN:
            raise KeyError(
                f'Unrecognized interaction module type {inter_type}')
        else:
            inter_module_type = STRUCT_TOKEN.get(inter_type)
        inter_module = inter_module_type(*args, **kwargs, **cfg_)
        return inter_module


@STRUCT_TOKEN.register_module('CSE')
class CSE(nn.Module):
    """
    Args:
        embed_dims (int): The dim of token inputted into MultiHeadAttention, 
            set as HW of feature map.

    """

    def __init__(self,
                 embed_dims=196,
                 num_heads=6,
                 **kwargs):
        super(CSE, self).__init__()
        self.attn = MultiheadAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            batch_first=True,
            **kwargs
        )

    def forward(self, feat, struct_tokens):
        """
            feat: (B,C,HW)
            struct_tokens: (B,K,HW)
        """
        output_st = self.attn(
            query=struct_tokens,
            key=feat,
            value=feat)  # (B,K,HW)
        return feat, output_st


@STRUCT_TOKEN.register_module('SSE')
class SSE(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(input):
        pass


@STRUCT_TOKEN.register_module('PWE')
class PWE(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(input):
        pass


class ConvBlock(nn.Module):
    """
    Args:
        in_channels (int): Number of channels of input(StructToken), equal to the num_classes.
        out_channels (int): Number of channels of output(StructToken), equal to the num_classes.
        hidden_channels (int): Number of hidden channels. Default 256.
        act_cfg (dict): Config of activation function. Default dict(type='ReLU').
        drop_cfg (int): Config of dropout. Whether to apply dropout after the first convolution. Default 0.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels=256,
                 act_cfg=dict(type='ReLU'),
                 drop_cfg=dict(type='Dropout', drop_prob=0.)):
        super().__init__()

        self.conv1 = ConvModule(
            in_channels, hidden_channels, 3, 1, act_cfg=act_cfg, padding=1)
        self.drop = build_dropout(drop_cfg)
        self.conv2 = ConvModule(
            hidden_channels, out_channels, 3, 1, act_cfg=None, padding=1)

    def forward(self, x, identity=None):
        if identity is None:
            identity = x
        x = self.conv2(self.drop(self.conv1(x)))
        return identity + x
