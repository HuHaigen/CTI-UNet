import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.utils import to_2tuple

from ..backbones.swinunet import FinalPatchExpand_X4
from ..builder import HEADS
from .decode_head import BaseDecodeHead


@HEADS.register_module()
class SwinUNetHead(BaseDecodeHead):

    def __init__(self,
                 img_size=224,
                 patch_size=4,
                 dim_scale=4,
                 embed_dims=96,
                 num_convs=0,
                 kernel_size=1,
                 dilation=1,
                 **kwargs):

        assert isinstance(img_size, (int, tuple))
        assert isinstance(patch_size, (int, tuple))

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        assert img_size[0] % patch_size[0] == 0 and \
            img_size[1] % patch_size[1] == 0

        super(SwinUNetHead, self).__init__(**kwargs)

        if num_convs == 0:
            assert self.channels == self.in_channels

        self.patches_resolution = [img_size[0] //
                                   patch_size[0], img_size[1] // patch_size[1]]

        self.up = FinalPatchExpand_X4(input_resolution=self.patches_resolution,
                                      dim_scale=dim_scale, dim=embed_dims)

        conv_padding = (kernel_size // 2) * dilation
        convs = []
        for i in range(num_convs):
            _in_channels = self.in_channels if i == 0 else self.channels
            convs.append(
                ConvModule(
                    _in_channels,
                    self.channels,
                    kernel_size=kernel_size,
                    padding=conv_padding,
                    dilation=dilation,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        if len(convs) == 0:
            self.convs = nn.Identity()
        else:
            self.convs = nn.Sequential(*convs)

    def _forward_feature(self, inputs):
        """Forward function.
        """

        H, W = self.patches_resolution
        B, L, _ = inputs.shape
        assert L == H*W, "input features has wrong size"

        x = self.up(inputs)
        x = x.view(B, 4*H, 4*W, -1)
        x = x.permute(0, 3, 1, 2)

        return x

    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        output = self.cls_seg(output)
        return output
