from mmseg.ops import resize

from ..builder import HEADS
from .setr_up_head import SETRUPHead


@HEADS.register_module()
class CTIPUPHead(SETRUPHead):
    
    def __init__(self, **kwargs):
        super(CTIPUPHead, self).__init__(**kwargs)
        
    def forward(self, x):
        x = x[self.in_index]

        n, hw_shape, dim = x.shape
        x = self.norm(x)
        x = x.transpose(1, 2).reshape(n, dim, int(hw_shape**0.5), -1).contiguous()

        for up_conv in self.up_convs:
            x = up_conv(x)
        out = self.cls_seg(x)
        return out

