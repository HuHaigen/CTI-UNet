
import torch.nn as nn
from ..builder import BACKBONES


@BACKBONES.register_module()
class MyModel(nn.Module):

    def __init__(self) -> None:
        super(MyModel, self).__init__()
