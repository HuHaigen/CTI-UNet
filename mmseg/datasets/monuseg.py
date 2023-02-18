

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class MoNuSegDataset(CustomDataset):
    """MoNuSeg dataset.
    2 classes
    """

    CLASSES = ('background', 'nuclei')

    PALETTE = [[0, 0, 0], [255, 255, 255]]

    def __init__(self, **kwargs):
        super(MoNuSegDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            **kwargs)
