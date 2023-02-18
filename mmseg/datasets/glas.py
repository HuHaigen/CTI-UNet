

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class GlasDataset(CustomDataset):
    """Glas dataset.
    2 classes
    """

    CLASSES = ('background', 'adenocarcinoma')

    PALETTE = [[0, 0, 0], [255, 255, 255]]

    def __init__(self, **kwargs):
        super(GlasDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            **kwargs)
