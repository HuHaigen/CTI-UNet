from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class ACDCDataset(CustomDataset):
    """ACDC dataset.
    4 classes
    """

    CLASSES = ('Background', 'RV', 'MYO', 'LV')

    PALETTE = [[0, 0, 0], [255, 0, 0], [0, 0, 255], [0, 255, 0]]

    def __init__(self, **kwargs):
        super(ACDCDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            **kwargs)
