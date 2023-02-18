

from .builder import DATASETS
from .custom import CustomDataset
from .pipelines import LoadAnnotationsFromNumpy


@DATASETS.register_module()
class SynapseDatasetNP(CustomDataset):
    """Synapse dataset.
    9 classes
    """

    CLASSES = ('Background', 'Aorta', 'Gallbladder', 'Kidney (L)', 'Kidney (R)',
               'Liver', 'Pancreas', 'Spleen', 'Stomach')

    PALETTE = [[0, 0, 0], [0, 65, 255], [5, 253, 1], [254, 0, 0], [0, 255, 255], [
        255, 32, 255], [255, 249, 5], [63, 208, 244], [241, 240, 234]]

    def __init__(self, **kwargs):
        super(SynapseDatasetNP, self).__init__(
            img_suffix='.npy',
            seg_map_suffix='.npy',
            **kwargs)
        
        self.gt_seg_map_loader = LoadAnnotationsFromNumpy()
