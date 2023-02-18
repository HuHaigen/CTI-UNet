from collections import OrderedDict

import mmcv
import numpy as np
from medpy import metric
from mmcv.utils import print_log
from prettytable import PrettyTable

from mmseg.core import eval_metrics, intersect_and_union, pre_eval_to_metrics
from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class SynapseDataset(CustomDataset):
    """Synapse dataset.
    9 classes
    """

    CLASSES = ('Background', 'Aorta', 'Gallbladder', 'Kidney (L)', 'Kidney (R)',
               'Liver', 'Pancreas', 'Spleen', 'Stomach')

    PALETTE = [[0, 0, 0], [0, 65, 255], [5, 253, 1], [254, 0, 0], [0, 255, 255], [
        255, 32, 255], [255, 249, 5], [63, 208, 244], [241, 240, 234]]

    def __init__(self, eval_hd=False, **kwargs):
        super(SynapseDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            **kwargs)
        self.eval_hd = eval_hd

    def pre_eval(self, preds, indices):
        """Reference the super class
        """
        if not isinstance(indices, list):
            indices = [indices]
        if not isinstance(preds, list):
            preds = [preds]

        pre_eval_results = []

        if self.eval_hd:
            for pred, index in zip(preds, indices):
                seg_map = self.get_gt_seg_map_by_idx(index)
                pre_eval_results.append((
                    intersect_and_union(
                        pred,
                        seg_map,
                        len(self.CLASSES),
                        self.ignore_index,
                        label_map=dict(),
                        reduce_zero_label=self.reduce_zero_label),
                    self._get_hd95(pred, seg_map, len(self.CLASSES))))
        else:
            for pred, index in zip(preds, indices):
                seg_map = self.get_gt_seg_map_by_idx(index)
                pre_eval_results.append(
                    intersect_and_union(
                        pred,
                        seg_map,
                        len(self.CLASSES),
                        self.ignore_index,
                        label_map=dict(),
                        reduce_zero_label=self.reduce_zero_label))

        return pre_eval_results

    def evaluate(self,
                 results,
                 metric='mIoU',
                 logger=None,
                 gt_seg_maps=None,
                 **kwargs):
        """Reference the super class
        """
        if isinstance(metric, str):
            metric = [metric]
        allowed_metrics = ['mIoU', 'mDice', 'mFscore', 'mHD']
        if not set(metric).issubset(set(allowed_metrics)):
            raise KeyError('metric {} is not supported'.format(metric))

        hd_flag = False
        if self.eval_hd and 'mHD' in metric:
            hd_flag = True

            _results, results_dis = [], []
            for x, y in results:
                _results.append(x)
                results_dis.append(y)

            results = _results
        
        if 'mHD' in metric:
            metric.remove('mHD')

        eval_results = {}
        # test a list of files
        if mmcv.is_list_of(results, np.ndarray) or mmcv.is_list_of(
                results, str):
            if gt_seg_maps is None:
                gt_seg_maps = self.get_gt_seg_maps()
            num_classes = len(self.CLASSES)
            ret_metrics = eval_metrics(
                results,
                gt_seg_maps,
                num_classes,
                self.ignore_index,
                metric,
                label_map=dict(),
                reduce_zero_label=self.reduce_zero_label)
        # test a list of pre_eval_results
        else:
            ret_metrics = pre_eval_to_metrics(results, metric)

        # add Hausdorff Distance metric
        if hd_flag:
            results_dis = np.asarray(results_dis)

            tmp = results_dis.copy()
            tmp[tmp != 0] = 1
            cnt_non_zero = tmp.sum(axis=0)

            mean_results_dis = results_dis.sum(axis=0) / cnt_non_zero / 100
            ret_metrics.update({'HD': mean_results_dis})

        # Because dataset.CLASSES is required for per-eval.
        if self.CLASSES is None:
            class_names = tuple(range(num_classes))
        else:
            class_names = self.CLASSES

        # summary table
        ret_metrics_summary = OrderedDict({
            ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })

        # remove the background from Hausdorff Distance metrics
        if hd_flag:
            assert 'HD' in ret_metrics_summary.keys()
            ret_metrics_summary['HD'] = np.round(
                np.nanmean(mean_results_dis[1:]) * 100, 2)

        # each class table
        ret_metrics.pop('aAcc', None)
        ret_metrics_class = OrderedDict({
            ret_metric: np.round(ret_metric_value * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
        ret_metrics_class.update({'Class': class_names})
        ret_metrics_class.move_to_end('Class', last=False)

        # for logger
        class_table_data = PrettyTable()
        for key, val in ret_metrics_class.items():
            class_table_data.add_column(key, val)

        summary_table_data = PrettyTable()
        for key, val in ret_metrics_summary.items():
            if key == 'aAcc':
                summary_table_data.add_column(key, [val])
            else:
                summary_table_data.add_column('m' + key, [val])

        print_log('per class results:', logger)
        print_log('\n' + class_table_data.get_string(), logger=logger)
        print_log('Summary:', logger)
        print_log('\n' + summary_table_data.get_string(), logger=logger)

        # each metric dict
        for key, value in ret_metrics_summary.items():
            if key == 'aAcc':
                eval_results[key] = value / 100.0
            else:
                eval_results['m' + key] = value / 100.0

        ret_metrics_class.pop('Class', None)
        for key, value in ret_metrics_class.items():
            eval_results.update({
                key + '.' + str(name): value[idx] / 100.0
                for idx, name in enumerate(class_names)
            })

        return eval_results

    def _get_hd95(self, pred, label, num_classes):
        """
        Args:
            pred (nparray): the segmentation logit after argmax, shape (N, H, W).
            label (nparray): the ground truth of image, shape (N, H, W).
            num_classes (int): the number of classes
        Returns:
            list[int]: the hd95 of each class
        """
        lst_hd95 = []
        for i in range(num_classes):
            pred_i = pred == i
            label_i = label == i
            if pred_i.sum() > 0 and label_i.sum() > 0:
                lst_hd95.append(metric.binary.hd95(pred_i, label_i))
            else:
                lst_hd95.append(0)
        return lst_hd95
