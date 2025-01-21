from abc import ABCMeta, abstractmethod


class BaseAssigner(metaclass=ABCMeta):
    """Base assigner that assigns boxes to ground truth boxes."""

    @abstractmethod
    def assign(self, bboxes, gt_bboxes, gt_bboxes_ignore=None, gt_labels=None):
        print('Filip YuNet Minify: Function fidx=0 assign called in mmdet/core/bbox/assigners/base_assigner.py:L9 ')
        """Assign boxes to either a ground truth boxes or a negative boxes."""
