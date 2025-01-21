from abc import ABCMeta, abstractmethod


class BaseBBoxCoder(metaclass=ABCMeta):
    """Base bounding box coder."""

    def __init__(self, **kwargs):
        print('Filip YuNet Minify: Function fidx=0 __init__ called in mmdet/core/bbox/coder/base_bbox_coder.py:L8 ')

    @abstractmethod
    def encode(self, bboxes, gt_bboxes):
        print('Filip YuNet Minify: Function fidx=1 encode called in mmdet/core/bbox/coder/base_bbox_coder.py:L12 ')
        """Encode deltas between bboxes and ground truth boxes."""

    @abstractmethod
    def decode(self, bboxes, bboxes_pred):
        print('Filip YuNet Minify: Function fidx=2 decode called in mmdet/core/bbox/coder/base_bbox_coder.py:L16 ')
        """Decode the predicted bboxes according to prediction and base
        boxes."""
