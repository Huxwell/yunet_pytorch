from ..builder import BBOX_CODERS
from .base_bbox_coder import BaseBBoxCoder


@BBOX_CODERS.register_module()
class PseudoBBoxCoder(BaseBBoxCoder):
    """Pseudo bounding box coder."""

    def __init__(self, **kwargs):
        print('Filip YuNet Minify: Function fidx=0 __init__ called in mmdet/core/bbox/coder/pseudo_bbox_coder.py:L10 ')
        super(BaseBBoxCoder, self).__init__(**kwargs)

    def encode(self, bboxes, gt_bboxes):
        print('Filip YuNet Minify: Function fidx=1 encode called in mmdet/core/bbox/coder/pseudo_bbox_coder.py:L13 ')
        """torch.Tensor: return the given ``bboxes``"""
        return gt_bboxes

    def decode(self, bboxes, pred_bboxes):
        print('Filip YuNet Minify: Function fidx=2 decode called in mmdet/core/bbox/coder/pseudo_bbox_coder.py:L17 ')
        """torch.Tensor: return the given ``pred_bboxes``"""
        return pred_bboxes
