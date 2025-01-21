from abc import ABCMeta
from mmcv.runner import BaseModule


class BaseDenseHead(BaseModule, metaclass=ABCMeta):
    """Base class for DenseHeads."""

    def __init__(self, init_cfg=None):
        super(BaseDenseHead, self).__init__(init_cfg)
