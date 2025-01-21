from mmcv.utils import Registry, build_from_cfg
IOU_CALCULATORS = Registry('IoU calculator')


def build_iou_calculator(cfg, default_args=None):
    print('Filip YuNet Minify: Function fidx=0 build_iou_calculator called in mmdet/core/bbox/iou_calculators/builder.py:L7 ')
    """Builder of IoU calculator."""
    return build_from_cfg(cfg, IOU_CALCULATORS, default_args)
