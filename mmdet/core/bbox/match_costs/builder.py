from mmcv.utils import Registry, build_from_cfg
MATCH_COST = Registry('Match Cost')


def build_match_cost(cfg, default_args=None):
    print('Filip YuNet Minify: Function fidx=0 build_match_cost called in mmdet/core/bbox/match_costs/builder.py:L7 ')
    """Builder of IoU calculator."""
    return build_from_cfg(cfg, MATCH_COST, default_args)
