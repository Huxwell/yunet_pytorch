from mmcv.utils import Registry, build_from_cfg
PRIOR_GENERATORS = Registry('Generator for anchors and points')
ANCHOR_GENERATORS = PRIOR_GENERATORS


def build_prior_generator(cfg, default_args=None):
    return build_from_cfg(cfg, PRIOR_GENERATORS, default_args)
