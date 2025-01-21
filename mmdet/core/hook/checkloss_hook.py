import torch
from mmcv.runner.hooks import HOOKS, Hook


@HOOKS.register_module()
class CheckInvalidLossHook(Hook):
    """Check invalid loss hook.

    This hook will regularly check whether the loss is valid
    during training.

    Args:
        interval (int): Checking interval (every k iterations).
            Default: 50.
    """

    def __init__(self, interval=50):
        print('Filip YuNet Minify: Function fidx=0 __init__ called in mmdet/core/hook/checkloss_hook.py:L18 ')
        self.interval = interval

    def after_train_iter(self, runner):
        print('Filip YuNet Minify: Function fidx=1 after_train_iter called in mmdet/core/hook/checkloss_hook.py:L21 ')
        if self.every_n_iters(runner, self.interval):
            assert torch.isfinite(runner.outputs['loss']), runner.logger.info(
                'loss become infinite or NaN!')
