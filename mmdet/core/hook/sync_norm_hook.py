from collections import OrderedDict
from mmcv.runner import get_dist_info
from mmcv.runner.hooks import HOOKS, Hook
from torch import nn
from ..utils.dist_utils import all_reduce_dict


def get_norm_states(module):
    print('Filip YuNet Minify: Function fidx=0 get_norm_states called in mmdet/core/hook/sync_norm_hook.py:L11 ')
    async_norm_states = OrderedDict()
    for name, child in module.named_modules():
        if isinstance(child, nn.modules.batchnorm._NormBase):
            for k, v in child.state_dict().items():
                async_norm_states['.'.join([name, k])] = v
    return async_norm_states


@HOOKS.register_module()
class SyncNormHook(Hook):
    """Synchronize Norm states after training epoch, currently used in YOLOX.

    Args:
        num_last_epochs (int): The number of latter epochs in the end of the
            training to switch to synchronizing norm interval. Default: 15.
        interval (int): Synchronizing norm interval. Default: 1.
    """

    def __init__(self, num_last_epochs=15, interval=1):
        print('Filip YuNet Minify: Function fidx=1 __init__ called in mmdet/core/hook/sync_norm_hook.py:L30 ')
        self.interval = interval
        self.num_last_epochs = num_last_epochs

    def before_train_epoch(self, runner):
        print('Filip YuNet Minify: Function fidx=2 before_train_epoch called in mmdet/core/hook/sync_norm_hook.py:L34 ')
        epoch = runner.epoch
        if epoch + 1 == runner.max_epochs - self.num_last_epochs:
            self.interval = 1

    def after_train_epoch(self, runner):
        print('Filip YuNet Minify: Function fidx=3 after_train_epoch called in mmdet/core/hook/sync_norm_hook.py:L40 ')
        """Synchronizing norm."""
        epoch = runner.epoch
        module = runner.model
        if (epoch + 1) % self.interval == 0:
            _, world_size = get_dist_info()
            if world_size == 1:
                return
            norm_states = get_norm_states(module)
            if len(norm_states) == 0:
                return
            norm_states = all_reduce_dict(norm_states, op='mean')
            module.load_state_dict(norm_states, strict=False)
