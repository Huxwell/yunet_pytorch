from mmcv.parallel import is_module_wrapper
from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class SetEpochInfoHook(Hook):
    """Set runner's epoch information to the model."""

    def before_train_epoch(self, runner):
        print('Filip YuNet Minify: Function fidx=0 before_train_epoch called in mmdet/core/hook/set_epoch_info_hook.py:L10 ')
        epoch = runner.epoch
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        model.set_epoch(epoch)
