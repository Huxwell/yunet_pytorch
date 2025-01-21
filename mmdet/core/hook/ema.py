import math
from mmcv.parallel import is_module_wrapper
from mmcv.runner.hooks import HOOKS, Hook


class BaseEMAHook(Hook):
    """Exponential Moving Average Hook.

    Use Exponential Moving Average on all parameters of model in training
    process. All parameters have a ema backup, which update by the formula
    as below. EMAHook takes priority over EvalHook and CheckpointHook. Note,
    the original model parameters are actually saved in ema field after train.

    Args:
        momentum (float): The momentum used for updating ema parameter.
            Ema's parameter are updated with the formula:
           `ema_param = (1-momentum) * ema_param + momentum * cur_param`.
            Defaults to 0.0002.
        skip_buffers (bool): Whether to skip the model buffers, such as
            batchnorm running stats (running_mean, running_var), it does not
            perform the ema operation. Default to False.
        interval (int): Update ema parameter every interval iteration.
            Defaults to 1.
        resume_from (str, optional): The checkpoint path. Defaults to None.
        momentum_fun (func, optional): The function to change momentum
            during early iteration (also warmup) to help early training.
            It uses `momentum` as a constant. Defaults to None.
    """

    def __init__(self, momentum=0.0002, interval=1, skip_buffers=False,
        resume_from=None, momentum_fun=None):
        print('Filip YuNet Minify: Function fidx=0 __init__ called in mmdet/core/hook/ema.py:L32 ')
        assert 0 < momentum < 1
        self.momentum = momentum
        self.skip_buffers = skip_buffers
        self.interval = interval
        self.checkpoint = resume_from
        self.momentum_fun = momentum_fun

    def before_run(self, runner):
        print('Filip YuNet Minify: Function fidx=1 before_run called in mmdet/core/hook/ema.py:L45 ')
        """To resume model with it's ema parameters more friendly.

        Register ema parameter as ``named_buffer`` to model.
        """
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        self.param_ema_buffer = {}
        if self.skip_buffers:
            self.model_parameters = dict(model.named_parameters())
        else:
            self.model_parameters = model.state_dict()
        for name, value in self.model_parameters.items():
            buffer_name = f"ema_{name.replace('.', '_')}"
            self.param_ema_buffer[name] = buffer_name
            model.register_buffer(buffer_name, value.data.clone())
        self.model_buffers = dict(model.named_buffers())
        if self.checkpoint is not None:
            runner.resume(self.checkpoint)

    def get_momentum(self, runner):
        print('Filip YuNet Minify: Function fidx=2 get_momentum called in mmdet/core/hook/ema.py:L67 ')
        return self.momentum_fun(runner.iter
            ) if self.momentum_fun else self.momentum

    def after_train_iter(self, runner):
        print('Filip YuNet Minify: Function fidx=3 after_train_iter called in mmdet/core/hook/ema.py:L71 ')
        """Update ema parameter every self.interval iterations."""
        if (runner.iter + 1) % self.interval != 0:
            return
        momentum = self.get_momentum(runner)
        for name, parameter in self.model_parameters.items():
            if parameter.dtype.is_floating_point:
                buffer_name = self.param_ema_buffer[name]
                buffer_parameter = self.model_buffers[buffer_name]
                buffer_parameter.mul_(1 - momentum).add_(parameter.data,
                    alpha=momentum)

    def after_train_epoch(self, runner):
        print('Filip YuNet Minify: Function fidx=4 after_train_epoch called in mmdet/core/hook/ema.py:L84 ')
        """We load parameter values from ema backup to model before the
        EvalHook."""
        self._swap_ema_parameters()

    def before_train_epoch(self, runner):
        print('Filip YuNet Minify: Function fidx=5 before_train_epoch called in mmdet/core/hook/ema.py:L89 ')
        """We recover model's parameter from ema backup after last epoch's
        EvalHook."""
        self._swap_ema_parameters()

    def _swap_ema_parameters(self):
        print('Filip YuNet Minify: Function fidx=6 _swap_ema_parameters called in mmdet/core/hook/ema.py:L94 ')
        """Swap the parameter of model with parameter in ema_buffer."""
        for name, value in self.model_parameters.items():
            temp = value.data.clone()
            ema_buffer = self.model_buffers[self.param_ema_buffer[name]]
            value.data.copy_(ema_buffer.data)
            ema_buffer.data.copy_(temp)


@HOOKS.register_module()
class ExpMomentumEMAHook(BaseEMAHook):
    """EMAHook using exponential momentum strategy.

    Args:
        total_iter (int): The total number of iterations of EMA momentum.
           Defaults to 2000.
    """

    def __init__(self, total_iter=2000, **kwargs):
        print('Filip YuNet Minify: Function fidx=7 __init__ called in mmdet/core/hook/ema.py:L112 ')
        super(ExpMomentumEMAHook, self).__init__(**kwargs)
        self.momentum_fun = lambda x: (1 - self.momentum) * math.exp(-(1 +
            x) / total_iter) + self.momentum


@HOOKS.register_module()
class LinearMomentumEMAHook(BaseEMAHook):
    """EMAHook using linear momentum strategy.

    Args:
        warm_up (int): During first warm_up steps, we may use smaller decay
            to update ema parameters more slowly. Defaults to 100.
    """

    def __init__(self, warm_up=100, **kwargs):
        print('Filip YuNet Minify: Function fidx=8 __init__ called in mmdet/core/hook/ema.py:L127 ')
        super(LinearMomentumEMAHook, self).__init__(**kwargs)
        self.momentum_fun = lambda x: min(self.momentum ** self.interval, (
            1 + x) / (warm_up + x))
