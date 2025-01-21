from mmcv.runner.hooks import HOOKS, Hook


@HOOKS.register_module()
class MemoryProfilerHook(Hook):
    """Memory profiler hook recording memory information: virtual memory, swap
    memory and memory of current process.

    Args:
        interval (int): Checking interval (every k iterations).
            Default: 50.
    """

    def __init__(self, interval=50):
        print('Filip YuNet Minify: Function fidx=0 __init__ called in mmdet/core/hook/memory_profiler_hook.py:L15 ')
        try:
            from psutil import swap_memory, virtual_memory
            self._swap_memory = swap_memory
            self._virtual_memory = virtual_memory
        except ImportError:
            raise ImportError(
                'psutil is not installed, please install it by: pip install psutil'
                )
        try:
            from memory_profiler import memory_usage
            self._memory_usage = memory_usage
        except ImportError:
            raise ImportError(
                'memory_profiler is not installed, please install it by: pip install memory_profiler'
                )
        self.interval = interval

    def after_iter(self, runner):
        print('Filip YuNet Minify: Function fidx=1 after_iter called in mmdet/core/hook/memory_profiler_hook.py:L34 ')
        if self.every_n_iters(runner, self.interval):
            virtual_memory = self._virtual_memory()
            swap_memory = self._swap_memory()
            process_memory = self._memory_usage()[0]
            factor = 1024 * 1024
            runner.logger.info(
                f'Memory information available_memory: {round(virtual_memory.available / factor)} MB, used_memory: {round(virtual_memory.used / factor)} MB, memory_utilization: {virtual_memory.percent} %, available_swap_memory: {round((swap_memory.total - swap_memory.used) / factor)}MB, used_swap_memory: {round(swap_memory.used / factor)} MB, swap_memory_utilization: {swap_memory.percent} %, current_process_memory: {round(process_memory)} MB'
                )
