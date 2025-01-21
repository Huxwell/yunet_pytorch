import json
import os
from datetime import datetime
from mmcv.runner.hooks import HOOKS, Hook


@HOOKS.register_module()
class YuNetSampleSizeStatisticsHook(Hook):

    def __init__(self, out_file, save_interval=50) ->None:
        print('Filip YuNet Minify: Function fidx=0 __init__ called in mmdet/core/hook/yunet_sample_size_statistics_hook.py:L11 ')
        super().__init__()
        self.size_container = {}
        self.shapeless2 = 0
        self.noimg = 0
        self.out_file = out_file
        self.total_sample_num = 0
        self.save_interval = save_interval
        self.batch_size = 0

    def before_run(self, runner):
        print('Filip YuNet Minify: Function fidx=1 before_run called in mmdet/core/hook/yunet_sample_size_statistics_hook.py:L21 ')
        work_dir = runner.work_dir
        self.out_file = os.path.join(work_dir, self.out_file)

    def before_epoch(self, runner):
        print('Filip YuNet Minify: Function fidx=2 before_epoch called in mmdet/core/hook/yunet_sample_size_statistics_hook.py:L25 ')
        self.epoch = runner.epoch
        if (self.epoch + 1) % self.save_interval == 0:
            self.dump_json()

    def before_train_iter(self, runner):
        print('Filip YuNet Minify: Function fidx=3 before_train_iter called in mmdet/core/hook/yunet_sample_size_statistics_hook.py:L30 ')
        gt_bbox_datas = runner.data_batch['gt_bboxes'].data[0]
        self.batch_size = len(gt_bbox_datas)
        for gt_bboxes in gt_bbox_datas:
            if len(gt_bboxes.shape) < 2:
                self.shapeless2 += 1
            elif gt_bboxes.shape[0] == 0:
                self.noimg += 1
            else:
                for gt_bbox in gt_bboxes:
                    w, h = int(gt_bbox[2] - gt_bbox[0]), int(gt_bbox[3] -
                        gt_bbox[1])
                    tag = f'{w},{h}'
                    if self.size_container.get(tag, None) is None:
                        self.size_container[tag] = 1
                    else:
                        self.size_container[tag] += 1
                    self.total_sample_num += 1

    def dump_json(self):
        print('Filip YuNet Minify: Function fidx=4 dump_json called in mmdet/core/hook/yunet_sample_size_statistics_hook.py:L50 ')
        with open(self.out_file, 'w') as f:
            json.dump({'datetime:': str(datetime.now()), 'Batch_size': self
                .batch_size, 'Total_sample': self.total_sample_num, 'Noimg':
                self.noimg, 'Shapeless2': self.shapeless2, 'data': self.
                size_container}, f)
