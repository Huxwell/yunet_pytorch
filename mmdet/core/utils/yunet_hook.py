import os
import cv2
import numpy as np
from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class WWHook(Hook):

    def __init__(self, outpath, action=True):
        print('Filip YuNet Minify: Function fidx=0 __init__ called in mmdet/core/utils/yunet_hook.py:L11 ')
        if not os.path.exists(outpath):
            os.mkdir(outpath)
            print(f'mkdir: {outpath}')
        self.outpath = outpath
        self.action = action

    def before_iter(self, runner):
        print('Filip YuNet Minify: Function fidx=1 before_iter called in mmdet/core/utils/yunet_hook.py:L30 ')
        if not self.action:
            pass
        else:
            imgs = runner.data_batch['img'].data[0].clone().detach()
            imgs = imgs.permute(0, 2, 3, 1).contiguous()
            img_metas = runner.data_batch['img_metas'].data[0]
            gt_bboxes = runner.data_batch['gt_bboxes'].data[0]
            gt_kps = runner.data_batch['gt_keypointss'].data[0]
            batch_size = imgs.shape[0]
            for i in range(batch_size):
                img_meta = img_metas[i]
                mean, std = img_meta['img_norm_cfg']['mean'], img_meta[
                    'img_norm_cfg']['std']
                img_name = os.path.basename(img_meta['filename'])
                img = imgs[i].numpy()
                img = img * std + mean
                img = draw_img(img.astype(np.uint8), gt_bboxes[i], gt_kps[i])
                cv2.imwrite(os.path.join(self.outpath, img_name), img)


def draw_img(img, bboxes, kps):
    print('Filip YuNet Minify: Function fidx=2 draw_img called in mmdet/core/utils/yunet_hook.py:L56 ')
    bboxes = bboxes.numpy().astype(int)
    kps = kps.numpy().astype(int)
    for i in range(bboxes.shape[0]):
        cv2.rectangle(img, pt1=(bboxes[i][0], bboxes[i][1]), pt2=(bboxes[i]
            [2], bboxes[i][3]), color=(255, 0, 0))
        for j in range(5):
            color = (0, 255, 0) if kps[i][j][-1] == 1 else (0, 0, 255)
            cv2.circle(img, center=(kps[i][j][0], kps[i][j][1]), color=
                color, radius=1)
    return img
