import warnings
import pycocotools
from pycocotools.coco import COCO as _COCO
from pycocotools.cocoeval import COCOeval as _COCOeval


class COCO(_COCO):
    """This class is almost the same as official pycocotools package.

    It implements some snake case function aliases. So that the COCO class has
    the same interface as LVIS class.
    """

    def __init__(self, annotation_file=None):
        print('Filip YuNet Minify: Function fidx=0 __init__ called in mmdet/datasets/api_wrappers/coco_api.py:L18 ')
        if getattr(pycocotools, '__version__', '0') >= '12.0.2':
            warnings.warn(
                'mmpycocotools is deprecated. Please install official pycocotools by "pip install pycocotools"'
                , UserWarning)
        super().__init__(annotation_file=annotation_file)
        self.img_ann_map = self.imgToAnns
        self.cat_img_map = self.catToImgs

    def get_ann_ids(self, img_ids=[], cat_ids=[], area_rng=[], iscrowd=None):
        print('Filip YuNet Minify: Function fidx=1 get_ann_ids called in mmdet/datasets/api_wrappers/coco_api.py:L27 ')
        return self.getAnnIds(img_ids, cat_ids, area_rng, iscrowd)

    def get_cat_ids(self, cat_names=[], sup_names=[], cat_ids=[]):
        print('Filip YuNet Minify: Function fidx=2 get_cat_ids called in mmdet/datasets/api_wrappers/coco_api.py:L30 ')
        return self.getCatIds(cat_names, sup_names, cat_ids)

    def get_img_ids(self, img_ids=[], cat_ids=[]):
        print('Filip YuNet Minify: Function fidx=3 get_img_ids called in mmdet/datasets/api_wrappers/coco_api.py:L33 ')
        return self.getImgIds(img_ids, cat_ids)

    def load_anns(self, ids):
        print('Filip YuNet Minify: Function fidx=4 load_anns called in mmdet/datasets/api_wrappers/coco_api.py:L36 ')
        return self.loadAnns(ids)

    def load_cats(self, ids):
        print('Filip YuNet Minify: Function fidx=5 load_cats called in mmdet/datasets/api_wrappers/coco_api.py:L39 ')
        return self.loadCats(ids)

    def load_imgs(self, ids):
        print('Filip YuNet Minify: Function fidx=6 load_imgs called in mmdet/datasets/api_wrappers/coco_api.py:L42 ')
        return self.loadImgs(ids)


COCOeval = _COCOeval
