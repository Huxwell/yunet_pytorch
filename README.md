# Training for libfacedetection in PyTorch

[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

It is the training program for [libfacedetection](https://github.com/ShiqiYu/libfacedetection). The source code is based on [MMDetection](https://github.com/open-mmlab/mmdetection). Some data processing functions from [SCRFD](https://github.com/deepinsight/insightface/tree/master/detection/scrfd/mmdet) modifications.

Visualization of our network architecture: [\[netron\]](https://netron.app/?url=https://raw.githubusercontent.com/ShiqiYu/libfacedetection.train/master/onnx/yunet_n_320_320.onnx).


### Contents

- [Training for libfacedetection in PyTorch](#training-for-libfacedetection-in-pytorch)
    - [Contents](#contents)
  - [Installation](#installation)
  - [Preparation](#preparation)
  - [Training](#training)
  - [Detection](#detection)
  - [Evaluation on WIDER Face](#evaluation-on-wider-face)
  - [Export CPP source code](#export-cpp-source-code)
  - [Export to onnx model](#export-to-onnx-model)
  - [Compare ONNX model with other works](#compare-onnx-model-with-other-works)
  - [Citation](#citation)

## Installation

1. Create conda environment. e.g.
   ```shell
   conda create -n yunet python=3.8
   conda activate yunet
   ```
1. Install [PyTorch](https://pytorch.org/) == v1.8.2 (LTS) following official instruction. e.g.\
   On GPU platforms (cu11.1):
   ```shell
   # LINUX:
   conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c nvidia
   # WINDOWS:
   conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c conda-forge
   ```
   On GPU platforms (cu10.2):
   ```shell
   conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch-lts
   ```
2. Install [MMCV](https://github.com/open-mmlab/mmcv) >= v1.3.17 but  <=1.6.0 following official instruction. e.g.
   ```shell
   # cu11.1
   pip install mmcv-full==1.3.17 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.0/index.html
   # cu10.2
   pip install mmcv-full==1.3.17 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.8.0/index.html
   ```
3. Clone this repository. We will call the cloned directory as `$TRAIN_ROOT`.
   ```Shell
   git clone https://github.com/ShiqiYu/libfacedetection.train.git
   cd libfacedetection.train
   ```
4. Install dependencies.
   ```shell
   python setup.py develop
   pip install -r requirements.txt
   ```

_Note_: 
   1. Codes are based on Python 3+.
   2. If meet error "ModuleNotFoundError: No module named 'torch.ao'", you can `Ctrl + click` to origin line and replace `torch.ao` to `torch`


## Preparation

1. Download the [WIDER Face](http://shuoyang1213.me/WIDERFACE/) dataset and its [evaluation tools](http://shuoyang1213.me/WIDERFACE/support/eval_script/eval_tools.zip).
2. Extract zip files under `$TRAIN_ROOT/data/widerface` as follows:
   ```shell
   $ tree data/widerface
   data/widerface
   ├── wider_face_split
   ├── WIDER_test
   ├── WIDER_train
   ├── WIDER_val
   └── labelv2
         ├── train
         │   └── labelv2.txt
         └── val
             ├── gt
             └── labelv2.txt
   ```

_NOTE: \
The `labelv2` comes from [SCRFD](https://github.com/deepinsight/insightface/tree/master/detection/scrfd)._

## Training

Following MMdetection training processing.

```Shell
python tools/train.py
```

## Detection

```Shell
python tools/detect_image.py ./configs/yunet_n.py ./weights/yunet_n.pth ./image.jpg
```

## Evaluation on WIDER Face

```shell
python tools/test_widerface.py ./configs/yunet_n.py ./weights/yunet_n.pth --mode 2
```

Performance on WIDER Face (Val): confidence_threshold=0.02, nms_threshold=0.45, in origin size:

```
AP_easy=0.892, AP_medium=0.883, AP_hard=0.811
```

## Export CPP source code

The following bash code can export a CPP file for project [libfacedetection](https://github.com/ShiqiYu/libfacedetection)

```Shell
python tools/yunet2cpp.py ./configs/yunet_n.py ./weights/yunet_n.pth
```

## Export to onnx model

Export to onnx model for [libfacedetection/example/opencv_dnn](https://github.com/ShiqiYu/libfacedetection/tree/master/example/opencv_dnn).

```shell
python tools/yunet2onnx.py ./configs/yunet_n.py ./weights/yunet_n.pth
```

## Compare ONNX model with other works

Inference on exported ONNX models using ONNXRuntime:

```shell
python tools/compare_inference.py ./onnx/yunet_n.onnx --mode AUTO --eval --score_thresh 0.02 --nms_thresh 0.45
```

Some similar approaches(e.g. SCRFD, Yolo5face, retinaface) to inference are also supported.

With Intel i7-12700K and `input_size = origin size, score_thresh = 0.02, nms_thresh = 0.45`, some results are list as follow:

| Model                   | AP_easy | AP_medium | AP_hard | #Params | Params Ratio | MFlops (320x320) | FPS(320x320) |
| ----------------------- | ------- | --------- | ------- | ------- | ------------ | ---------------- | ------------ |
| SCRFD0.5(ICLR2022)      | 0.892   | 0.885     | 0.819   | 631,410 |     8.32x    |      184         |     284      |
| Retinaface0.5(CVPR2020) | 0.907   | 0.883     | 0.742   | 426,608 |     5.62X    |      245         |     235      |
| YuNet_n(Ours)           | 0.892   | 0.883     | 0.811   | 75,856  |     1.00x    |      149         |     456      |
| YuNet_s(Ours)           | 0.887   | 0.871     | 0.768   | 54,608  |     0.72x    |      96          |     537      |

The compared models can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1PmnX0LPkQxGali2dvRqABr0VnE8OJ7FA?usp=sharing).

## Citation
We published a paper for the main idea of this repository:

```
@article{yunet,
  title={YuNet: A Tiny Millisecond-level Face Detector},
  author={Wu, Wei and Peng, Hanyang and Yu, Shiqi},
  journal={Machine Intelligence Research},
  pages={1--10},
  year={2023},
  doi={10.1007/s11633-023-1423-y},
  publisher={Springer}
}
```
The paper can be open accessed at https://link.springer.com/article/10.1007/s11633-023-1423-y.

The loss used in training is EIoU, a novel extended IoU. More details can be found in:

```
@article{eiou,
 author={Peng, Hanyang and Yu, Shiqi},
 journal={IEEE Transactions on Image Processing},
 title={A Systematic IoU-Related Method: Beyond Simplified Regression for Better Localization},
 year={2021},
 volume={30},
 pages={5032-5044},
 doi={10.1109/TIP.2021.3077144}
}
```

The paper can be open accessed at https://ieeexplore.ieee.org/document/9429909.

We also published a paper on face detection to evaluate different methods.

```
@article{facedetect-yu,
  author={Feng, Yuantao and Yu, Shiqi and Peng, Hanyang and Li, Yan-Ran and Zhang, Jianguo},
  journal={IEEE Transactions on Biometrics, Behavior, and Identity Science}, 
  title={Detect Faces Efficiently: A Survey and Evaluations}, 
  year={2022},
  volume={4},
  number={1},
  pages={1-18},
  doi={10.1109/TBIOM.2021.3120412}
}
```

The paper can be open accessed at https://ieeexplore.ieee.org/document/9580485
