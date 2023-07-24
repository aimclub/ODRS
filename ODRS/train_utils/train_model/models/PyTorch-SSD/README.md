# PyTorch SSD
PyTorch implementation of [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325).

## Results
### PASCAL VOC
* Training: 07+12 trainval
* Evaluation: 07 test

| Model                | Input size | mAP<sub>0.5</sub> | Configuration                                                                |
|----------------------|:----------:|:-----------------:|------------------------------------------------------------------------------|
| SSD300               | 300        | 77.1              | [configs/voc/ssd300.yaml](configs/voc/ssd300.yaml)                           |
| SSD512               | 512        | 79.4              | [configs/voc/ssd512.yaml](configs/voc/ssd512.yaml)                           |
| MobileNetV2 SSDLite  | 320        | 70.7              | [configs/voc/mobilenetV2_ssdlite.yaml](configs/voc/mobilenetV2_ssdlite.yaml) |

### COCO
* Training: train2017
* Evaluation: val2017

| Model                | Input size | mAP<sub>0.5:0.95</sub> | Configuration                                        |
|----------------------|:----------:|:----------------------:|------------------------------------------------------|
| SSD300               | 300        | 25.3                   | [configs/coco/ssd300.yaml](configs/coco/ssd300.yaml) |
| SSD512               | 512        | 29.4                   | [configs/coco/ssd512.yaml](configs/coco/ssd512.yaml) |

**Note**: We run `coco_eval.py` to obtain the COCO mAP scores, as described in [Evaluation](#evaluation) section.
In `coco_eval.py`, [pycocotools](https://github.com/cocodataset/cocoapi) is used for mAP calculation.

## Requirements
* Python ≥ 3.6
* Install libraries: `pip install -r requirements.txt`

## Data Preparation
### PASCAL VOC
```bash
cd datasets/voc/

wget http://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
wget http://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar
wget http://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar
tar xf VOCtrainval_06-Nov-2007.tar
tar xf VOCtest_06-Nov-2007.tar
tar xf VOCtrainval_11-May-2012.tar

python prepare.py --root VOCdevkit/
```
### COCO
```bash
cd datasets/coco/

wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip train2017.zip
unzip val2017.zip
unzip annotations_trainval2017.zip

python prepare.py --root .
```

## Configuration
We use YAML for configuration management. See `configs/*/*.yaml` for examples.
You can modify the settings as needed.

## Training
```bash
python train.py --cfg <CONFIG_FILE> --logdir <LOG_DIRECTORY>

# For example, to train SSD300 on PASCAL VOC:
python train.py --cfg configs/voc/ssd300.yaml --logdir runs/voc_ssd300/exp0/
```
To visualize training progress using TensorBoard:
```bash
tensorboard --logdir <LOG_DIRECTORY>
```
An interrupted training can be resumed by:
```bash
# Run train.py with --resume to restore the latest saved checkpoint file in the log directory.
python train.py --cfg <CONFIG_FILE> --logdir <LOG_DIRECTORY> --resume
```

## Evaluation
### PASCAL VOC
```bash
python eval.py --cfg <CONFIG_FILE> --pth <LOG_DIRECTORY>/best.pth --dataset datasets/voc/val.json
```

### COCO
Run `coco_eval.py` to calculate the COCO mAP metric using [pycocotools](https://github.com/cocodataset/cocoapi):
```bash
python coco_eval.py --cfg <CONFIG_FILE> --pth <LOG_DIRECTORY>/best.pth --coco_dir <COCO_DIR>
```
**Note**: <COCO_DIR> can be `datasets/coco/` if you prepare the COCO dataset according to the
instructions in [Data Preparation](#data-preparation).