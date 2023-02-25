
# ODRC

ODRC - it an open source recommendation system for training object detection models. Our system allows you to choose the most 
profitable existing object recognition models based on user preferences and data. In addition to choosing the 
architecture of the model, the system will help you start training and configure the environment.


<center><img src="doc/img/alg_scheme.png" width="400"></center>

Framework provides an opportunity to train the most popular object recognition models (including setting up the environment 
and choosing the architecture of a specific model). Considered two-stage detectors models such as Faster R-CNN and Mask R-CNN as 
well as one-stage detectors such as SSD and YOLO (including families v5, v7, v8).

<center><img src="doc/img/model_list.png" width="400"></center>
The recommendation algorithm is based on production rules. The primary set of rules (knowledge base) is formed on 
the basis of the results of the analysis of scientific sources and standard data sets, but also empirical processing 
of data sets from specific industries.
The main criteria for drawing up the rules were chosen:

* Dimension of the model 
* The value of metrics (mAP, Recall, Accuracy) for selected datasets
* The speed of the model on GPU and CPU
* Supported image format and dimension

The experiments were carried out on the following data sets:

| Dataset                                                                | Description                                                                                                               |
|------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------|
| Extended WaRP                                                          | An extended version of the WaRP dataset provided by [Insystem](https://insystem.io/)                                      |
| [WaRP](https://github.com/AIRI-Institute/WaRP/tree/main)               | Waste Recycling Plant includes labeled images of an industrial waste sorting plant.                                       |
| [COCO](https://cocodataset.org/#home)                                  | COCO is a large-scale object detection, segmentation, and captioning dataset.                                             |
| [Pascal VOC](https://pjreddie.com/projects/pascal-voc-dataset-mirror/) | Very popular dataset for building and evaluating algorithms for image classification, object detection, and segmentation. |

The list of criteria and data sets will be expanded

## Setup environment script
This script is used to configure the learning environment.

### Arguments for setup
- `yolov8` - the architecture we want to use {`yolov5`, `yolov7`, `yolov8`};

### Run script for setup
To start setup environment, enter the following command in the command line: 

```commandline
bash -l create_env.sh yolov8
```

## Training script
The developed script allows you to train different versions of yolo architectures and split the existing dataset of images into samples.


### Configurations for training
The `config` folder contains configurations for each of the architectures:
* `DATA_PATH` - Path to dataset;
* `CLASSES` - Information about the classes contained in the image set;
* `IMG_SIZE` - Size of input images as integer or w,h;
* `BATCH_SIZE` - Batch size for training;
* `EPOCHS` - Number of epochs to train for;
* `CONFIG_PATH` - Path to config dataset;
* `MODEL_PATH` - Path to model file (yaml);
* `SPLIT_TRAIN_VALUE` - Percentage allocated for the training dataset;
* `SPLIT_VAL_VALUE` - Percentage allocated for the validation dataset;
* `SPLIT_TEST_VALUE` - Percentage allocated for the test dataset.


### Arguments for training
- `yolov8` - the chosen architecture that we want to train {`yolov5`, `yolov8`, `yolov7`};
- `True` - you need to split the data set into samples {`True`, `False`}. This parameter can be omitted, the initial value is `False`.


### Run script for training
To start the training, enter the following command in the command line: 

```commandline
bash -l custom_train.sh yolov8 True
```

