# Recommendation system for training object detection models

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
- `yolov8` - the chosen architecture that we want to train {`yolov5`, `yolov7`, `yolov8`};
- `True` - you need to split the data set into samples {`True`, `False`}. This parameter can be omitted, the initial value is `False`.


### Run script for training
To start the training, enter the following command in the command line: 

```commandline
bash -l custom_train.sh yolov8 True
```

