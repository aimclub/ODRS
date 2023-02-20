# Recommendation system for training object detection models

## Description
The developed script allows you to train different versions of yolo architectures and split the existing dataset of images into samples.

## Configurations
The `config` folder contains configurations for each of the architectures:
* `DATA_PATH` - Path to dataset;
* `CLASSES` - Information about the classes contained in the image set;
* `IMG_SIZE` - Size of input images as integer or w,h;
* `BATCH_SIZE` - Batch size for training;
* `EPOCHS` - Number of epochs to train for;
* `CONFIG_PATH` - Path to config dataset;
* `MODEL_PATH` - Path to model file (yaml);
* `PATH_SPLIT_TRAIN` - Path to train dataset;
* `PATH_SPLIT_VALID` - Path to validation dataset.


### Arguments
- `yolov8` - the chosen architecture that we want to train {`yolov5`, `yolov6`, `yolov7`};
- `True` - do you need to split the data set into samples {`True`, `False`}. This parameter can be omitted, the initial value is `False`.


### Run script
To start the training, enter the following command in the command line: 

```commandline
bash -l custom_train.sh yolov8 True
```

