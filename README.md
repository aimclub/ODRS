
# ODRC
[![PythonVersion](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-blue)](https://pypi.org/project/scikit-learn/)

ODRC - it an open source recommendation system for training object detection models. Our system allows you to choose the most 
profitable existing object recognition models based on user preferences and data. In addition to choosing the 
architecture of the model, the system will help you start training and configure the environment.


<center><img src="docs/img/alg_scheme.png" width="400"></center>

Framework provides an opportunity to train the most popular object recognition models (including setting up the environment 
and choosing the architecture of a specific model). Considered two-stage detectors models such as Faster R-CNN and Mask R-CNN as 
well as one-stage detectors such as SSD and YOLO (including families v5, v7, v8).

<center><img src="docs/img/model_list.png" width="400"></center>
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
| [WaRP](https://github.com/AIRI-Institute/WaRP/tree/main)               | Waste Recycling Plant includes labeled images of an industrial waste sorting plant.                                       |                                           |
| [Pascal VOC](https://pjreddie.com/projects/pascal-voc-dataset-mirror/) | Very popular dataset for building and evaluating algorithms for image classification, object detection, and segmentation. |

The list of criteria and data sets will be expanded

## Contents

- [Installation](#installation)
- [Dataset structure](#dataset-structure)
- [ML Recommendation system](#ml-recommendation-system)
- [Detectors Training](#detectors-training)
- [Contacts](#contacts)

## Installation

Download repository and install the necessary dependencies using the following commands:

```markdown
git clone https://github.com/saaresearch/ODRS.git
cd ODRS/
pip install -r requirements.txt 
```
## Dataset structure
To use the recommendation system or train the desired detector, put your dataset in yolo format in the ***user_datasets/yolo*** directory. The set can have the following structures:
```markdown
user_datasets
|_ _yolo
    |_ _ <folder_name_your_dataset>
        |_ _train
            |_ _images
                    |_ <name_1>.jpg
                    |_ ...
                    |_ <name_N>.jpg
            |_ _labels
                    |_ <name_1>.txt
                    |_ ...
                    |_ <name_N>.txt
        |_ _valid
            |_ _images
                    |_ <name_1>.jpg
                    |_ ...
                    |_ <name_N>.jpg
            |_ _labels
                    |_ <name_1>.txt
                    |_ ...
                    |_ <name_N>.txt
        |_ _test
            |_ _images
                    |_ <name_1>.jpg
                    |_ ...
                    |_ <name_N>.jpg
            |_ _labels
                    |_ <name_1>.txt
                    |_ ...
                    |_ <name_N>.txt

```
***or you can use the following structure, then your set will be automatically divided into samples:***

```markdown
user_datasets
|_ _yolo
    |_ _ <folder_name_your_dataset>
            |_ <name_1>.jpg
            |_ ...
            |_ <name_N>.jpg
            |_ ...
            |_ <name_1>.txt
            |_ ...
            |_ <name_N>.txt

```

Add to the root directory of the project ***.txt*** a file containing the names of all classes in your set of images.

Example **classes.txt**:
```markdown
boat
car
dock
jetski
lift
```
## ML Recommendation system
After you have placed your dataset in the folder ***user_datasets/yolo*** and created in the root directory ***.txt*** a file containing the names of all classes in your set of images. You can start working with the main functionality of the project.

1. In order to use the recommendation system, you need to configure **ml_config.yaml**. Go to the desired directory:
    ```markdown
    cd ODRS/ml_utils/config/
    ```
2. Open **ml_config.yaml** and set the necessary parameters and paths:
    ```markdown
    #dataset_path: path to data folder
    #classes_path: path to classes.txt
    #GPU: True/False
    #speed: 1 - 5 if u want max speed choose 5 or u wanna lower speed 1
    #accuracy: 1 - 10 if u want max accuracy choose 10 or u wanna lower acc 1


    dataset_path: "/media/farm/ssd_1_tb_evo_sumsung/ODRC_2/ODRS/user_datasets/yolo/Aerial_Maritime"
    classes_path: "/media/farm/ssd_1_tb_evo_sumsung/ODRC_2/ODRS/classes.txt"

    GPU: True
    speed: 1
    accuracy: 10

    models_array: ["yolov5l", "yolov5m", "yolov5n", "yolov5s", "yolov5x",
                "yolov7x", "yolov7", "yolov7-tiny", "yolov8x6", "yolov8x",
                "yolov8s", "yolov8n", "yolov8m", "faster-rcnn", "ssd"]

    ```
3. Go to the script **ml_model_optimizer.py ** and start it:
    ```markdown
    cd ..
    python ml_model_optimizer.py
    ```
4. If everything worked successfully, you will see something like the following answer:
    ```markdown
    Number of images: 1016
    W: 800
    H: 600
    Gini Coefficient: 64.0
    Number of classes: 5
    Top models for training:
    1) yolov5x
    2) yolov5l
    3) yolov8x6
    ```

## Detectors Training
1. Go to the directory containing ***custom_config.yaml*** in which the training parameters are specified.
2. Setting up training parameters:
    ```markdown
    # Path to data
    DATA_PATH: "/media/farm/ssd_1_tb_evo_sumsung/ODRC_2/ODRS/user_datasets/yolo/Aerial_Maritime"

    # parameters for autosplit dataset
    SPLIT_TRAIN_VALUE: 0.6
    SPLIT_VAL_VALUE: 0.35
    SPLIT_TEST_VALUE: 0.05


    # Name *.txt file with names classes
    CLASSES: "classes.txt"

    IMG_SIZE: "510"
    BATCH_SIZE: "16"
    EPOCHS: "10"

    # This file generated automaticaly
    CONFIG_PATH: "dataset.yaml"

    # NOTE: using only for yolo architecture
    #1)YOLOV5: To view models for the yolov5 architecture, use this path: ODRS/ODRS/train_utils/train_model/models/yolov5/models
    #2)YOLOV7: To view models for the yolov7 architecture, use this path: ODRS/ODRS/train_utils/train_model/models/yolov7/cfg/training
    #3)YOLOV8: To view models for the yolov8 architecture, use this path: ODRS/ODRS/train_utils/train_model/models/ultralytics/ultralytics/models/v8/

    MODEL_PATH: '/media/farm/ssd_1_tb_evo_sumsung/ODRC_2/ODRS/ODRS/train_utils/train_model/models/ultralytics/ultralytics/models/v8/yolov8l.yaml'

    # NOTE: using only for yolo architecture 
    GPU_COUNT: 4
    SELECT_GPU: "0,1,2,3"
    ```
3. Starting training:
**NOTE**: If, for example, you specified in ***custom_config.yaml***, the path to the yolov5 model, and you want to start yolov8, training will not start.
- `yolov8` - the chosen architecture that we want to train {`yolov5`, `yolov8`, `yolov7`, `ssd`, `rcnn`};

    ```markdown
    cd ODRS/ODRS/train_utils/train_model
    python custom_train_all.py --arch yolov8
    ```
4. After the training, you will see in the root directory ***ODRS*** a new directory ***runs***, all the results of experiments will be saved in it. For convenience, the result of each experiment is saved in a separate folder in the following form:
    ```markdown
    <year>-<mounth>-<day>_<hours>-<minutes>-<seconds>_<acrh>
    |_ _exp
        |_...
    ```

## Contacts
- [Telegram channel](https://t.me/) 
- [VK group](<https://vk.com/>)


