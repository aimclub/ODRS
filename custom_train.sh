#!/bin/bash

#example: bash -l custom_train.sh yolov8 True

if [ "$1" = "yolov5" ] || [ "$1" = "yolov7" ] || [ "$1" = "yolov8" ]; then
    source /home/farm/anaconda3/bin/activate $1
    if [ "$2" = "True" ]; then
        python custom_train_all.py --arch $1 --split True
    else
        python custom_train_all.py --arch $1
    fi
else
    echo "Enter one of possible options {yolov5/yolov7/yolov8/SSD/Faster-RCNN}"
fi