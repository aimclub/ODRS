#!/bin/bash
if [ "$1" = "yolov5" ] || [ "$1" = "yolov7" ] || [ "$1" = "yolov8" ]; then
    conda create --name $1 python=3.8.13
    if [ "$1" = "yolov8" ]; then
        pip install -r ultralytics/requirements.txt
    else 
        pip install -r $1/requirements.txt
    fi
else
    echo "Enter one of possible options {yolov5/yolov7/yolov8}"
fi
echo "Create conda environment for "$1
