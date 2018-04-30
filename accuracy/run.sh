#!/bin/bash

CONFIG=$1
CUDA_VISIBLE_DEVICES=$2
ITER=${3:-1}
python main.py /disk/ImageNet2012/ "$CONFIG.json" --arch=vgg16_bn --lr=1e-3 --lr-epochs=5 --batch-size=64 --batch-iter=$ITER --prefix="vgg16_bn.$CONFIG"
