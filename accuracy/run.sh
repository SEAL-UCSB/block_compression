#!/bin/bash

CONFIG=$1
ITER=${3:-1}
CUDA_VISIBLE_DEVICES=$2 python main.py /disk/ImageNet2012/ "$CONFIG.json" --arch=vgg16 --lr=1e-3 --lr-epochs=5 --batch-size=64 --batch-iter=$ITER --prefix="vgg16_bn.$CONFIG" --print-freq=$((10 * $ITER))
