#!/bin/bash

CONFIG=$1
ITER=${4:-1}
CUDA_VISIBLE_DEVICES=$2 python main.py /disk/ImageNet2012/ "$CONFIG.json" --arch=$3 --lr=1e-3 --lr-epochs=5 --batch-size=64 --batch-iter=$ITER --prefix="vgg16_bn.$CONFIG" --print-freq=$((10 * $ITER))
