#!/bin/bash

CONFIG=$1
DEV=$2
ARCH=$3
ITER=${4:-1}
CUDA_VISIBLE_DEVICES=$DEV python main.py /disk/ImageNet2012/ "$CONFIG.json" --arch=$ARCH --lr=1e-3 --lr-epochs=5 --batch-size=64 --batch-iter=$ITER --prefix="$ARCH.$CONFIG.no_shuffle" --print-freq=$((10 * $ITER)) --no-shuffle
