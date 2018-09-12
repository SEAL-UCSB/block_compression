# Block Compression
## Introduction
The block compression project includes the code to test the accuracy and speedup for block-wise pruning with/without reordering algorithm.

## Documentation

### Speedup
The 'speedup' folder contains code and result for speedup.
To run this code, make sure tensorflow, pytorch and [blocksparse](https://github.com/openai/blocksparse) (from OpenAI) are installed.

We have sparse and dense result for convolution and matrix multiplication.
The sparse cases are based on blocksparse library with tensorflow.
The baseline cases are based on pytorch because tensorflow does not support convolution without cudnn, and we plan to use cublas as baseline.

To get the results, just run the four python code in that folder.

### Accuracy
The 'accuracy' folder contains code and configurations for accuracy.
To run this code, make sure pytorch, torchvision are installed and run the following command to prune and fine-tune the model.
```bash
CUDA_VISIBLE_DEVICES=0 python -W ignore main.py /path/to/ImageNet/ "config.json" --arch=VGG16_bn --lr=1e-3 --batch-size=64 --prefix=VGG16_bn
```
You should specify the path to ImageNet dataset, the model architecture used (in torchvision), and the pruning configurations (the json file).

We have provide many configuration files (all the json files).
It provide the block size for each dimension of each layer and the pruning rates.

### Important nodes
Pruning codes are implemented on GPU.
It takes about 1 minute and 12GB GPU memory to prune VGG16.
Although I wrote the CPU implementations as well, it still has some bugs and takes really a long time to run.
Make sure your GPU device provides enought memory.
