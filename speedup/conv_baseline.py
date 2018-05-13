#!/usr/bin/env python

import torch
import sys
import time

# cudnn is tooooo fast!! it has too many in house optimization. we use cublas as baseline
torch.backends.cudnn.enabled = False

times = 100

def profile(bs, iw, ih, ic, oc, kw, kh):
    conv = torch.nn.Conv2d(ic, oc, kernel_size=(kw, kh), stride=(1, 1), padding=(kw/2, kh/2)).cuda()
    x = torch.autograd.Variable(torch.randn(bs, ic, iw, ih)).cuda()
    # run once for initialization overhead
    y = conv(x)
    torch.cuda.synchronize()
    begin = time.time()
    for i in range(times):
        y = conv(x)
    torch.cuda.synchronize()
    end = time.time()

    return (end - begin) / times


vgg16_config = [ 
    [224, 224, 3, 64, 3, 3],
    [224, 224, 64, 64, 3, 3],
    [112, 112, 64, 128, 3, 3],
    [112, 112, 128, 128, 3, 3],
    [56, 56, 128, 256, 3, 3],
    [56, 56, 256, 256, 3, 3],
    [56, 56, 256, 256, 3, 3],
    [28, 28, 256, 512, 3, 3],
    [28, 28, 512, 512, 3, 3],
    [28, 28, 512, 512, 3, 3],
    [14, 14, 512, 512, 3, 3],
    [14, 14, 512, 512, 3, 3],
    [14, 14, 512, 512, 3, 3]
]

total = 0
for i in range(11):
    l = vgg16_config[i]
    execution_time = profile(bs=64, iw=l[0], ih=l[1], ic=l[2], oc=l[3], kw=l[4], kh=l[5])
    total += execution_time
    print(execution_time)

print(total)
