import time

import torch
import numpy


def block_matmul(W, X, m, n):
    Y = []
    for i in range(m):
        y = 0
        for j in range(n):
            if W[i][j] is not None:
                y += torch.mm(X[j], W[i][j])
        Y.append(y)
    return Y


def measure(b, w, h, m, n, p):
    # generate cases
    num_total_block = m * n
    num_prune_block = p
    num_remain_block = num_total_block - num_prune_block
    mask = numpy.array([False] * num_prune_block + [True] * num_remain_block)
    numpy.random.shuffle(mask)
    mask = mask.reshape((m, n))
    W = [[
            torch.rand(w, h).cuda() if mask[i, j] else None
            for j in range(n)
        ]
        for i in range(m)
    ]
    X = [
        torch.rand(b, w).cuda()
        for j in range(n)
    ]

    torch.cuda.synchronize()
    start = time.time()
    Y = block_matmul(W, X, m, n)
    torch.cuda.synchronize()
    end = time.time()
    return start - end

for p in range(64):
    print(p, measure(64, 128, 128, 8, 8, 0))
