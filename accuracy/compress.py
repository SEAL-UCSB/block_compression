import numpy as np
import itertools

__all__ = ['blocksparse']

def blocksparse(x, block_sizes, pruning_rate):
    """Blocksparse pruning
    Args:
        x (np.ndarray): a tensor to be pruned.
        block_sizes (tuple[int]): block sizes for each dimension of `x'.
        pruning_rate (float): pruning rate.

    Returns:
        (tuple[tuple[int]], np.ndarray): shuffle orders and mask
    """
    ## prepare

    dim_sizes = x.shape

    # if block_size is 0 on some dimention, we will not prune on that dimension, namely, set the block size to dim size
    block_sizes = [bs if bs > 0 else ds for bs, ds in zip(block_sizes, dim_sizes)]
    block_nums = [int((ds - 1) / bs) + 1 for bs, ds in zip(block_sizes, dim_sizes)]

    # create a dummy matrix x_ that can be dividide into full blocks
    dim_sizes_ = [bs * bn for bs, bn in zip(block_sizes, block_nums)]
    x_ = np.zeros(shape=dim_sizes_)
    
    # copy data from x to x_
    x_[tuple(slice(ds) for ds in dim_sizes)] = np.abs(x)

    # shuffle order for each dimension
    orders = [np.arange(ds) for ds in dim_sizes_]

    num_blocks = np.prod(block_nums)
    num_pruned_blocks = int(num_blocks * pruning_rate) 


    ## initialize EM algorithm
    prev_pruned_sum = np.sum(x_)

    ## EM iteration
    print("=> begin EM algorithm: block_sizes %s, pruning_rate %f" % (str(block_sizes), pruning_rate))

    while True:
        ## E step: choose blocks to be pruned

        # compute sum of each block
        block_sums = np.zeros(shape=block_nums)
        for indices in itertools.product(*(tuple(range(ds)) for ds in block_nums)):
            block_sums[indices] = np.sum(x_[tuple(slice(i*bs, (i+1)*bs) for i, bs in zip(indices, block_sizes))])

        # choose the blocks to be pruned
        block_mask = np.zeros(shape=block_nums, dtype=bool)
        for indices in zip(*(np.unravel_index(block_sums.argsort(axis=None)[:num_pruned_blocks], dims=block_nums))):
            block_mask[indices] = True

        print("==> E-step: %f" % np.sort(block_sums.flatten())[:num_pruned_blocks].sum())

        ## M step: generate a new shuffle order
        for axis in range(len(dim_sizes_)):
            # handle one dimension each time
            reduce_dims = tuple(d for d in range(len(dim_sizes_)) if d != axis)
            dim_mask = block_mask.any(axis=reduce_dims)
            dim_sort = x_.sum(axis=reduce_dims).argsort()
            block_size = block_sizes[axis]
            pruned_start = 0
            remain_start = dim_mask.sum() * block_size

            # generate the new order for the dimension
            dim_order = np.zeros(shape=(dim_sizes_[axis],), dtype=int)
            for i, marked in enumerate(dim_mask):
                if marked:
                    dim_order[i*block_size:(i+1)*block_size] = dim_sort[pruned_start:pruned_start+block_size]
                    pruned_start += block_size
                else:
                    dim_order[i*block_size:(i+1)*block_size] = dim_sort[remain_start:remain_start+block_size]
                    remain_start += block_size
            # shuffle the tensor and order according to the new order
            x_ = x_[tuple(dim_order if i == axis else slice(None) for i in range(len(dim_sizes_)))]
            orders[axis] = orders[axis][dim_order] 

        # Check for break
        pruned_sum = 0
        for indices in itertools.product(*(tuple(range(bn)) for bn in block_nums)):
            if block_mask[indices]:
                pruned_sum += x_[tuple(slice(i*bs, (i+1)*bs) for i, bs in zip(indices, block_sizes))].sum()
        print("==> M-step: %f" % pruned_sum)
        if pruned_sum >= prev_pruned_sum:
            break
        else:
            prev_pruned_sum = pruned_sum

    ## generate mask
    mask = np.zeros(shape=dim_sizes, dtype=bool)
    for indices in itertools.product(*(tuple(range(bn)) for bn in block_nums)):
        mask[tuple(slice(i*bs, (i+1)*bs) for i, bs in zip(indices, block_sizes))] = not block_mask[indices]

    # generate reverse shuffle to shuffle the mask
    orders = tuple(tuple(o for o in order if o < ds) for ds, order in zip(dim_sizes, orders))
    reverse_orders = tuple([-1 for _ in range(ds)] for ds in dim_sizes)
    for axis in range(len(dim_sizes)):
        for i, v in enumerate(orders[axis]):
            reverse_orders[axis][v] = i

    for axis in range(len(reverse_orders)):
        slices = tuple(reverse_orders[axis] if i == axis else slice(None) for i in range(len(reverse_orders)))
        mask = mask[slices] 

    return orders, mask
