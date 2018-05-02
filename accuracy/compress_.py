import numpy as np
import itertools

__all__ = ['blocksparse']

def blocksparse(X, block_sizes, pruning_rate):
    """Blocksparse pruning
    
    For input tensor X, we use an EM algorithm to determine the best shuffling order.
        E-step: measure the importance of each block and mark the least important blocks to be pruned.
        M-step: shuffle different dimensions of X to minimize the elements inside pruned blocks.

    The M-step is not that easy, we also use an iterative algorithm to minimize the elements.
    We fix other dimensions and shuffle one dimension each time.

    suppose the mask is M, we have S = XM^T where S_ij represents the pruned value if we put the i-th row to j-th mask. Thus S + S^T describe the pruned value when we swap i and j. The original pruned value in the two columns are tr(S) + tr(S)^T.

    Thus, we find the maximum value of (tr(S) + tr(S)^T) - (S + S^T) and swap i j if it is larger than zero.

    Args:
        x (np.ndarray): a tensor to be pruned.
        block_sizes (tuple[int]): block sizes for each dimension of 'x'.
        pruning_rate (float): pruning rate.

    Returns:
        (tuple[tuple[int]], np.ndarray): shuffle orders and mask
    """
    ## prepare
    X = np.abs(X) 
    
    dim_sizes = X.shape
    num_dims = len(dim_sizes)

    block_sizes = [bs if bs > 0 else ds for bs, ds in zip(block_sizes, dim_sizes)]
    block_nums = [int((ds - 1) / bs) + 1 for bs, ds in zip(block_sizes, dim_sizes)]
    orders = [np.arange(ds) for ds in dim_sizes]
    num_blocks = np.prod(block_nums)
    num_pruned_blocks = int(num_blocks * pruning_rate)

    ## EM iteration
    print("=> begin EM iteration: block_sizes %s, pruning_rate %f" % (str(block_sizes), pruning_rate))
    while True:
        ## E step: choose block to be pruned
        # compute sum of each block
        block_sums = X.reshape(tuple(itertools.chain.from_iterable((bn, bs) for bn, bs in zip(block_nums, block_sizes)))).sum(axis=tuple(2 * i + 1 for i in range(num_dims)))
        # choose the blocks to be pruned
        block_mask = np.zeros_like(block_sums)
        block_mask[np.unravel_index(block_sums.argsort(axis=None)[:num_pruned_blocks], dims=block_nums)] = 1
        mask = (block_mask[tuple([slice(None), None] * num_dims)] * np.ones(block_sizes)[tuple([None, slice(None)] * num_dims)]).reshape(dim_sizes)

        prev_pruned_sum = (X * mask).sum()
        print("==> E-step: pruned sum is %f" % prev_pruned_sum) 

        ## M step: determine the best order
        for axis in range(num_dims):
            contraction_dims = [i for i in range(num_dims) if i != axis]
            S = np.tensordot(X, mask, axes=(contraction_dims, contraction_dims))
            while True:
                D = S.diagonal()
                C = (D[:, None] + D[None, :]) - (S + np.transpose(S))
                if np.max(C, axis=None) < 1e-5:
                    break
                i, j = np.unravel_index(np.argmax(C, axis=None), C.shape)

                ## swap i, j
                S[[i,j],:] = S[[j,i],:]
                orders[axis][[i,j]] = orders[axis][[j,i]]
                X[tuple([i,j] if k == axis else slice(None) for k in range(num_dims))] = X[tuple([j,i] if k == axis else slice(None) for k in range(num_dims))]
                #print("====> Swap gain %f, pruned sum is %f" % (np.max(C, axis=None), (X * mask).sum()))
            print("===> axis %d, pruned sum is %f" % (axis, (X * mask).sum()))
        
        pruned_sum = (X * mask).sum()
        print("==> M-step: pruned sum is %f" % pruned_sum)
        if prev_pruned_sum - pruned_sum < 1e-5:
            break
        else:
            prev_pruned_sum = pruned_sum

    ## generate reverse mask
    for axis in range(num_dims):
        mask[tuple(orders[dim] if dim == axis else slice(None) for dim in range(num_dims))] = mask 
    return orders, mask

