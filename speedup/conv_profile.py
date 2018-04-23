from blocksparse.conv import BlocksparseConv

import tensorflow as tf
from tensorflow.python.client import timeline
import numpy as np
import json

import sys
import os
import random

def profile(bs, iw, ih, ic, oc, kw, kh, bi, bo, sp):
    """Get the execution time of blocksparse
    Args:
	bs (int): batch size
	iw (int): image width
	ih (int): image height
	ic (int): input channels
	oc (int): output channels
	kw (int): kernel width
	kh (int): kernel height
	bi (int): block input size
	bo (int): block output size
	sp (float): sparsity

    Returns:
	(float, int): actual sparsity and execution time in us
    """
    num_input_blocks = ic / bi
    num_output_blocks = oc / bo
    num_blocks = num_input_blocks * num_output_blocks
    num_pruned_blocks = int(num_blocks * sp)
    num_remain_blocks = num_blocks - num_pruned_blocks
    actual_sparsity = num_pruned_blocks / float(num_blocks)
    
    # generate layout
    layout = np.array([0] * num_pruned_blocks + [1] * num_remain_blocks)
    np.random.shuffle(layout)
    layout = layout.reshape((num_input_blocks, num_output_blocks))

    # generate BCK according to layout
    # BCK is a list of blocks, each block is a tuple of two list: row indices and column indices
    BCK = []
    for i in range(num_input_blocks):
	for j in range(num_output_blocks):
	    if layout[i, j] == 1:
		BCK.append((
		    [c for c in range(i * bi, (i + 1) * bi)],
		    [k for k in range(j * bo, (j + 1) * bo)]
		))
    TRS = (kw, kh)
    DHW = (iw, ih)


    # generate random shuffle order
    indices = range(oc)
    random.shuffle(indices)
    
    tf.reset_default_graph()
    with tf.Session() as sess:
    	# generate operation
    	bs_conv = BlocksparseConv(BCK, TRS, DHW)

    	# build computational graph
    	x = tf.placeholder(tf.float32, shape=bs_conv.i_shape(bs))
    	k = tf.get_variable("k", shape=bs_conv.f_shape(), dtype=tf.float32)
    	i = tf.constant(indices)
    	y = bs_conv(k, x)
    	y = tf.gather(y, i, axis=1)
    
    	# run and profile
    	options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    	run_metadata = tf.RunMetadata()
    	sess.run(tf.global_variables_initializer())
    	sess.run(y, feed_dict={x: np.ones(shape=bs_conv.i_shape(bs), dtype='float32')}, options=options, run_metadata=run_metadata)
    	fetched_timeline = timeline.Timeline(run_metadata.step_stats)
    	chrome_trace = fetched_timeline.generate_chrome_trace_format()
    with open('timeline.json', 'w') as f:
	f.write(chrome_trace)

    # parse the trace
    with open('timeline.json', 'r') as f:
	o = json.load(f)['traceEvents']
    	conv_time = int(next(item for item in o if item['name'] == u'BlocksparseConv')['dur'])
        gather_time = int(next(item for item in o if item['name'].startswith(u'Gather'))['dur'])

    os.remove('timeline.json')

    return actual_sparsity, conv_time + gather_time

with open('conv_exec_time.csv', 'w') as f:
    f.write('block_size, sparsity, execution_time\n')
    for block_size in (8, 16, 32, 64, 128, 256, 512): 
    	for sparsity in (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
    	    actual_sparsity, execution_time = profile(bs=64, iw=28, ih=28, ic=512, oc=512, kw=3, kh=3, bi=int(round(block_size / 9.)), bo=block_size, sp=sparsity)
	    f.write('%d, %f, %d\n' % (block_size, actual_sparsity, execution_time))
	    print('%d, %f, %d' % (block_size, actual_sparsity, execution_time))
