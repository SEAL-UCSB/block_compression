#!/usr/bin/env python

from blocksparse.conv import BlocksparseConv

import tensorflow as tf
from tensorflow.python.client import timeline
import numpy as np
import json

import sys
import os
import random
from collections import namedtuple
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
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
    if bi > ic:
        bi = ic
    if bo > oc:
        bo = oc
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

alexnet_config = {
    "conv1": [56, 56, 3, 64, 11, 11],
    "conv2": [28, 28, 64, 192, 5, 5],
    "conv3": [14, 14, 192, 384, 3, 3],
    "conv4": [14, 14, 384, 256, 3, 3],
    "conv5": [14, 14, 256, 256, 3, 3]
}

vgg16_config = {
    "conv1.1": [224, 224, 3, 64, 3, 3],
    "conv1.2": [224, 224, 64, 64, 3, 3],
    "conv2.1": [112, 112, 64, 128, 3, 3],
    "conv2.2": [112, 112, 128, 128, 3, 3],
    "conv3.1": [56, 56, 128, 256, 3, 3],
    "conv3.2": [56, 56, 256, 256, 3, 3],
    "conv3.3": [56, 56, 256, 256, 3, 3],
    "conv4.1": [28, 28, 256, 512, 3, 3],
    "conv4.2": [28, 28, 512, 512, 3, 3],
    "conv4.3": [28, 28, 512, 512, 3, 3],
    "conv5.1": [14, 14, 512, 512, 3, 3],
    "conv5.2": [14, 14, 512, 512, 3, 3],
    "conv5.3": [14, 14, 512, 512, 3, 3]
}
"""
for k, v in vgg16_config.items():
    print('process %s' % (k))
    with open('vgg16_%s.csv' % k, 'w') as f:
        f.write('block_size, sparsity, execution_time\n')
        for block_size in (8, 16, 32, 64, 128, 256): 
            if block_size > v[3] or block_size > v[3]:
                 break
    	    for sparsity in (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
    	        actual_sparsity, execution_time = profile(bs=64, iw=v[0], ih=v[1], ic=v[2], oc=v[3], kw=v[4], kh=v[5], bi=3, bo=block_size, sp=sparsity)
	        f.write('%d, %f, %d\n' % (block_size, actual_sparsity, execution_time))
	    	print('%d, %f, %d' % (block_size, actual_sparsity, execution_time))
"""
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

# profile config
config_fn = sys.argv[1]
Config = namedtuple('Config', ['block_sizes', 'pruning_rates'])
with open(config_fn, 'r') as f:
    configuration = json.load(f)
configuration = Config(**configuration['vgg16_bn'])

total = 0
for i in range(11):
    l = vgg16_config[i]
    b = configuration.block_sizes[i]
#    b = [1, 1]
    s = configuration.pruning_rates[i]
    _, execution_time = profile(bs=64, iw=l[0], ih=l[1], ic=l[2], oc=l[3], kw=l[4], kh=l[5], bi=b[0] if b[0] > 0 else l[2], bo=b[1] if b[1] > 0 else l[3], sp=s)
    execution_time *= 1e-6
    total += execution_time
    print('===> %f' % execution_time)

print(total)

