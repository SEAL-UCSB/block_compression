from collections import namedtuple

__all__ = ['configuration']

Config = namedtuple('Config', ['block_sizes', 'pruning_rates'])

configuration = {
    "vgg16": Config(
        block_sizes=[
            (8, 1, 0, 0), (8, 1, 0, 0), # conv1
            (8, 1, 0, 0), (8, 1, 0, 0), # conv2
            (8, 1, 0, 0), (8, 1, 0, 0), (8, 1, 0, 0), # conv3
            (8, 1, 0, 0), (8, 1, 0, 0), (8, 1, 0, 0), # conv4
            (8, 1, 0, 0), (8, 1, 0, 0), (8, 1, 0, 0), # conv5
            (8, 8), (8, 8), (8, 10) # fc
        ],
        pruning_rates = [
            0.42, 0.78, # conv1
            0.66, 0.64, # conv2
            0.47, 0.76, 0.58, # conv3
            0.68, 0.78, 0.66, # conv4
            0.65, 0.71, 0.64, # conv5
            0.96, 0.96, 0.77, # fc
        ]
    )
}
            
