from collections import namedtuple

__all__ = ['configuration']

Config = namedtuple('Config', ['block_sizes', 'pruning_rates'])

configuration = {
    "vgg16": Config(
        block_sizes=[
            (32, 0, 0, 0), (32, 2, 0, 0), # conv1
            (64, 8, 0, 0), (64, 8, 0, 0), # conv2
            (64, 8, 0, 0), (64, 8, 0, 0), (64, 8, 0, 0), # conv3
            (64, 8, 0, 0), (64, 8, 0, 0), (64, 8, 0, 0), # conv4
            (64, 8, 0, 0), (64, 8, 0, 0), (64, 8, 0, 0), # conv5
            (64, 64), (64, 64), (64, 50) # fc
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
            
