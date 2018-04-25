from collections import namedtuple

__all__ = ['configuration']

Config = namedtuple('Config', ['block_sizes', 'pruning_rates'])

configuration = {
    "vgg16": Config(
        block_sizes=[
            (32, 0, 0, 0), (32, 2, 0, 0), # conv1
            (64, 8, 0, 0), (64, 8, 0, 0), # conv2
            (128, 16, 0, 0), (128, 16, 0, 0), (128, 16, 0, 0), # conv3
            (256, 32, 0, 0), (256, 32, 0, 0), (256, 32, 0, 0), # conv4
            (256, 32, 0, 0), (256, 32, 0, 0), (256, 32, 0, 0), # conv5
            (256, 256), (256, 256), (256, 200) # fc
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
            
