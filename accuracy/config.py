from collections import namedtuple

__all__ = ['configuration']

Config = namedtuple('Config', ['block_sizes', 'pruning_rates'])

configuration = {
    "vgg16": Config(
        block_sizes=[
            (0, 128, 0, 0), (16, 128, 0, 0), # conv1
            (16, 128, 0, 0), (16, 128, 0, 0), # conv2
            (16, 128, 0, 0), (16, 128, 0, 0), (16, 128, 0, 0), # conv3
            (16, 128, 0, 0), (16, 128, 0, 0), (16, 128, 0, 0), # conv4
            (16, 128, 0, 0), (16, 128, 0, 0), (16, 128, 0, 0), # conv5
            (128, 128), (128, 128), (128, 100) # fc
        ],
        pruning_rates = [
            0.3, 0.5, # conv1
            0.5, 0.5, # conv2
            0.5, 0.5, 0.5, # conv3
            0.5, 0.5, 0.5, # conv4
            0.5, 0.5, 0.5, # conv5
            0.7, 0.7, 0.3, # fc
        ]
    )
}
            
