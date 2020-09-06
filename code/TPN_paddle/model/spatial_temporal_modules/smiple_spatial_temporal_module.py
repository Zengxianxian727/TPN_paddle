import paddle.fluid as fluid
import numpy as np
from paddle.fluid.layer_helper import LayerHelper
#from paddle.fluid.dygraph.nn import Conv2D, Pool2D, BatchNorm, Linear
from paddle.fluid.dygraph.nn import Conv3D, BatchNorm, Linear
import paddle.fluid.dygraph.nn as nn

class SimpleSpatialTemporalModule(fluid.dygraph.Layer):
    def __init__(self, spatial_type='avg', spatial_size=7, temporal_size=1):
        super(SimpleSpatialTemporalModule, self).__init__()

        assert spatial_type in ['avg']
        self.spatial_type = spatial_type

        self.spatial_size = spatial_size if not isinstance(spatial_size, int) else (spatial_size, spatial_size)
        self.temporal_size = temporal_size
        self.pool_size = (self.temporal_size,) + self.spatial_size

    def forward(self, input):
        out = fluid.layers.pool3d(input, self.pool_size, pool_type='avg', pool_stride=1, pool_padding=0)
        return out