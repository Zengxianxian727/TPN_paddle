import paddle.fluid as fluid
import numpy as np
from paddle.fluid.layer_helper import LayerHelper
#from paddle.fluid.dygraph.nn import Conv2D, Pool2D, BatchNorm, Linear
from paddle.fluid.dygraph.nn import Conv3D, BatchNorm, Linear
import paddle.fluid.dygraph.nn as nn

# class _SimpleConsensus(torch.autograd.Function):


class SimpleConsensus(fluid.dygraph.Layer):
    def __init__(self, consensus_type, dim=1):
        super(SimpleConsensus, self).__init__()

        assert consensus_type in ['avg']
        self.consensus_type = consensus_type
        self.dim = dim

    def forward(self, x):
        self.shape = x.shape
        if self.consensus_type == 'avg':
            output = fluid.layers.reduce_mean(x, dim=self.dim, keep_dim=True)
        else:
            output = None

        return output    # 这里是有问题的，它原文要求重新做一个_SimpleConsensus的。