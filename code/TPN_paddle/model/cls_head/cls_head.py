import paddle.fluid as fluid
import numpy as np
from paddle.fluid.layer_helper import LayerHelper
#from paddle.fluid.dygraph.nn import Conv2D, Pool2D, BatchNorm, Linear
from paddle.fluid.dygraph.nn import Conv3D, BatchNorm, Linear
import paddle.fluid.dygraph.nn as nn

class ClsHead(fluid.dygraph.Layer):

    def __init__(self,
                 with_avg_pool=True,
                 temporal_feature_size=1,
                 spatial_feature_size=7,
                 dropout_ratio=0.8,
                 in_channels=2048,
                 num_classes=101,
                 fcn_testing=False,
                 init_std=0.01):
        super(ClsHead, self).__init__()

        self.with_avg_pool = with_avg_pool
        self.dropout_ratio = dropout_ratio
        self.in_channels = in_channels
        self.dropout_ratio = dropout_ratio
        self.temporal_feature_size = temporal_feature_size
        self.spatial_feature_size = spatial_feature_size
        self.init_std = init_std
        self.fcn_testing = fcn_testing

        if self.fcn_testing:
            self.new_cls = None
            self.in_channels = in_channels
            self.num_classes = num_classes
        self.fc_cls = Linear(in_channels, num_classes,
                            )

    def forward(self, x):
        if not self.fcn_testing:
            if len(x.shape) == 4:
                x = fluid.layers.unsqueeze(x, axes=2)
            assert x.shape[1] == self.in_channels
            assert x.shape[2] == self.temporal_feature_size
            assert x.shape[3] == self.spatial_feature_size
            assert x.shape[4] == self.spatial_feature_size
            if self.with_avg_pool:
                x = fluid.layers.pool3d(x, pool_size=(self.temporal_feature_size, self.spatial_feature_size, self.spatial_feature_size), pool_type='avg',
                                        pool_stride=(1, 1, 1), pool_padding=(0, 0, 0))
            if self.dropout_ratio != 0:
                x = fluid.layers.dropout(x, self.dropout_ratio)
            x = fluid.layers.reshape(x, (x.shape[0], -1))
            cls_score = self.fc_cls(x)
            return cls_score
        else:
            if self.with_avg_pool:
                x = fluid.layers.pool3d(x, pool_size=(self.temporal_feature_size, self.spatial_feature_size, self.spatial_feature_size), pool_type='avg',
                                        pool_stride=(1, 1, 1), pool_padding=(0, 0, 0))
            if self.new_cls is None:
                self.new_cls = Conv3D(self.in_channels, self.num_classes, 1, 1, 0)
                # 这里需要再想想
                self.fc_cls = None
            class_map = self.new_cls(x)
            return class_map

    def loss(self,
             cls_score,
             labels):
        losses = dict()
        losses['loss_cls'] = fluid.layers.softmax_with_cross_entropy(cls_score, labels)

        return losses
