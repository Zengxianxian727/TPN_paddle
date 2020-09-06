import paddle.fluid as fluid
import numpy as np
from paddle.fluid.layer_helper import LayerHelper
#from paddle.fluid.dygraph.nn import Conv2D, Pool2D, BatchNorm, Linear
from paddle.fluid.dygraph.nn import Conv3D, BatchNorm, Linear
import paddle.fluid.dygraph.nn as nn

#from ...utils.config import Config


class Identity(fluid.dygraph.Layer):
    
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class ConvModule(fluid.dygraph.Layer):
    
    def __init__(
        self,
        inplanes,
        planes,
        kernel_size,
        stride,
        padding,
        bias=False,
        groups=1):

        super(ConvModule, self).__init__()
        self.conv = Conv3D(
                            num_channels=inplanes, 
                            num_filters=planes, 
                            filter_size=kernel_size, 
                            stride=stride, 
                            padding=padding, 
                            bias_attr=bias, 
                            groups=groups,
                            param_attr=fluid.initializer.XavierInitializer(uniform=True, fan_in=None, fan_out=None, seed=0),)
        self.bn = BatchNorm(num_channels=planes, act='relu',
                            param_attr=fluid.initializer.ConstantInitializer(value=1.0, force_cpu=False),
                            bias_attr=fluid.initializer.ConstantInitializer(value=0.0, force_cpu=False))

    def forward(self, x):
        out = self.bn(self.conv(x))
        return out

class AuxHead(fluid.dygraph.Layer):
    
    def __init__(self, inplanes, planes, loss_weight=0.5):
        super(AuxHead, self).__init__()
        self.convs = ConvModule(inplanes, inplanes * 2, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), bias=False)
        self.loss_weight = loss_weight
        self.dropout_ratio = 0.5
        self.fc = Linear(inplanes*2, planes,
                         param_attr=fluid.initializer.NormalInitializer(loc=0.0, scale=1.0, seed=0),
                         bias_attr=fluid.initializer.ConstantInitializer(value=0.0, force_cpu=False))

    def forward(self, x, target=None):
        if target is None:
            return None
        loss = dict()
        x = self.convs(x)
        x = fluid.layers.adaptive_pool3d(x, pool_size=1, pool_type='avg').squeeze(-1).squeeze(-1).squeeze(-1)
        x = fluid.layers.dropout(x, self.dropout_ratio)
        x = self.fc(x)
        loss['loss_aux'] = self.loss_weight * fluid.layers.softmax_with_cross_entropy(x, target)
        return loss

class TemporalModulation(fluid.dygraph.Layer):
    
    def __init__(self, inplanes, planes, downsample_scale=8):
        super(TemporalModulation, self).__init__()
        self.conv = Conv3D(inplanes, planes, (3, 1, 1), (1, 1, 1), (1, 0, 0),bias_attr=False, groups=32)
        self.downsample_scale=downsample_scale

    def forward(self, x):
        x = self.conv(x)
        x = fluid.layers.pool3d(x, (self.downsample_scale, 1, 1), pool_type='max', pool_stride=(self.downsample_scale, 1, 1), pool_padding=(0, 0, 0), ceil_mode=True)
        return x

class Upsampling(fluid.dygraph.Layer):
    def __init__(self, scale=(2, 1, 1),):
        super(Upsampling, self).__init__()
        self.scale = scale

    def forward(self, x):
        x_shape = x.shape
        x = fluid.layers.reshape(x, [-1, x_shape[1]*x_shape[2], x_shape[3], x_shape[4]])
        x = fluid.layers.interpolate(x, scale=self.scale, resample='NEAREST')
        x = fluid.layers.reshape(x, [x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4]])
        return x

class Downampling(fluid.dygraph.Layer):
    def __init__(self,
                 inplanes,
                 planes,
                 kernel_size=(3, 1, 1),
                 stride=(1, 1, 1),
                 padding=(1, 0, 0),
                 bias=False,
                 groups=1,
                 norm=False,
                 activation=False,
                 downsample_position='after',
                 downsample_scale=(1, 2, 2),):
        super(Downampling, self).__init__()
        self.conv = Conv3D(inplanes, planes, kernel_size, stride, padding, bias_attr=bias, groups=groups)
        self.norm1 = norm
        self.norm = BatchNorm(planes) if norm else None
        self.activation = activation 
        assert (downsample_position in ['before', 'after'])
        self.downsample_position = downsample_position
        self.downsample_scale = downsample_scale
        
    def forward(self, x):
        if self.downsample_position == 'before':
            x = fluid.layers.pool3d(x, self.downsample_scale, pool_type='max', pool_stride=self.downsample_scale, pool_padding=(0, 0, 0), ceil_mode=True)
        x = self.conv(x)
        if self.norm1:
            x = self.norm(x)
        if self.activation:
            x = fluid.layers.relu(x)
        if self.downsample_position == 'after':
            x = fluid.layers.pool3d(x, self.downsample_scale, pool_type='max', pool_stride=self.downsample_scale, pool_padding=(0, 0, 0), ceil_mode=True)
        return x

class LevelFusion(fluid.dygraph.Layer):
    def __init__(self,
                 in_channels=[1024, 1024],
                 mid_channels=[1024, 1024],
                 out_channels=2048,
                 ds_scales=[(1, 1, 1), (1, 1, 1)],):

        super(LevelFusion, self).__init__()
        self.ops=fluid.dygraph.LayerList()
        num_ins = len(in_channels)
        for i in range(num_ins):
            op = Downampling(in_channels[i], mid_channels[i], kernel_size=(1, 1, 1),
                             stride=(1, 1, 1), padding=(0, 0, 0), bias=False, groups=32,
                             norm=True, activation=True, downsample_position='before',
                             downsample_scale=ds_scales[i])
            self.ops.append(op)
        
        in_dims = np.sum(mid_channels)
        self.fusion_conv = fluid.dygraph.Sequential(
            Conv3D(in_dims, out_channels, 1, 1, 0, bias_attr=False),
            BatchNorm(out_channels, act='relu'),
        )

    def forward(self, inputs):
        out = [self.ops[i](feature) for i, feature in enumerate(inputs)]
        out = fluid.layers.concat(out, 1)
        out = self.fusion_conv(out)
        return out

class SpatialModulation(fluid.dygraph.Layer):
    def __init__(self, inplanes=[1024, 2048], planes=2048,):
        super(SpatialModulation, self).__init__()
        self.spatial_modulation = fluid.dygraph.LayerList()
        for i, dim in enumerate(inplanes):
            op = fluid.dygraph.LayerList()
            ds_factor = planes // dim
            ds_num = int(np.log2(ds_factor))
            if ds_num < 1:
                op = Identity()
            else:
                for dsi in range(ds_num):
                    in_factor = 2 ** dsi
                    out_factor = 2 ** (dsi + 1)
                    op.append(ConvModule(dim* in_factor, dim * out_factor, kernel_size=(1, 3, 3), stride=(1, 2, 2),
                                         padding=(0, 1, 1), bias=False))
            self.spatial_modulation.append(op)

    def forward(self, inputs):
        out = []
        for i, feature in enumerate(inputs):
            if isinstance(self.spatial_modulation[i], fluid.dygraph.LayerList):
                out_ = inputs[i]
                for III, op in enumerate(self.spatial_modulation[i]):
                    out_ = op(out_)
                out.append(out_)
            else:
                out.append(self.spatial_modulation[i](inputs[i]))
        return out

class TPN(fluid.dygraph.Layer):

    def __init__(self,
                 in_channels=[256, 512, 1024, 2048],
                 out_channels=256,
                 spatial_modulation_config=None,
                 temporal_modulation_config=None,
                 upsampling_config=None,
                 downsampling_config=None,
                 level_fusion_config=None,
                 aux_head_config=None,):
        super(TPN, self).__init__()
        assert isinstance(in_channels, list)
        assert isinstance(out_channels, int)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)

        spatial_modulation_config = spatial_modulation_config
        temporal_modulation_config = temporal_modulation_config
        upsampling_config = upsampling_config
        downsampling_config = downsampling_config
        aux_head_config = aux_head_config
        level_fusion_config = level_fusion_config
        self.temporal_modulation_ops = fluid.dygraph.LayerList()
        self.upsampling_ops = fluid.dygraph.LayerList()
        self.downsampling_ops = fluid.dygraph.LayerList()
        self.level_fusion_op = LevelFusion(**level_fusion_config)
        self.spatial_modulation = SpatialModulation(**spatial_modulation_config)
        for i in range(0, self.num_ins, 1):
            inplanes = in_channels[-1]
            planes = out_channels

            if temporal_modulation_config is not None:
                param = temporal_modulation_config.param
                param.downsample_scale = temporal_modulation_config.scales[i]
                param.inplanes = inplanes
                param.planes = planes
                print(inplanes, planes)
                print(param)
                temporal_modulation = TemporalModulation(**param)
                self.temporal_modulation_ops.append(temporal_modulation)

            if i < self.num_ins - 1:
                if upsampling_config is not None:

                    upsampling = Upsampling(**upsampling_config)
                    self.upsampling_ops.append(upsampling)

                if downsampling_config is not None:
                    param = downsampling_config.param
                    param.inplanes = planes
                    param.planes = planes
                    param.downsample_scale = downsampling_config.scales
                    downsampling = Downampling(**param)
                    self.downsampling_ops.append(downsampling)

        out_dims = level_fusion_config.out_channels

        # Two pyramids
        self.level_fusion_op2 = LevelFusion(**level_fusion_config)

        self.pyramid_fusion_op = fluid.dygraph.Sequential(
            Conv3D(out_dims * 2, 2048, 1, 1, 0, bias_attr=False),
            BatchNorm(2048, act='relu')
        )
        
        if aux_head_config is not None:
            aux_head_config.inplanes = self.in_channels[-2]
            self.aux_head = AuxHead(**aux_head_config)
        else:
            self.aux_head = None

    # def init_weights(self):

    def forward(self, inputs, target=None):
        loss = None

        if self.aux_head is not None:
            loss = self.aux_head(inputs[-2], target)

        outs = self.spatial_modulation(inputs)

        outs = [temporal_modulation(outs[i]) for i, temporal_modulation in enumerate(self.temporal_modulation_ops)]

        temporal_modulation_outs = outs

        if self.upsampling_ops is not None:
            for i in range(self.num_ins -1, 0, -1):
                outs[i - 1] = outs[i - 1] + self.upsampling_ops[i - 1](outs[i])

        topdownouts = self.level_fusion_op2(outs)
        outs = temporal_modulation_outs

        if self.downsampling_ops is not None:
            for i in range(0, self.num_ins -1, 1):
                outs[i + 1] = outs[i + 1] + self.downsampling_ops[i](outs[i])
        
        outs = self.level_fusion_op(outs)

        outs = self.pyramid_fusion_op(fluid.layers.concat([topdownouts, outs],1))

        return outs, loss











