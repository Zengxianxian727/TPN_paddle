import paddle.fluid as fluid
import numpy as np
from paddle.fluid.layer_helper import LayerHelper
#from paddle.fluid.dygraph.nn import Conv2D, Pool2D, BatchNorm, Linear
from paddle.fluid.dygraph.nn import Conv3D, BatchNorm, Linear
import paddle.fluid.dygraph.nn as nn

import logging
# 3x3x3的卷积模块
def conv3x3x3(in_planes, out_planes, spatial_stride=1, temporal_stride=1, dilation=1):
    "3x3x3 convolution with padding"
    return nn.Conv3D(
        num_channels=in_planes,
        num_filters=out_planes,
        filter_size=3,
        stride=(temporal_stride, spatial_stride, spatial_stride),
        padding=dilation,
        dilation=dilation,
        bias_attr=False,
    )
# 1x3x3的卷积模块
def conv1x3x3(in_planes, out_planes, spatial_stride=1, temporal_stride=1, dilation=1):
    "1x3x3 convolution with padding"
    return nn.Conv3D(
        num_channels=in_planes,
        num_filters=out_planes,
        filter_size=(1, 3, 3),
        stride=(temporal_stride, spatial_stride, spatial_stride),
        padding=(0, dilation, dilation),
        dilation=dilation,
        bias_attr=False,
    )
# 定义bottleneckBlock,   但nonlocal 和 with_cp没搞定
class Bottleneck(fluid.dygraph.Layer):
    expansion=4

    def __init__(self,
                 inplanes,
                 planes,
                 spatial_stride=1,
                 temporal_stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 if_inflate=True,
                 inflate_style='3x1x1',
                 if_nonlocal=True,
                 nonlocal_cfg=None,
                 with_cp=False):
        super(Bottleneck, self).__init__()
        assert style in ['pytorch', 'caffe']
        assert inflate_style in ['3x1x1', '3x3x3']
        self.inplanes = inplanes
        self.planes = planes

        if style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = spatial_stride
            self.conv1_stride_t = 1
            self.conv2_stride_t = temporal_stride
        else:
            self.conv1_stride = spatial_stride
            self.conv2_stride = 1
            self.conv1_stride_t = temporal_stride
            self.conv2_stride_t = 1
        if if_inflate:
            if inflate_style == '3x1x1':
                self.conv1 = Conv3D(
                    num_channels=inplanes,
                    num_filters=planes,
                    filter_size=(3, 1, 1),
                    stride=(self.conv1_stride_t, self.conv1_stride, self.conv1_stride),
                    padding=(1, 0, 0),
                    bias_attr=False)
                self.conv2 = Conv3D(
                    num_channels=planes,
                    num_filters=planes,
                    filter_size=(1, 3, 3),
                    stride=(self.conv2_stride_t, self.conv2_stride, self.conv2_stride),
                    padding=(0, dilation, dilation),
                    dilation=(1, dilation, dilation),
                    bias_attr=False)
            else:
                self.conv1 = Conv3D(
                    num_channels=inplanes,
                    num_filters=planes,
                    filter_size=1,
                    stride=(self.conv1_stride_t, self.conv1_stride, self.conv1_stride),
                    bias_attr=False)
                self.conv2 = Conv3D(
                    num_channels=planes,
                    num_filters=planes,
                    filter_size=3,
                    stride=(self.conv2_stride_t, self.conv2_stride, self.conv2_stride),
                    padding=(1, dilation, dilation),
                    dilation=(1, dilation, dilation),
                    bias_attr=False)
        else:
            self.conv1 = Conv3D(
                num_channels=inplanes,
                num_filters=planes,
                filter_size=1,
                stride=(1, self.conv1_stride, self.conv1_stride),
                bias_attr=False)
            self.conv2 = Conv3D(
                num_channels=planes,
                num_filters=planes,
                filter_size=(1, 3, 3),
                stride=(1, self.conv2_stride, self.conv2_stride),
                padding=(0, dilation, dilation),
                dilation=(1, dilation, dilation),
                bias_attr=False)

        self.bn1 = BatchNorm(planes, act='relu')
        self.bn2 = BatchNorm(planes, act='relu')
        self.conv3 = Conv3D(
            planes, planes * self.expansion, filter_size=1, bias_attr=False)
        self.bn3 = BatchNorm(planes * self.expansion)
        self.downsample = downsample
        self.spatial_tride = spatial_stride
        self.temporal_tride = temporal_stride
        self.dilation = dilation
        self.with_cp = with_cp

        if if_nonlocal and nonlocal_cfg is not None:
            nonlocal_cfg_ = nonlocal_cfg.copy()
            nonlocal_cfg_['in_channels'] = planes * self.expansion
            self.nonlocal_block = None #build_nonlocal_block(nonlocal_cfg_)
        else:
            self.nonlocal_block = None

    def forward(self, x):

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.bn1(out)

            out = self.conv2(out)
            out = self.bn2(out)

            out = self.conv3(out)
            out = self.bn3(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity
            
            return out

        if self.with_cp:
            #out = cp.checkpoint(_inner_forward, x)
            out=0
        else:
            out = _inner_forward(x)

        out = fluid.layers.relu(out)

        if self.nonlocal_block is not None:
            out = self.nonlocal_block(out)

        return out


def make_res_layer(
                    block,
                    inplanes,
                    planes,
                    blocks,
                    spatial_stride=1,
                    temporal_stride=1,
                    dilation=1,
                    style='pytorch',
                    inflate_freq=1,
                    inflate_style='3x1x1',
                    nonlocal_freq=1,
                    nonlocal_cfg=None,
                    with_cp=False):
    
    inflate_freq = inflate_freq if not isinstance(inflate_freq, int) else (inflate_freq,) * blocks
    nonlocal_freq = nonlocal_freq if not isinstance(nonlocal_freq, int) else (nonlocal_freq,) * blocks
    assert len(inflate_freq) == blocks
    assert len(nonlocal_freq) == blocks
    downsample = None
    if spatial_stride != 1 or inplanes != planes * block.expansion:
        downsample = fluid.dygraph.Sequential(
            Conv3D(
                num_channels=inplanes,
                num_filters=planes * block.expansion,
                filter_size=1,
                stride=(temporal_stride, spatial_stride, spatial_stride),
                bias_attr=False),
            BatchNorm(planes * block.expansion),
        )

    layers = []
    layers.append(
        block(
            inplanes,
            planes,
            spatial_stride,
            temporal_stride,
            dilation,
            downsample,
            style=style,
            if_inflate=(inflate_freq[0] == 1),
            inflate_style=inflate_style,
            if_nonlocal=(nonlocal_freq[0] == 1),
            nonlocal_cfg=nonlocal_cfg,
            with_cp=with_cp))
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(
            block(inplanes,
                  planes,
                  1, 1,
                  dilation,
                  style=style,
                  if_inflate=(inflate_freq[i] == 1),
                  inflate_style=inflate_style,
                  if_nonlocal=(nonlocal_freq[i] == 1),
                  nonlocal_cfg=nonlocal_cfg,
                  with_cp=with_cp))

    return fluid.dygraph.Sequential(*layers)

class ResNet_I3D(fluid.dygraph.Layer):
    arch_settings = {
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(
                self,
                depth,
                pretrained=None,
                pretrained2d=True,
                num_stages=4,
                spatial_strides=(1, 2, 2, 2),
                temporal_strides=(1, 1, 1, 1),
                dilations=(1, 1, 1, 1),
                out_indices=(0, 1, 2, 3),
                conv1_kernel_t=5,
                conv1_stride_t=2,
                pool1_kernel_t=1,
                pool1_stride_t=2,
                style='pytorch',
                frozen_stages=-1,
                inflate_freq=(1, 1, 1, 1),
                inflate_stride=(1, 1, 1, 1),
                inflate_style='3x1x1',
                nonlocal_stages=(-1, ),
                nonlocal_freq=(0, 1, 1, 0),
                nonlocal_cfg=None,
                bn_eval=False,
                bn_frozen=False,
                partial_bn=False,
                with_cp=False):
        super(ResNet_I3D, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError('invalid depth {} for resnet'.format(depth))
        self.depth = depth
        self.pretrained = pretrained
        self.pretrained2d = pretrained2d
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.spatial_strides = spatial_strides
        self.temporal_strides = temporal_strides
        self.dilations = dilations
        assert len(spatial_strides) == len(temporal_strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.style = style
        self.frozen_stages = frozen_stages
        self.inflate_freqs = inflate_freq if not isinstance(inflate_freq, int) else (inflate_freq,) * num_stages
        self.inflate_style = inflate_style
        self.nonlocal_stages = nonlocal_stages
        self.nonlocal_freqs = nonlocal_freq if not isinstance(nonlocal_freq, int) else (nonlocal_freq,) * num_stages
        self.nonlocal_cfg = nonlocal_cfg
        self.bn_eval = bn_eval
        self.bn_frozen = bn_frozen
        self.partial_bn = partial_bn
        self.with_cp = with_cp
        self.pool1_kernel_t = pool1_kernel_t
        self.pool1_stride_t = pool1_stride_t

        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = 64

        self.conv1 = Conv3D(
            3, 64, filter_size=(conv1_kernel_t, 7, 7), stride=(conv1_stride_t, 2, 2),
            padding=((conv1_kernel_t - 1) // 2, 3, 3), bias_attr=False)
        self.bn1 = BatchNorm(64, act='relu')
        #self.maxpool = nn.MaxPool3d(kernel_size=(pool1_kernel_t, 3, 3), stride=(pool1_stride_t, 2, 2),
        #                            padding=(pool1_kernel_t // 2, 1, 1))

        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            spatial_stride = spatial_strides[i]
            temporal_stride = temporal_strides[i]
            dilation = dilations[i]
            planes = 64 * 2 ** i
            res_layer = make_res_layer(
                self.block,
                self.inplanes,
                planes,
                num_blocks,
                spatial_stride=spatial_stride,
                temporal_stride=temporal_stride,
                dilation=dilation,
                style=self.style,
                inflate_freq=self.inflate_freqs[i],
                inflate_style=self.inflate_style,
                nonlocal_freq=self.nonlocal_freqs[i],
                nonlocal_cfg=self.nonlocal_cfg if i in self.nonlocal_stages else None,
                with_cp=with_cp)
            self.inplanes = planes * self.block.expansion
            layer_name = 'layer{}'.format(i + 1)
            self.add_sublayer(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self.feat_dim = self.block.expansion * 64 * 2 ** (
                len(self.stage_blocks) - 1)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        #x = self.relu(x)
        x = fluid.layers.pool3d(input=x, pool_size=(self.pool1_kernel_t, 3, 3), pool_type='max', 
                                pool_stride=(self.pool1_stride_t, 2, 2), pool_padding=(self.pool1_kernel_t //2, 1, 1))
        #self.maxpool = nn.MaxPool3d(kernel_size=(pool1_kernel_t, 3, 3), stride=(pool1_stride_t, 2, 2),
        #                            padding=(pool1_kernel_t // 2, 1, 1))
        #x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        if len(outs) == 1:
            return outs[0]
        else:
            return tuple(outs)

if __name__ == '__main__':
    with fluid.dygraph.guard():
        network = ResNet_I3D(50)
        img = np.zeros([10, 3, 32, 224, 224]).astype('float32')
        img = fluid.dygraph.to_variable(img)
        outs = network(img) #.numpy()
        for i, out in enumerate(outs):
            out = out.numpy()
            print('i')
            print(out.shape)
        #print(outs)
        #print(outs.shape)

