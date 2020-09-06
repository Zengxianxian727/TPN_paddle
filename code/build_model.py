
import paddle.fluid as fluid

from TPN_paddle.model.backbone.resnet_slow import ResNet_I3D
from TPN_paddle.model.neck.TPN import TPN
from TPN_paddle.model.cls_head.cls_head import ClsHead
from TPN_paddle.model.spatial_temporal_modules.smiple_spatial_temporal_module import SimpleSpatialTemporalModule
from TPN_paddle.model.segmental_consensuses.simple_consensus import SimpleConsensus


class I3D_TPN(fluid.dygraph.Layer):

    def __init__(self, config_setting):
        super(I3D_TPN, self).__init__()
        self.config = config_setting

        if 'BACKBONE' in self.config.keys():
            self.backbone = ResNet_I3D(**self.config.BACKBONE)
            if self.config.BACKBONE.pretrained is not None:
                model, _ = fluid.dygraph.load_dygraph(self.config.BACKBONE.pretrained)
                self.backbone.load_dict(model)
                print('loaded pretrained Backbone')
        else:
            self.backbone = None
        if 'NECK' in self.config.keys():
            self.necks = TPN(**self.config.NECK)
        else:
            self.necks = None
        if 'STMODULE' in self.config.keys():
            self.st_network = SimpleSpatialTemporalModule(**self.config.STMODULE)
        else:
            self.st_network = None
        if 'SEGMENTALCONSENSUS' in self.config.keys():
            self.sc_network = SimpleConsensus(**self.config.SEGMENTALCONSENSUS)
        else:
            self.sc_network = None
        
        if 'CLSHEAD' in self.config.keys():
            self.cls_network = ClsHead(**self.config.CLSHEAD)
        else:
            self.cls_network = None

    def forward(self, input, label=None):
        loss_TPN = None
        if len(input.shape) > 5:   # N * L * 3 * T * H * W
            bs = input.shape[0]
            input = fluid.layers.reshape(input, (-1,)+tuple(input.shape[2:]))
            seglen = input.shape[0] / bs
        else:
            bs = input.shape[0]
            seglen = 1


        outs = self.backbone(input) #.numpy()

        if self.necks is not None:
            new_input = outs #[-2:]
            outs, loss_TPN = self.necks(new_input, label)

        if self.st_network is not None:
            new_input = outs
            outs = self.st_network(new_input)

        if self.sc_network is not None:
            new_input = outs
            #print(new_input.shape)
            #print(seglen)
            new_input = fluid.layers.reshape(new_input, (-1, int(seglen),) + tuple(new_input.shape[1:]))
            outs = fluid.layers.squeeze(self.sc_network(new_input), axes=[1])

        if self.cls_network is not None:
            new_input = outs
            outs = self.cls_network(new_input)

        y = outs

        if label is not None:
            #print(y.shape)
            #print(label.shape)
            acc = fluid.layers.accuracy(input=y, label=label)
            return y, acc, loss_TPN
        
        else:
            return y

