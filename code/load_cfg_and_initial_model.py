import argparse
import ast
import logging
import paddle.fluid as fluid
import numpy as np

#from config import parse_config, merge_configs, print_configs
from config import parse_config, merge_configs, print_configs
from TPN_paddle.model.backbone.resnet_slow import ResNet_I3D
from TPN_paddle.model.neck.TPN import TPN
from TPN_paddle.model.cls_head.cls_head import ClsHead
from TPN_paddle.model.spatial_temporal_modules.smiple_spatial_temporal_module import SimpleSpatialTemporalModule
from TPN_paddle.model.segmental_consensuses.simple_consensus import SimpleConsensus

logging.root.handlers = []
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(filename='logger.log', level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser("Paddle Video train script")
    parser.add_argument(
        '--model_name',
        type=str,
        default='tpn',
        help='name of model to train.')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/TPN_kinetics_new.txt',
        help='path to config file of model')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='training batch size. None to use config file setting.')
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=None,
        help='learning rate use for training. None to use config file setting.')
    parser.add_argument(
        '--pretrain',
        type=str,
        default=None,
        help='path to pretrain weights. None to use default weights path in  ~/.paddle/weights.'
    )
    parser.add_argument(
        '--use_gpu',
        type=ast.literal_eval,
        default=False,
        help='default use gpu.')
    parser.add_argument(
        '--epoch',
        type=int,
        default=100,
        help='epoch number, 0 for read from config file')
    parser.add_argument(
        '--save_dir',
        type=str,
        default='checkpoints_models',
        help='directory name to save train snapshoot')
    args = parser.parse_args()
    return args


args = parse_args()
#cfg = load('work/configs/TPN_kinetics.py')
print(args)

config = parse_config(args.config)
#print(config.NECK.temporal_modulation_config.scales)
#print('ABC' in config.keys())
#print(config.NECK)
#print_configs(config,'Train')

with fluid.dygraph.guard():
    if 'BACKBONE' in config.keys():
        network = ResNet_I3D(**config.BACKBONE)
    else:
        network = None
    if 'NECK' in config.keys():
        neck_network = TPN(**config.NECK)
    else:
        neck_network = None
    if 'STMODULE' in config.keys():
        st_network = SimpleSpatialTemporalModule(**config.STMODULE)
    else:
        st_network = None
    if 'SEGMENTALCONSENSUS' in config.keys():
        sc_network = SimpleConsensus(**config.SEGMENTALCONSENSUS)
    else:
        sc_network = None
    
    if 'CLSHEAD' in config.keys():
        cls_network = ClsHead(**config.CLSHEAD)
    else:
        cls_network = None
    
    img = np.zeros([5, 3, 32, 224, 224]).astype('float32')
    if len(img.shape) > 5:   # N * L * 3 * T * H * W
        bs = img.shape[0]
        img = fluid.layers.reshape(img, (-1,)+tuple(img.shape[2:]))
        seglen = img.shape[0] / bs
    else:
        bs = img.shape[0]
        seglen = 1
    img = fluid.dygraph.to_variable(img)
    outs = network(img) #.numpy()
    #print(outs[0].shape)
    for i, out in enumerate(outs):
        out = out.numpy()
        print('%d',i)
        print(out.shape)

    new_input = outs #[-2:]

    outs_TPN = neck_network(new_input)
    #print(outs_TPN.numpy().shape)
    print(outs_TPN[0].shape)

    outs = outs_TPN[0]

    if st_network is not None:
        new_input = outs
        outs_ST = st_network(new_input)
        print('ST_output')
        print(outs_ST.shape)
        outs = outs_ST

    if sc_network is not None:
        new_input = outs
        new_input = fluid.layers.reshape(new_input, (-1, int(seglen),) + tuple(new_input.shape[1:]))
        
        outs_SC = sc_network(new_input)
        print('SC_output')
        print(outs_SC.shape)
        outs = fluid.layers.squeeze(outs_SC, axes=[1])


    new_input = outs
    outs_CLS = cls_network(new_input)
    print(outs_CLS.shape)
        





