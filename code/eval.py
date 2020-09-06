import os
import sys
import time
import argparse
import ast
import logging
import numpy as np
import paddle.fluid as fluid

#from model import TSN1
from build_model import I3D_TPN
from reader_new import KineticsReader
#from reader_new_imageio import KineticsReader
from config import parse_config, merge_configs, print_configs

from tqdm import tqdm

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
        default='kinetics400_tpn_r50f32s2',
        help='path to pretrain weights. None to use default weights path in  ~/.paddle/weights.'
    )
    parser.add_argument(
        '--use_gpu',
        type=ast.literal_eval,
        default=True,
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


def eval(args):
    # parse config
    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    with fluid.dygraph.guard(place):
        config = parse_config(args.config)
        val_config = merge_configs(config, 'valid', vars(args))
        print_configs(val_config, 'Valid')

        #根据自己定义的网络，声明train_model
        #train_model = TSN1.TSNResNet(layers=train_config['MODEL']['num_layers'], class_dim=train_config['MODEL']['num_classes'], seg_num=train_config['MODEL']['seg_num'])
        train_model = I3D_TPN(config)
        #opt = fluid.optimizer.Momentum(0.01, 0.9, parameter_list=train_model.parameters())

        if args.pretrain:
            # 加载上一次训练的模型，继续训练
            model, _ = fluid.dygraph.load_dygraph(args.pretrain)
            train_model.load_dict(model)


        # get reader
        val_reader = KineticsReader(args.model_name.upper(), 'valid', val_config).create_reader()

        print('go to eval')
        train_model.eval()
        acc_list = []
        for batch_id, data in enumerate(tqdm(val_reader())):
            #print(len(data))
            print('eval %d' % batch_id)
            dy_x_data = np.array([x[0] for x in data]).astype('float32')
            y_data = np.array([[x[1]] for x in data]).astype('int64')
            img = fluid.dygraph.to_variable(dy_x_data)
            label = fluid.dygraph.to_variable(y_data)
            label.stop_gradient = True
            out_jpg, acc_jpg, _ = train_model(img, label)
            out = out_jpg
            acc = fluid.layers.accuracy(input=out, label=label)
            #print(acc)
            acc_list.append(acc.numpy()[0])

        print("TPN: %.6f" % np.mean(acc_list))
                


if __name__ == "__main__":
    args = parse_args()
    # check whether the installed paddle is compiled with GPU
    logger.info(args)

    eval(args)
