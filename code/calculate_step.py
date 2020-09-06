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
from config import parse_config, merge_configs, print_configs

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


def train(args):
    # parse config
    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    with fluid.dygraph.guard(place):
        config = parse_config(args.config)
        train_config = merge_configs(config, 'train', vars(args))
        print_configs(train_config, 'Train')

        #根据自己定义的网络，声明train_model
        #train_model = TSN1.TSNResNet(layers=train_config['MODEL']['num_layers'], class_dim=train_config['MODEL']['num_classes'], seg_num=train_config['MODEL']['seg_num'])
        train_model = I3D_TPN(config)
        if train_config.TRAIN.optimizer_type == 'SGD':
            opt = fluid.optimizer.Momentum(learning_rate=train_config.TRAIN.learning_rate, 
                                           momentum=train_config.TRAIN.momentum, 
                                           parameter_list=train_model.parameters(), 
                                           use_nesterov=train_config.TRAIN.use_nesterov, 
                                           grad_clip=fluid.clip.GradientClipByNorm(clip_norm=40),
                                           regularization=fluid.regularizer.L2Decay(regularization_coeff=train_config.TRAIN.l2_weight_decay))
        elif train_config.TRAIN.optimizer_type == 'Adam':
            opt = fluid.optimizer.Adam(
                                        learning_rate=train_config['TRAIN']['learning_rate'],  
                                        regularization=fluid.regularizer.L2Decay(train_config['TRAIN']['l2_weight_decay']),
                                        parameter_list=train_model.parameters(), 
            )

        if args.pretrain:
            # 加载上一次训练的模型，继续训练
            model, _ = fluid.dygraph.load_dygraph(args.save_dir + '/tsn_model')
            train_model.load_dict(model)

        # build model
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

        # get reader
        train_config.TRAIN.batch_size = train_config.TRAIN.batch_size
        train_reader = KineticsReader(args.model_name.upper(), 'train', train_config).create_reader()
        print('go to training')
        epochs = args.epoch or train_model.epoch_num()
        for i in range(epochs):
            for batch_id, data in enumerate(train_reader()):
                k =0
                if batch_id % 10 == 1:
                    print("epoch {} step {}".format(i, batch_id))

            print(batch_id)
                


if __name__ == "__main__":
    args = parse_args()
    # check whether the installed paddle is compiled with GPU
    logger.info(args)

    train(args)
