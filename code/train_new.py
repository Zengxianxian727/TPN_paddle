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
        default=None,
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
    parser.add_argument(
        '--use_data_parallel',
        type=ast.literal_eval,
        default=False,
        help='multi GPU training or one gpu training'
    )
    args = parser.parse_args()
    return args


def train(args):
    # parse config
    #place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    if args.use_gpu:
        if args.use_data_parallel:
            place = fluid.CUDAPlace(fluid.dygraph.parallel.Env().dev_id)
        else:
            place = fluid.CUDAPlace(0)
    else:
        fluid.CPUPlace()
    with fluid.dygraph.guard(place):
        config = parse_config(args.config)
        train_config = merge_configs(config, 'train', vars(args))
        print_configs(train_config, 'Train')

        val_config = merge_configs(config, 'valid', vars(args))
        print_configs(val_config, "Valid")

        if args.use_data_parallel:
            strategy = fluid.dygraph.parallel.prepare_context()

        #根据自己定义的网络，声明train_model
        #train_model = TSN1.TSNResNet(layers=train_config['MODEL']['num_layers'], class_dim=train_config['MODEL']['num_classes'], seg_num=train_config['MODEL']['seg_num'])
        train_model = I3D_TPN(config)
        step = train_config.TRAIN.step if train_config.TRAIN.step is not None else int(train_config.TRAIN.all_num / train_config.TRAIN.batch_size)
        print("step for lr decay: %d" % step)
        decay_epoch = train_config.TRAIN.learning_rate_decay_epoch
        learning_rate_decay = train_config.TRAIN.learning_rate_decay
        base_lr = train_config.TRAIN.learning_rate
        bd = [step * e for e in decay_epoch]
        lr = [base_lr * (learning_rate_decay ** i) for i in range(len(bd) + 1)]
        if train_config.TRAIN.optimizer_type == 'SGD':
            opt = fluid.optimizer.Momentum(learning_rate=fluid.layers.piecewise_decay(boundaries=bd, values=lr), 
                                           momentum=train_config.TRAIN.momentum, 
                                           parameter_list=train_model.parameters(), 
                                           use_nesterov=train_config.TRAIN.use_nesterov, 
                                           grad_clip=fluid.clip.GradientClipByNorm(clip_norm=40),
                                           regularization=fluid.regularizer.L2Decay(regularization_coeff=train_config.TRAIN.l2_weight_decay))
        elif train_config.TRAIN.optimizer_type == 'Adam':
            opt = fluid.optimizer.Adam(
                                        learning_rate=fluid.layers.piecewise_decay(boundaries=bd, values=lr),  
                                        regularization=fluid.regularizer.L2Decay(train_config['TRAIN']['l2_weight_decay']),
                                        parameter_list=train_model.parameters(), 
            )

        

        if args.pretrain:
            # 加载上一次训练的模型，继续训练
            model, _ = fluid.dygraph.load_dygraph(args.save_dir + '/tsn_model')
            train_model.load_dict(model)

        if args.use_data_parallel:
            train_model = fluid.dygraph.parallel.DataParallel(train_model, strategy)

        # build model
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

        # get reader
        train_config.TRAIN.batch_size = train_config.TRAIN.batch_size
        train_reader = KineticsReader(args.model_name.upper(), 'train', train_config).create_reader()

        if args.use_data_parallel:
            train_reader = fluid.contrib.reader.distributed_batch_reader(
                train_reader)

        val_reader = KineticsReader(args.model_name.upper(), 'valid', val_config).create_reader()
        print('go to training')
        epochs = args.epoch or train_model.epoch_num()
        acc_history = 0.0
        for i in range(epochs):
            for batch_id, data in enumerate(train_reader()):
                dy_x_data = np.array([x[0] for x in data]).astype('float32')
                y_data = np.array([[x[1]] for x in data]).astype('int64')
                #print(dy_x_data.shape)
                img = fluid.dygraph.to_variable(dy_x_data)
                label = fluid.dygraph.to_variable(y_data)
                label.stop_gradient = True
                
                out, acc, loss_TPN = train_model(img, label)
                
                loss = fluid.layers.softmax_with_cross_entropy(out, label)
                avg_loss = fluid.layers.mean(loss)
                avg_TPN_loss = fluid.layers.mean(loss_TPN)

                all_loss = avg_loss + avg_TPN_loss
                if args.use_data_parallel:
                    print(args.use_data_parallel)
                    all_loss = train_model.scale_loss(all_loss)
                    all_loss.backward()
                    train_model.apply_collective_grads()
                else:
                    all_loss.backward()

                opt.minimize(all_loss)
                train_model.clear_gradients()
                
                
                
                if batch_id % train_config.TRAIN.visual_step == 0:
                    #opt._learning_rate = float(opt._learning_rate) / 10
                    current_lr = opt.current_step_lr()
                    logger.info("Loss at epoch {} step {}: {}, AUX loss: {} acc: {}, current_lr: {}".format(i, batch_id, avg_loss.numpy(), avg_TPN_loss.numpy(), acc.numpy(), current_lr))
                    print("Loss at epoch {} step {}: {}, AUX loss: {}, acc: {}, current_lr: {}".format(i, batch_id, avg_loss.numpy(), avg_TPN_loss.numpy(), acc.numpy(), current_lr))
                    fluid.dygraph.save_dygraph(train_model.state_dict(), args.save_dir + '/I3D_tpn_model')

            print('go to eval')
            acc_list = []
            train_model.eval()
            for batch_id, data in enumerate(tqdm(val_reader())):
                dy_x_data = np.array([x[0] for x in data]).astype('float32')
                y_data = np.array([[x[1]] for x in data]).astype('int64')
                #print(dy_x_data.shape)
                img = fluid.dygraph.to_variable(dy_x_data)
                label = fluid.dygraph.to_variable(y_data)
                label.stop_gradient = True
                out_jpg, acc_jpg, _ = train_model(img, label)
                out = out_jpg
                acc = fluid.layers.accuracy(input=out, label=label)
                #out_jpg, out_flow, acc = val_model(img, flow_img, label)
                acc_list.append(acc.numpy()[0])

            #print("JPG+FLOW验证集准确率为:{}".format(np.mean(acc_list)))
            #print("JPG验证集准确率为:{}".format(np.mean(acc_list_jpg)))
            #print("FLOW验证集准确率为:{}".format(np.mean(acc_list_flow)))
            print("TPN验证集准确率为:%.6f" % (np.mean(acc_list)))
            print("BEST   TPN验证集准确率为:%.6f" % (acc_history))

            if np.mean(acc_list) > acc_history:
                acc_history = np.mean(acc_list)
                fluid.dygraph.save_dygraph(train_model.state_dict(), args.save_dir + '/tpn_best')
                print("TPN BEST验证集准确率为:{}".format(np.mean(acc_list)))

            train_model.train()

        logger.info("Final loss: {}".format(avg_loss.numpy()))
        print("Final loss: {}".format(avg_loss.numpy()))
                


if __name__ == "__main__":
    args = parse_args()
    # check whether the installed paddle is compiled with GPU
    logger.info(args)

    train(args)
