import argparse
import torch
import time
import os
import logging

import numpy as np

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def parse_args():
    parser = argparse.ArgumentParser(description="Re-implementation of Singleshotpose by Tekin et al. Training/Validation scripts")

    parser.add_argument('--experiment', type=str, default='cfg/ape.data') # data config
    parser.add_argument('--modelcfg', type=str, default='cfg/yolo-pose.cfg') # network config
    parser.add_argument('--weightfile', type=str, default='cfg/darknet19_448.conv.23') # imagenet initialized weights
    parser.add_argument('--pretrain_num_epochs', type=int, default=15) # how many epoch to pretrain

    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: %(default)d)')

    parser.add_argument('--log-freq', default=25, type=int, help='log frequency during training (default %(default)d)')
    parser.add_argument('--output-dir', default='run', help='path where to save')

    parser.add_argument('--validate-only', action='store_true', default=False,
                        help='Validation mode. Must be used with --weightfile to load model weights.')

    parser.add_argument('--checkname', type=str, default=None, help='set the checkpoint name')
    parser.add_argument('--kw', type=str, default='exp',
                        help='Keyword for the experiment name (default: %(default)s)')
    
    parser.add_argument('--add-graph', action='store_true', default=False,
                        help='Add model graph to tensorboardx (for debugging)')

    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    parser.add_argument('--config', type=str, default=None,
                        help='Load parameters from json file. CLI arguments will override the config file.')

    args = parser.parse_args()

    if args.config is not None:
        import json
        config_params = json.load(open(args.config))
        parser.set_defaults(**config_params)
        args = parser.parse_args()
        args.config = None  # remove config from args.

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            if type(args.gpu_ids) is str:
                args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.checkname is None:
        args.checkname = os.path.basename(args.experiment)

    print(args)

    return args


def create_logger(exp_path, is_val, level=logging.INFO):
    if is_val:
        phase = 'val'
    else:
        phase = 'train'

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}.log'.format(phase, time_str)
    final_log_file = exp_path + os.sep + log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file), format=head)
    logger = logging.getLogger()
    logger.setLevel(level)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    return logger
