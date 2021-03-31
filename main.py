import os
import torch
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from datetime import datetime
from models.handler import train, test,retrain
import argparse
import pandas as pd
import numpy as np
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--train', type=bool, default=True)
parser.add_argument('--evaluate', type=bool, default=False)
parser.add_argument('--finetune', type=bool, default=False)
parser.add_argument('--dataset', type=str, default='PEMS08')  #PeMS07
parser.add_argument('--window_size', type=int, default=12)
parser.add_argument('--horizon', type=int, default=12)
parser.add_argument('--train_length', type=float, default=6)
parser.add_argument('--valid_length', type=float, default=2)
parser.add_argument('--test_length', type=float, default=2)
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--multi_layer', type=int, default=5)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--validate_freq', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--norm_method', type=str, default='z_score')
parser.add_argument('--optimizer', type=str, default='N') #
parser.add_argument('--early_stop', type=bool, default=False)
parser.add_argument('--exponential_decay_step', type=int, default=5)
parser.add_argument('--decay_rate', type=float, default=0.5)
parser.add_argument('--dropout_rate', type=float, default=0.5)
parser.add_argument('--leakyrelu_rate', type=int, default=0.2)
parser.add_argument('--lradj', type=int, default=1,help='adjust learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-5)
# Action Part

parser.add_argument('--input_dim', type=int, default=170)################
parser.add_argument('--num_stacks', type=int, default=1)
parser.add_argument('--share-weight', default=0, type=int, help='share weight or not in attention q,k,v')
parser.add_argument('--temp', default=0, type=int, help='Use temporature weights or not, if false, temp=1')
parser.add_argument('--hidden-size', default=1, type=float, help='hidden channel of module')###################################
parser.add_argument('--INN', default=1, type=int, help='use INN or basic strategy')###########################
parser.add_argument('--head_size', default=16, type=int, help='hidden channel of module')
parser.add_argument('--kernel', default=3, type=int, help='kernel size')
parser.add_argument('--dilation', default=1, type=int, help='dilation')


args = parser.parse_args()
print(f'Training configs: {args}')
data_file = os.path.join('dataset', args.dataset + '.npz')
result_train_file = os.path.join('output', args.dataset, 'train')
result_test_file = os.path.join('output', args.dataset, 'test')

data = np.load(data_file,allow_pickle=True)
data = data['data'][:,:,0]


# 07  12671  228
# 03          358
# 04  16992 307 3
# 08   170
# split data
train_ratio = args.train_length / (args.train_length + args.valid_length + args.test_length)
valid_ratio = args.valid_length / (args.train_length + args.valid_length + args.test_length)
test_ratio = 1 - train_ratio - valid_ratio
train_data = data[:int(train_ratio * len(data))]
valid_data = data[int(train_ratio * len(data)):int((train_ratio + valid_ratio) * len(data))]
test_data = data[int((train_ratio + valid_ratio) * len(data)):]

torch.manual_seed(0)
if __name__ == '__main__':
    torch.manual_seed(4321)  # reproducible
    torch.cuda.manual_seed_all(4321)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True  # Can change it to False --> default: False
    torch.backends.cudnn.enabled = True
    writer = SummaryWriter('./run/exp1')
    if args.train:
        try:
            before_train = datetime.now().timestamp()
            _, normalize_statistic = train(train_data, valid_data, test_data, args, result_train_file, writer)
            after_train = datetime.now().timestamp()
            print(f'Training took {(after_train - before_train) / 60} minutes')
        except KeyboardInterrupt:
            print('-' * 99)
            print('Exiting from training early')
    #
    # if args.evaluate:
    #
    #     before_evaluation = datetime.now().timestamp()
    #     test(test_data, train_data, args, result_train_file, result_test_file, epoch = 5)
    #     after_evaluation = datetime.now().timestamp()
    #     print(f'Evaluation took {(after_evaluation - before_evaluation) / 60} minutes')

    # if args.finetune:
    #
    #     before_evaluation = datetime.now().timestamp()
    #     _, normalize_statistic = retrain(train_data, test_data, args, result_train_file, epoch =52)
    #     after_train = datetime.now().timestamp()
    #     print(f'Training took {(after_train - before_train) / 60} minutes')
    #
    # print('done')
