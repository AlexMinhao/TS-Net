import os
import torch
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from datetime import datetime
from models.handler import train, trainSemi, trainEco2Deco, test,retrain
import argparse
import pandas as pd
import numpy as np
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--train', type=bool, default=True)
parser.add_argument('--evaluate', type=bool, default=True)
parser.add_argument('--finetune', type=bool, default=False)
parser.add_argument('--dataset', type=str, default='PEMS08')  #PeMS07
parser.add_argument('--window_size', type=int, default=12)
parser.add_argument('--horizon', type=int, default=12)
parser.add_argument('--train_length', type=float, default=6)
parser.add_argument('--valid_length', type=float, default=2)
parser.add_argument('--test_length', type=float, default=2)
parser.add_argument('--epoch', type=int, default=80)
parser.add_argument('--lr', type=float, default=0.001)

parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--validate_freq', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--norm_method', type=str, default='z_score')
parser.add_argument('--optimizer', type=str, default='N') #
parser.add_argument('--early_stop', type=bool, default=False)
parser.add_argument('--exponential_decay_step', type=int, default=5)
parser.add_argument('--decay_rate', type=float, default=0.5)

parser.add_argument('--lradj', type=int, default=6,help='adjust learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--model_name', type=str, default='Normal')
# Action Part
parser.add_argument('--input_dim', type=int, default=170)################
parser.add_argument('--num_stacks', type=int, default=1)

parser.add_argument('--hidden-size', default=1, type=float, help='hidden channel of module')###################################
parser.add_argument('--INN', default=1, type=int, help='use INN or basic strategy')###########################

parser.add_argument('--kernel', default=5, type=int, help='kernel size')
parser.add_argument('--dilation', default=1, type=int, help='dilation')
parser.add_argument('--positionalEcoding', type = bool , default=True)

args = parser.parse_args()
print(f'Training configs: {args}')
data_file = os.path.join('dataset', args.dataset + '.npz')
result_train_file = os.path.join('output', args.dataset, 'train')
result_test_file = os.path.join('output', args.dataset, 'test')

data = np.load(data_file,allow_pickle=True)
data = data['data'][:,:,0]


# 07M  12671   228
# 03  26208   358
# 04  16992   307
# 07  28224   883
# 08  17856   170
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
    writer = SummaryWriter('./run/{}_ReOrder'.format(args.model_name))
    if args.train:
        try:
            before_train = datetime.now().timestamp()
            if args.model_name == "Semi":
                print("===================Semi-Start=========================")
                _, normalize_statistic = trainSemi(train_data, valid_data, test_data, args, result_train_file, writer)
                after_train = datetime.now().timestamp()
                print(f'Training took {(after_train - before_train) / 60} minutes')
                print("===================Semi-End=========================")
            elif args.model_name == "TwoDecoder":
                print("===================TwoDecoder-Start=========================")
                _, normalize_statistic = trainEco2Deco(train_data, valid_data, test_data, args, result_train_file, writer)
                after_train = datetime.now().timestamp()
                print(f'Training took {(after_train - before_train) / 60} minutes')
                print("===================TwoDecoder-End=========================")
            else:
                print("===================Normal-Start=========================")
                _, normalize_statistic = train(train_data, valid_data, test_data, args, result_train_file, writer)
                after_train = datetime.now().timestamp()
                print(f'Training took {(after_train - before_train) / 60} minutes')
                print("===================Normal-End=========================")
        except KeyboardInterrupt:
            print('-' * 99)
            print('Exiting from training early')
    #
    if args.evaluate:

        before_evaluation = datetime.now().timestamp()
        test(test_data, train_data, args, result_train_file, result_test_file, epoch = None)
        after_evaluation = datetime.now().timestamp()
        print(f'Evaluation took {(after_evaluation - before_evaluation) / 60} minutes')

    # if args.finetune:
    #
    #     before_evaluation = datetime.now().timestamp()
    #     _, normalize_statistic = retrain(train_data, test_data, args, result_train_file, epoch =52)
    #     after_train = datetime.now().timestamp()
    #     print(f'Training took {(after_train - before_train) / 60} minutes')
    #
    # print('done')
