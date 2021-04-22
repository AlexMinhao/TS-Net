import argparse
import os
import torch

from ETTH_util.exp.exp_informer import Exp_Informer

parser = argparse.ArgumentParser(description='[Informer] Long Sequences Forecasting')

parser.add_argument('--model', type=str, required=False, default='IDCN', help='model of the experiment')

parser.add_argument('--data', type=str, required=False, default='ETTh2', help='data')
parser.add_argument('--root_path', type=str, default='./ETTH_util/data/ETT/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh2.csv', help='location of the data file')
parser.add_argument('--features', type=str, default='M', help='features [S, M]')
parser.add_argument('--target', type=str, default='OT', help='target feature')

parser.add_argument('--seq_len', type=int, default=24, help='input series length')
parser.add_argument('--pred_len', type=int, default=24, help='predict series length')




parser.add_argument('--label_len', type=int, default=24, help='help series length')
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=128, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=7, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=2, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=256, help='dimension of fcn')
parser.add_argument('--factor', type=int, default=3, help='prob sparse factor')




parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--attn', type=str, default='prob', help='attention [prob, full]')
parser.add_argument('--embed', type=str, default='fixed', help='embedding type [fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')

parser.add_argument('--itr', type=int, default=2, help='each params run iteration')
parser.add_argument('--epochs', type=int, default=10, help='train epochs')  # default=6,
parser.add_argument('--batchSize', type=int, default=64, help='input data batch size')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')  # default=3
parser.add_argument('--lr', type=float, default=0.0001, help='optimizer learning rate')  # default=0.0001,
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')

parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')

parser.add_argument('--share-weight', default=0, type=int, help='share weight or not in attention q,k,v')
parser.add_argument('--temp', default=0, type=int, help='Use temporature weights or not, if false, temp=1')
parser.add_argument('--hidden-size', default=1, type=int, help='hidden channel of module')
parser.add_argument('--INN', default=1, type=int, help='use INN or basic strategy')
parser.add_argument('--kernel', default=5, type=int, help='kernel size')
parser.add_argument('--dilation', default=1, type=int, help='dilation')
parser.add_argument('--window_size', default=12, type=int, help='dilation')

parser.add_argument('--positionalEcoding', type=bool, default=False)
parser.add_argument('--missingRatio', type=float, default=0.4)

# TCN


parser.add_argument('--levels', type=int, default=7,
                    help='# of levels (default: 8)')
parser.add_argument('--nhid', type=int, default=32,
                    help='number of hidden units per layer (default: 30)')
# parser.add_argument('--horizon', type=int, default=24)

args = parser.parse_args()

print(f'Training configs: {args}')

data_parser = {
    'ETTh1': {'data': 'ETTh1.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1]},
    'ETTh2': {'data': 'ETTh2.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1]},
    'ETTm1': {'data': 'ETTm1.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1]},
}
if args.data in data_parser.keys():
    data_info = data_parser[args.data]
    args.data_path = data_info['data']
    args.target = data_info['T']
    args.enc_in, args.dec_in, args.c_out = data_info[args.features]

torch.manual_seed(4321)  # reproducible
torch.cuda.manual_seed_all(4321)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True  # Can change it to False --> default: False
torch.backends.cudnn.enabled = True

Exp = Exp_Informer

for ii in range(args.itr):
    setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_eb{}_{}_{}'.format(args.model, args.data,
                                                                                          args.features,
                                                                                          args.seq_len, args.label_len,
                                                                                          args.pred_len,
                                                                                          args.d_model, args.n_heads,
                                                                                          args.e_layers, args.d_layers,
                                                                                          args.d_ff, args.attn,
                                                                                          args.embed, args.des, ii)

    exp = Exp(args)
    print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    exp.train(setting)

    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting)