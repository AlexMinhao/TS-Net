import os
import torch
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from datetime import datetime
from models.handler import train, trainSemi, trainEco2Deco, test, trainOverLap, retrain
import argparse
import pandas as pd
import numpy as np
from data_loader.forecast_dataloader import ForecastDataset, de_normalized
from utils.loss import smooth_l1_loss

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
parser.add_argument('--epoch', type=int, default=50)
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


def validate(model, epoch, forecast_loss, dataloader, device, normalize_method, statistic,
             node_cnt, window_size, horizon, writer,
             result_file=None, test=False):
    start = datetime.now()
    print("===================Validate Normal=========================")
    forecast_norm, target_norm, mid_norm, input_norm = inference(model, dataloader, device,
                                                                 node_cnt, window_size, horizon)
    if normalize_method and statistic:
        forecast = de_normalized(forecast_norm, normalize_method, statistic)
        target = de_normalized(target_norm, normalize_method, statistic)
        mid = de_normalized(mid_norm, normalize_method, statistic)
        input = de_normalized(input_norm, normalize_method, statistic)
    else:
        forecast, target = forecast_norm, target_norm
        forecast, target, mid = forecast_norm, target_norm, mid_norm
        forecast, target, mid, input = forecast_norm, target_norm, mid_norm, input_norm

    beta = 0.1
    forecast_norm = torch.from_numpy(forecast_norm).float()
    mid_norm = torch.from_numpy(mid_norm).float()
    target_norm = torch.from_numpy(target_norm).float()

    loss = forecast_loss(forecast_norm, target_norm) + forecast_loss(mid_norm, target_norm)
    loss_F = forecast_loss(forecast_norm, target_norm)
    loss_M = forecast_loss(mid_norm, target_norm)

    # score = evaluate(target, forecast)
    score = evaluate(target, forecast)
    score1 = evaluate(target, mid)
    score_final_detail = evaluate(target, forecast, by_step=True)
    print('by step:MAPE&MAE&RMSE', score_final_detail)
    end = datetime.now()

    if test:
        print(f'TEST: RAW : MAE {score[1]:7.2f};MAPE {score[0]:7.2f}; RMSE {score[2]:7.2f}.')
        print(f'TEST: RAW-Mid : MAE {score1[1]:7.2f}; MAPE {score[0]:7.2f}; RMSE {score1[2]:7.2f}.')

        writer.add_scalar('Test MAE_final', score[1], global_step=epoch)
        writer.add_scalar('Test MAE_Mid', score1[1], global_step=epoch)
        writer.add_scalar('Test RMSE_final', score[2], global_step=epoch)
        writer.add_scalar('Test RMSE_Mid', score1[2], global_step=epoch)

        writer.add_scalar('Test Loss_final', loss_F, global_step=epoch)
        writer.add_scalar('Test Loss_Mid', loss_M, global_step=epoch)


    else:
        print(f'VAL: RAW : MAE {score[1]:7.2f}; RMSE {score[2]:7.2f}.')
        print(f'VAL: RAW-Mid : MAE {score1[1]:7.2f}; RMSE {score1[2]:7.2f}.')
        writer.add_scalar('VAL MAE_final', score[1], global_step=epoch)
        writer.add_scalar('VAL MAE_Mid', score1[1], global_step=epoch)
        writer.add_scalar('VAL RMSE_final', score[2], global_step=epoch)
        writer.add_scalar('VAL RMSE_Mid', score1[2], global_step=epoch)

        writer.add_scalar('VAL Loss_final', loss_F, global_step=epoch)
        writer.add_scalar('VAL Loss_Mid', loss_M, global_step=epoch)

    if result_file:
        if not os.path.exists(result_file):
            os.makedirs(result_file)
        step_to_print = 0
        forcasting_2d = forecast[:, step_to_print, :]
        forcasting_2d_target = target[:, step_to_print, :]

        np.savetxt(f'{result_file}/target.csv', forcasting_2d_target, delimiter=",")
        np.savetxt(f'{result_file}/predict.csv', forcasting_2d, delimiter=",")
        np.savetxt(f'{result_file}/predict_abs_error.csv',
                   np.abs(forcasting_2d - forcasting_2d_target), delimiter=",")
        np.savetxt(f'{result_file}/predict_ape.csv',
                   np.abs((forcasting_2d - forcasting_2d_target) / forcasting_2d_target), delimiter=",")

    return dict(mae=score[1], mape=score[0], rmse=score[2])


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 1:
        lr_adjust = {epoch: args.lr * (0.95 ** ((epoch - 1) // 1))}
    elif args.lradj == 2:
        # lr_adjust = {
        #     2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
        #     10: 5e-7, 15: 1e-7, 20: 5e-8
        # }
        lr_adjust = {
            20: 0.0005, 40: 0.0001, 60: 0.00005, 80: 0.00001

        }
    elif args.lradj == 3:
        # lr_adjust = {
        #     2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
        #     10: 5e-7, 15: 1e-7, 20: 5e-8
        # }
        lr_adjust = {
            20: 0.0005, 25: 0.0001, 35: 0.00005, 55: 0.00001
            , 70: 0.000001
        }
    elif args.lradj == 4:
        # lr_adjust = {
        #     2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
        #     10: 5e-7, 15: 1e-7, 20: 5e-8
        # }
        lr_adjust = {
            30: 0.0005, 40: 0.0003, 50: 0.0001, 65: 0.00001
            , 80: 0.000001
        }
    elif args.lradj == 5:
        # lr_adjust = {
        #     2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
        #     10: 5e-7, 15: 1e-7, 20: 5e-8
        # }
        lr_adjust = {
            40: 0.0001, 60: 0.00005
        }
    elif args.lradj == 6:
        # lr_adjust = {
        #     2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
        #     10: 5e-7, 15: 1e-7, 20: 5e-8
        # }
        lr_adjust = {
            0: 0.0001, 5: 0.0005, 10: 0.001, 20: 0.0001, 30: 0.00005, 40: 0.00001
            , 70: 0.000001
        }
    elif args.lradj == 61:
        # lr_adjust = {
        #     2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
        #     10: 5e-7, 15: 1e-7, 20: 5e-8
        # }
        lr_adjust = {
            0: 0.0001, 5: 0.0005, 10: 0.001, 25: 0.0005, 35: 0.0001, 50: 0.00001
            , 70: 0.000001
        }

    elif args.lradj == 7:
        # lr_adjust = {
        #     2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
        #     10: 5e-7, 15: 1e-7, 20: 5e-8
        # }
        lr_adjust = {
            10: 0.0001, 30: 0.00005, 50: 0.00001
            , 70: 0.000001
        }

    elif args.lradj == 8:
        # lr_adjust = {
        #     2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
        #     10: 5e-7, 15: 1e-7, 20: 5e-8
        # }
        lr_adjust = {
            0: 0.0005, 5: 0.0008, 10: 0.001, 20: 0.0001, 30: 0.00005, 40: 0.00001
            , 70: 0.000001
        }
    elif args.lradj == 9:
        # lr_adjust = {
        #     2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
        #     10: 5e-7, 15: 1e-7, 20: 5e-8
        # }
        lr_adjust = {
            0: 0.0001, 10: 0.0005, 20: 0.001, 40: 0.0001, 45: 0.00005, 50: 0.00001
            , 70: 0.000001
        }

    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


def train(train_data, valid_data, test_data, args, result_file, writer):
    node_cnt = train_data.shape[1]

    print("===================Train Normal=========================")
    part = [[1, 1], [0, 0], [0, 0]]

    print('level number {}, level details: {}'.format(len(part), part))
    # model = WASN(args, num_classes=args.horizon, num_stacks = args.num_stacks, first_conv = args.input_dim,
    #                   number_levels=len(part),
    #                   number_level_part=part)

    model = IDCNet(args, num_classes=args.horizon, input_len=args.window_size, input_dim=args.input_dim,
                   number_levels=len(part),
                   number_level_part=part)

    print('Parameters of need to grad is:{} M'.format(count_params(model) / 1000000.0))
    in1 = torch.randn(args.batch_size, args.window_size, args.input_dim)
    flops, params = profile(model, inputs=(in1,))
    macs, params = clever_format([flops, params], "%.3f")
    print('MACs: {}, Parameters: {}'.format(macs, params))
    #    print_model_parm_flops(model)
    model.to(args.device)
    if len(train_data) == 0:
        raise Exception('Cannot organize enough training data')
    if len(valid_data) == 0:
        raise Exception('Cannot organize enough validation data')
    if len(test_data) == 0:
        raise Exception('Cannot organize enough test data')
    if args.norm_method == 'z_score':
        train_mean = np.mean(train_data, axis=0)
        train_std = np.std(train_data, axis=0)
        normalize_statistic = {"mean": train_mean.tolist(), "std": train_std.tolist()}
    elif args.norm_method == 'min_max':
        train_min = np.min(train_data, axis=0)
        train_max = np.max(train_data, axis=0)
        normalize_statistic = {"min": train_min, "max": train_max}
    else:
        normalize_statistic = None

    if args.optimizer == 'RMSProp':
        my_optim = torch.optim.RMSprop(params=model.parameters(), lr=args.lr, eps=1e-08)
    else:
        my_optim = torch.optim.Adam(params=model.parameters(), lr=args.lr, betas=(0.9, 0.999),
                                    weight_decay=args.weight_decay)
        # my_optim = torch.optim.AdamW(params=model.parameters(), lr=args.lr)

    my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=my_optim, gamma=args.decay_rate)

    train_set = ForecastDataset(train_data, window_size=args.window_size, horizon=args.horizon,
                                normalize_method=args.norm_method, norm_statistic=normalize_statistic)
    valid_set = ForecastDataset(valid_data, window_size=args.window_size, horizon=args.horizon,
                                normalize_method=args.norm_method, norm_statistic=normalize_statistic)
    test_set = ForecastDataset(test_data, window_size=args.window_size, horizon=args.horizon,
                               normalize_method=args.norm_method, norm_statistic=normalize_statistic)
    train_loader = torch_data.DataLoader(train_set, batch_size=args.batch_size, drop_last=False, shuffle=True,
                                         num_workers=1)
    valid_loader = torch_data.DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=1)
    test_loader = torch_data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=1)

    # forecast_loss = nn.MSELoss(reduction='mean').to(args.device)
    forecast_loss = nn.L1Loss().to(args.device)
    #    forecast_loss = nn.SmoothL1Loss().to(args.device)
    #    forecast_loss =  smooth_l1_loss
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        total_params += param
    print(f"Total Trainable Params: {total_params}")

    best_validate_mae = np.inf
    best_test_mae = np.inf
    validate_score_non_decrease_count = 0

    performance_metrics = {}
    for epoch in range(args.epoch):

        adjust_learning_rate(my_optim, epoch, args)
        epoch_start_time = time.time()
        model.train()
        loss_total = 0
        loss_total_F = 0
        loss_total_M = 0
        cnt = 0
        for i, (inputs, target) in enumerate(train_loader):
            inputs = inputs.to(args.device)  # torch.Size([32, 12, 228])
            target = target.to(args.device)  # torch.Size([32, 3, 228])
            model.zero_grad()
            forecast, res = model(inputs)
            loss = forecast_loss(forecast, target) + forecast_loss(res, target)
            loss_F = forecast_loss(forecast, target)
            loss_M = forecast_loss(res, target)
            # beta = 0.1 #for the threshold of the smooth L1 loss
            # loss = forecast_loss(forecast, target, beta) + forecast_loss(res, target, beta)
            # loss_F = forecast_loss(forecast, target, beta)
            # loss_M = forecast_loss(res, target, beta)
            cnt += 1
            loss.backward()
            my_optim.step()
            loss_total += float(loss)
            loss_total_F += float(loss_F)
            loss_total_M += float(loss_M)

        print(
            '| end of epoch {:3d} | time: {:5.2f}s | train_total_loss {:5.4f}, loss_F {:5.4f}, loss_M {:5.4f}  '.format(
                epoch, (
                        time.time() - epoch_start_time), loss_total / cnt, loss_total_F / cnt, loss_total_M / cnt))

        writer.add_scalar('Train_loss_tatal', loss_total / cnt, global_step=epoch)
        writer.add_scalar('Train_loss_Mid', loss_total_F / cnt, global_step=epoch)
        writer.add_scalar('Train_loss_Final', loss_total_M / cnt, global_step=epoch)

        # save_model(model, result_file, epoch)
        if (epoch + 1) % args.exponential_decay_step == 0:
            my_lr_scheduler.step()
        if (epoch + 1) % args.validate_freq == 0:
            is_best_for_now = False
            print('------ validate on data: VALIDATE ------')
            performance_metrics = \
                validate(model, epoch, forecast_loss, valid_loader, args.device, args.norm_method, normalize_statistic,
                         node_cnt, args.window_size, args.horizon,
                         writer, result_file=None, test=False)
            test_metrics = validate(model, epoch, forecast_loss, test_loader, args.device, args.norm_method,
                                    normalize_statistic,
                                    node_cnt, args.window_size, args.horizon,
                                    writer, result_file=None, test=True)
            if best_validate_mae > performance_metrics['mae']:
                best_validate_mae = performance_metrics['mae']
                is_best_for_now = True
                validate_score_non_decrease_count = 0
                print('got best validation result:', performance_metrics, test_metrics)
            else:
                validate_score_non_decrease_count += 1
            if best_test_mae > test_metrics['mae']:
                best_test_mae = test_metrics['mae']
                print('got best test result:', test_metrics)

            # save model
            if is_best_for_now:
                save_model(model, result_file)
            # if epoch%4==0:
            #     save_model(model, result_file,epoch=epoch)
        # early stop
        if args.early_stop and validate_score_non_decrease_count >= args.early_stop_step:
            break
    return performance_metrics, normalize_statistic


def test(test_data, train_data, args, result_train_file, result_test_file, epoch):
    # with open(os.path.join(result_train_file, 'norm_stat.json'),'r') as f:
    #     normalize_statistic = json.load(f)

    test_mean = np.mean(train_data, axis=0)
    test_std = np.std(train_data, axis=0)
    normalize_statistic = {"mean": test_mean.tolist(), "std": test_std.tolist()}

    forecast_loss = smooth_l1_loss  # nn.MSELoss(reduction='mean').to(args.device)
    model = load_model(result_train_file, epoch=epoch)
    node_cnt = test_data.shape[1]
    test_set = ForecastDataset(test_data, window_size=args.window_size, horizon=args.horizon,
                               normalize_method=args.norm_method, norm_statistic=normalize_statistic)
    test_loader = torch_data.DataLoader(test_set, batch_size=args.batch_size, drop_last=False,
                                        shuffle=False, num_workers=0)
    performance_metrics = validate(model=model, epoch=100, forecast_loss=forecast_loss, dataloader=test_loader,
                                   device=args.device, normalize_method=args.norm_method, statistic=normalize_statistic,
                                   node_cnt=node_cnt, window_size=args.window_size, horizon=args.horizon,
                                   result_file=result_test_file, writer=None, test=True)
    mae, rmse, mape = performance_metrics['mae'], performance_metrics['rmse'], performance_metrics['mape']
    print('Performance on test set: | MAE: {:5.2f} | MAPE: {:5.2f} | RMSE: {:5.4f}'.format(mae, mape, rmse))

    # model, forecast_loss, dataloader, device, normalize_method











if __name__ == '__main__':
    torch.manual_seed(4321)  # reproducible
    torch.cuda.manual_seed_all(4321)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True  # Can change it to False --> default: False
    torch.backends.cudnn.enabled = True
    writer = SummaryWriter('./run/{}_Atten_cat'.format(args.model_name))
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
            elif args.model_name == "OverLap":
                print("===================OverLap-Start=========================")
                _, normalize_statistic = trainOverLap(train_data, valid_data, test_data, args, result_train_file, writer)
                after_train = datetime.now().timestamp()
                print(f'Training took {(after_train - before_train) / 60} minutes')
                print("===================OverLap-End=========================")
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
