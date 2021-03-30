import json
from datetime import datetime

from data_loader.forecast_dataloader import ForecastDataset, de_normalized
from models.base_model import Model
import torch
import torch.nn as nn
import torch.utils.data as torch_data
import numpy as np
import time
import os

from models.StackTWaveNetTransformerEncoder import WASN

from utils.math_utils import evaluate
from thop import profile, clever_format
from utils.flops import print_model_parm_flops

from utils.loss import smooth_l1_loss

def save_model(model, model_dir, epoch=None):
    if model_dir is None:
        return
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    epoch = str(epoch) if epoch else ''
    file_name = os.path.join(model_dir, epoch + 'Final_best08STWN.pt')
    with open(file_name, 'wb') as f:
        torch.save(model, f)


def load_model(model_dir, epoch=None):
    if not model_dir:
        return
    epoch = str(epoch) if epoch else ''
    file_name = os.path.join(model_dir, epoch + 'Final_best08STWN.pt')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(file_name):
        return
    with open(file_name, 'rb') as f:
        model = torch.load(f)
    return model

def count_params(model, ):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def inference(model, dataloader, device, node_cnt, window_size, horizon):
    forecast_set = []
    Mid_set = []
    target_set = []
    input_set = []
    model.eval()
    with torch.no_grad():
        for i, (inputs, target) in enumerate(dataloader):
            inputs = inputs.to(device)
            target = target.to(device)
            step = 0
            forecast_steps = np.zeros([inputs.size()[0], horizon, node_cnt], dtype=np.float)
            Mid_steps = np.zeros([inputs.size()[0], horizon, node_cnt], dtype=np.float)
            while step < horizon:
                forecast_result, Mid_result = model(inputs)
                len_model_output = forecast_result.size()[1]
                if len_model_output == 0:
                    raise Exception('Get blank inference result')
                inputs[:, :window_size - len_model_output, :] = inputs[:, len_model_output:window_size,
                                                                   :].clone()
                inputs[:, window_size - len_model_output:, :] = forecast_result.clone()
                forecast_steps[:, step:min(horizon - step, len_model_output) + step, :] = \
                    forecast_result[:, :min(horizon - step, len_model_output), :].detach().cpu().numpy()

                Mid_steps[:, step:min(horizon - step, len_model_output) + step, :] = \
                    Mid_result[:, :min(horizon - step, len_model_output), :].detach().cpu().numpy()

                step += min(horizon - step, len_model_output)
            forecast_set.append(forecast_steps)
            Mid_set.append(Mid_steps)
            target_set.append(target.detach().cpu().numpy())
            input_set.append(inputs.detach().cpu().numpy())

    return np.concatenate(forecast_set, axis=0), np.concatenate(target_set, axis=0),np.concatenate(Mid_set, axis=0), np.concatenate(input_set, axis=0),


def validate(model, forecast_loss, dataloader, device, normalize_method, statistic,
             node_cnt, window_size, horizon,
             result_file=None,test=False):
    start = datetime.now()

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

    # score = evaluate(target, forecast)
    score = evaluate(target, forecast)
    score1 = evaluate(target, mid)


    end = datetime.now()


    if test:
        print(f'TEST: RAW : MAE {score[1]:7.2f}; RMSE {score[2]:7.2f}.')
        print(f'TEST: RAW-Mid : MAE {score1[1]:7.2f}; RMSE {score1[2]:7.2f}.')
    else:
        print(f'VAL: RAW : MAE {score[1]:7.2f}; RMSE {score[2]:7.2f}.')
        print(f'VAL: RAW-Mid : MAE {score1[1]:7.2f}; RMSE {score1[2]:7.2f}.')

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

    return dict(mae=score[1], rmse=score[2])


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj== 1:
        lr_adjust = {epoch: args.lr * (0.95 ** ((epoch-1) // 5))}
    elif args.lradj==2:
        # lr_adjust = {
        #     2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
        #     10: 5e-7, 15: 1e-7, 20: 5e-8
        # }
        lr_adjust = {
            20: 0.0005, 40: 0.0001, 60: 0.00005, 80: 0.00001

        }
    elif args.lradj==3:
        # lr_adjust = {
        #     2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
        #     10: 5e-7, 15: 1e-7, 20: 5e-8
        # }
        lr_adjust = {
            20: 0.0005, 25: 0.0001, 35: 0.00005, 55: 0.00001
            , 70: 0.000001
        }
    elif args.lradj==4:
        # lr_adjust = {
        #     2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
        #     10: 5e-7, 15: 1e-7, 20: 5e-8
        # }
        lr_adjust = {
            30: 0.0005, 40: 0.0003, 50: 0.0001, 65: 0.00001
            , 80: 0.000001
        }
    elif args.lradj==5:
        # lr_adjust = {
        #     2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
        #     10: 5e-7, 15: 1e-7, 20: 5e-8
        # }
        lr_adjust = {
            40: 0.0001, 60: 0.00005
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))

def train(train_data, valid_data, test_data, args, result_file):
    node_cnt = train_data.shape[1]
    model = Model(node_cnt, 2, args.window_size, args.multi_layer, horizon=args.horizon)
    # part = [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]  # Best model
    # part = [[1, 1], [1,1], [1,1], [0, 0], [0, 0], [0, 0], [0, 0]]  # Best model

    part = [[1, 1], [0, 0], [0, 0]]
    # # part = [[0, 1], [0, 0]]
    # part = [[0, 0]]
    print('level number {}, level details: {}'.format(len(part), part))
    model = WASN(args, num_classes=args.horizon, num_stacks = args.num_stacks, first_conv = args.input_dim,
                      number_levels=len(part),
                      number_level_part=part,
                      haar_wavelet=False)

    print('Parameters of need to grad is:{} M'.format(count_params(model) / 1000000.0))
    in1 = torch.randn(8,12,170)
    flops, params = profile(model, inputs=(in1, ))
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
        my_optim = torch.optim.Adam(params=model.parameters(), lr=args.lr, betas=(0.9, 0.999))
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

#    forecast_loss = nn.MSELoss(reduction='mean').to(args.device)
    #forecast_loss = nn.L1Loss().to(args.device)
#    forecast_loss = nn.SmoothL1Loss().to(args.device)
    forecast_loss =  smooth_l1_loss
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
            # loss = forecast_loss(forecast, target) + forecast_loss(res, target)
            # loss1 = forecast_loss(forecast, target)
            # loss2 = forecast_loss(res, target)
            beta = 0.1 #for the threshold of the smooth L1 loss
            loss = forecast_loss(forecast, target, beta) + forecast_loss(res, target, beta)
            loss_F = forecast_loss(forecast, target, beta)
            loss_M = forecast_loss(res, target, beta)
            cnt += 1
            loss.backward()
            my_optim.step()
            loss_total += float(loss)
            loss_total_F  += float(loss_F)
            loss_total_M  += float(loss_M)

        print('| end of epoch {:3d} | time: {:5.2f}s | train_total_loss {:5.4f}, loss_F {:5.4f}, loss_M {:5.4f}  '.format(epoch, (
                time.time() - epoch_start_time), loss_total / cnt, loss_total_F / cnt, loss_total_M / cnt))
        # save_model(model, result_file, epoch)
        if (epoch+1) % args.exponential_decay_step == 0:
            my_lr_scheduler.step()
        if (epoch + 1) % args.validate_freq == 0:
            is_best_for_now = False
            print('------ validate on data: VALIDATE ------')
            performance_metrics = \
                validate(model, forecast_loss, valid_loader, args.device, args.norm_method, normalize_statistic,
                         node_cnt, args.window_size, args.horizon,
                         result_file=None, test=False)
            test_metrics=validate(model, forecast_loss, test_loader, args.device, args.norm_method, normalize_statistic,
                         node_cnt, args.window_size, args.horizon,
                         result_file=None, test=True)
            if best_validate_mae > performance_metrics['mae']:
                best_validate_mae = performance_metrics['mae']
                is_best_for_now = True
                validate_score_non_decrease_count = 0
                print('got best validation result:',performance_metrics, test_metrics)
            else:
                validate_score_non_decrease_count += 1
            if best_test_mae > test_metrics['mae']:
                best_test_mae = test_metrics['mae']
                print('got best test result:', test_metrics)
                
            # save model
            # if is_best_for_now:
            #     save_model(model, result_file)
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


    forecast_loss = nn.MSELoss(reduction='mean').to(args.device)
    model = load_model(result_train_file,epoch=epoch)
    node_cnt = test_data.shape[1]
    test_set = ForecastDataset(test_data, window_size=args.window_size, horizon=args.horizon,
                               normalize_method=args.norm_method, norm_statistic=normalize_statistic)
    test_loader = torch_data.DataLoader(test_set, batch_size=args.batch_size, drop_last=False,
                                        shuffle=False, num_workers=0)
    performance_metrics = validate(model = model, forecast_loss = forecast_loss, dataloader = test_loader, device =args.device, normalize_method = args.norm_method, statistic = normalize_statistic,
                      node_cnt = node_cnt, window_size = args.window_size, horizon =args.horizon,
                      result_file=None)
    mae, rmse = performance_metrics['mae'], performance_metrics['rmse']
    print('Performance on test set: | MAE: {:5.2f} | RMSE: {:5.4f}'.format(mae, rmse))

    # model, forecast_loss, dataloader, device, normalize_method


def retrain(train_data, valid_data, args, result_file, epoch):
    node_cnt = train_data.shape[1]

    model = load_model(result_file, epoch=epoch)


    model.to(args.device)
    train_mean = np.mean(train_data, axis=0)
    train_std = np.std(train_data, axis=0)
    normalize_statistic = {"mean": train_mean.tolist(), "std": train_std.tolist()}

    if args.optimizer == 'RMSProp':
        my_optim = torch.optim.RMSprop(params=model.parameters(), lr=args.lr, eps=1e-08)
    else:
        my_optim = torch.optim.Adam(params=model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=my_optim, gamma=args.decay_rate)

    train_set = ForecastDataset(train_data, window_size=args.window_size, horizon=args.horizon,
                                normalize_method=args.norm_method, norm_statistic=normalize_statistic)
    valid_set = ForecastDataset(valid_data, window_size=args.window_size, horizon=args.horizon,
                                normalize_method=args.norm_method, norm_statistic=normalize_statistic)
    train_loader = torch_data.DataLoader(train_set, batch_size=args.batch_size, drop_last=False, shuffle=True,
                                         num_workers=0)
    valid_loader = torch_data.DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    forecast_loss = nn.MSELoss(reduction='mean').to(args.device)

    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        total_params += param
    print(f"Total Trainable Params: {total_params}")

    best_validate_mae = np.inf
    validate_score_non_decrease_count = 0
    performance_metrics = {}
    for epoch in range(args.epoch):

        adjust_learning_rate(my_optim, epoch, args)
        epoch_start_time = time.time()
        model.train()
        loss_total = 0
        cnt = 0
        for i, (inputs, target) in enumerate(train_loader):
            inputs = inputs.to(args.device)  # torch.Size([32, 12, 228])
            target = target.to(args.device)  # torch.Size([32, 3, 228])
            model.zero_grad()
            forecast, _ = model(inputs)
            loss = forecast_loss(forecast, target)
            cnt += 1
            loss.backward()
            my_optim.step()
            loss_total += float(loss)
        print('| end of epoch {:3d} | time: {:5.2f}s | train_total_loss {:5.4f}'.format(epoch, (
                time.time() - epoch_start_time), loss_total / cnt))
        # save_model(model, result_file, epoch)
        if (epoch+1) % args.exponential_decay_step == 0:
            my_lr_scheduler.step()
        if (epoch + 1) % args.validate_freq == 0:
            is_best_for_now = False
            print('------ validate on data: VALIDATE ------')
            performance_metrics = \
                validate(model, forecast_loss, valid_loader, args.device, args.norm_method, normalize_statistic,
                         node_cnt, args.window_size, args.horizon,
                         result_file=result_file,test=False)
            if best_validate_mae > performance_metrics['mae']:
                best_validate_mae = performance_metrics['mae']
                is_best_for_now = True
                validate_score_non_decrease_count = 0
            else:
                validate_score_non_decrease_count += 1
            # # save model
            # # if is_best_for_now:
            # #     save_model(model, result_file)
            # if epoch%1==0:
            #     save_model(model, result_file,epoch=epoch)

        # early stop
        if args.early_stop and validate_score_non_decrease_count >= args.early_stop_step:
            break
    return performance_metrics, normalize_statistic
