import json
from datetime import datetime

from data_loader.forecast_dataloader import ForecastDataset,ForecastTestDataset, de_normalized
from models.base_model import Model
import torch
import torch.nn as nn
import torch.utils.data as torch_data
import numpy as np
import time
import os

# from models.StackTWaveNetTransformerEncoder import WASN

# from models.StackTWaveNetEnco2Deco import WASN
# from models.StackTWaveNetEcoDecoSemi import WASN
# from models.StackTWaveNetOverLap import WASN
# from models.IDCN import IDCNet
from models.LSTNet import Model
from models.TCN import TCN
from models.IDCN import IDCNet
from models.Transformer import Transformer

from utils.math_utils import evaluate, creatMask
from thop import profile, clever_format
from utils.flops import print_model_parm_flops

from utils.loss import smooth_l1_loss

def save_model(model, model_dir, epoch=None):
    if model_dir is None:
        return
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    epoch = str(epoch) if epoch else ''
    file_name = os.path.join(model_dir, epoch + 'Final_best08EcoDeco1563.pt')
    with open(file_name, 'wb') as f:
        torch.save(model, f)


def load_model(model_dir, epoch=None):
    if not model_dir:
        return
    epoch = str(epoch) if epoch else ''
    file_name = os.path.join(model_dir, epoch + 'Final_best08EcoDeco1563.pt')
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
            input_set.append(inputs.detach().cpu().numpy())
            step = 0
            forecast_steps = np.zeros([inputs.size()[0], horizon, node_cnt], dtype=np.float)
            Mid_steps = np.zeros([inputs.size()[0], horizon, node_cnt], dtype=np.float)
            while step < horizon:
                # print(i, inputs.shape[0])
                # input_save = inputs.detach().cpu().numpy()
                # np.save('F:\\school\\Papers\\timeseriesNew\\TS-Net\\output\\PEMS08\\' + 'inputNPEbt1.npy', input_save)
                # target_save = target.detach().cpu().numpy()
                # np.save('F:\\school\\Papers\\timeseriesNew\\TS-Net\\output\\PEMS08\\' + 'targetNPEbt1.npy', target_save)
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

            result_save = np.concatenate(forecast_set, axis=0)
            np.save('F:\\school\\Papers\\timeseriesNew\\TS-Net\\output\\PEMS08\\' + 'predNPEbt1.npy', result_save)
            target_save = np.concatenate(target_set, axis=0)
            np.save('F:\\school\\Papers\\timeseriesNew\\TS-Net\\output\\PEMS08\\' + 'targetNPEbt1.npy', target_save)

    return np.concatenate(forecast_set, axis=0), np.concatenate(target_set, axis=0),np.concatenate(Mid_set, axis=0), np.concatenate(input_set, axis=0)

def inferenceOverLap(model, dataloader, device, node_cnt, window_size, horizon):
    forecast_set = []
    Mid_set = []
    target_set = []
    input_set = []
    model.eval()
    with torch.no_grad():
        for i, (inputs, target) in enumerate(dataloader):
            inputs = inputs.to(device)
            target = target.to(device)
            input_set.append(inputs.detach().cpu().numpy())
            step = 0
            forecast_steps = np.zeros([inputs.size()[0], horizon, node_cnt], dtype=np.float)
            Mid_steps = np.zeros([inputs.size()[0], int(horizon/2), node_cnt], dtype=np.float)
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

                Mid_steps[:, step:min(int(horizon/2) - step, int(len_model_output/2)) + step, :] = \
                    Mid_result[:, :min(int(horizon/2) - step, int(len_model_output/2)), :].detach().cpu().numpy()

                step += min(horizon - step, len_model_output)
            forecast_set.append(forecast_steps)
            Mid_set.append(Mid_steps)
            target_set.append(target.detach().cpu().numpy())


    return np.concatenate(forecast_set, axis=0), np.concatenate(target_set, axis=0),np.concatenate(Mid_set, axis=0), np.concatenate(input_set, axis=0)


def validate(model, epoch, forecast_loss, dataloader, device, normalize_method, statistic,
             node_cnt, window_size, horizon, writer,
             result_file=None,test=False):
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
    score_final_detail = evaluate(target, forecast,by_step=True)
    print('by step:MAPE&MAE&RMSE',score_final_detail)
    end = datetime.now()

    if writer:
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
    if args.lradj== 1:
        lr_adjust = {epoch: args.lr * (0.95 ** ((epoch-1) // 1))}
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
    elif args.lradj==6:
        # lr_adjust = {
        #     2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
        #     10: 5e-7, 15: 1e-7, 20: 5e-8
        # }
        lr_adjust = {
            0: 0.0001, 5: 0.0005, 10:0.001, 20: 0.0001, 30: 0.00005, 40: 0.00001
            , 70: 0.000001
        }
    elif args.lradj==61:
        # lr_adjust = {
        #     2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
        #     10: 5e-7, 15: 1e-7, 20: 5e-8
        # }
        lr_adjust = {
            0: 0.0001, 5: 0.0005, 10:0.001, 25: 0.0005, 35: 0.0001, 50: 0.00001
            , 70: 0.000001
        }

    elif args.lradj==7:
        # lr_adjust = {
        #     2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
        #     10: 5e-7, 15: 1e-7, 20: 5e-8
        # }
        lr_adjust = {
            10: 0.0001, 30: 0.00005, 50: 0.00001
            , 70: 0.000001
        }

    elif args.lradj==8:
        # lr_adjust = {
        #     2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
        #     10: 5e-7, 15: 1e-7, 20: 5e-8
        # }
        lr_adjust = {
            0: 0.0005, 5: 0.0008, 10:0.001, 20: 0.0001, 30: 0.00005, 40: 0.00001
            , 70: 0.000001
        }
    elif args.lradj==9:
        # lr_adjust = {
        #     2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
        #     10: 5e-7, 15: 1e-7, 20: 5e-8
        # }
        lr_adjust = {
            0: 0.0001, 10: 0.0005, 20:0.001, 40: 0.0001, 45: 0.00005, 50: 0.00001
            , 70: 0.000001
        }

    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))

def train(data, train_data, valid_data, test_data, args, result_file, writer):
    node_cnt = train_data.shape[1]


    print("===================Train Normal=========================")
    part = [[1, 1], [0, 0], [0, 0]]

    print('level number {}, level details: {}'.format(len(part), part))
    # model = WASN(args, num_classes=args.horizon, num_stacks = args.num_stacks, first_conv = args.input_dim,
    #                   number_levels=len(part),
    #                   number_level_part=part)

    model = IDCNet(args, num_classes=args.horizon, input_len=args.window_size, input_dim=args.input_dim,
                   number_levels=len(part),
                   number_level_part=part, concat_len=None)

    print('Parameters of need to grad is:{} M'.format(count_params(model) / 1000000.0))
    in1 = torch.randn(args.batch_size, args.window_size, args.input_dim)
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
    # if args.norm_method == 'z_score':
    #     train_mean = np.mean(train_data, axis=0)
    #     train_std = np.std(train_data, axis=0)
    #     normalize_statistic = {"mean": train_mean.tolist(), "std": train_std.tolist()}
    # elif args.norm_method == 'min_max':
    #     train_min = np.min(train_data, axis=0)
    #     train_max = np.max(train_data, axis=0)
    #     normalize_statistic = {"min": train_min, "max": train_max}
    # else:
    #     normalize_statistic = None

    if args.normtype == 0:
        train_mean = np.mean(train_data, axis=0)
        train_std = np.std(train_data, axis=0)
        train_normalize_statistic = {"mean": train_mean.tolist(), "std": train_std.tolist()}
        val_mean = np.mean(valid_data, axis=0)
        val_std = np.std(valid_data, axis=0)
        val_normalize_statistic = {"mean": val_mean.tolist(), "std": val_std.tolist()}
        test_mean = np.mean(test_data, axis=0)
        test_std = np.std(test_data, axis=0)
        test_normalize_statistic = {"mean": test_mean.tolist(), "std": test_std.tolist()}
    elif args.normtype == 1:
        data_mean = np.mean(data, axis=0)
        data_std = np.std(data, axis=0)
        train_normalize_statistic = {"mean": data_mean.tolist(), "std": data_std.tolist()}
        val_normalize_statistic = {"mean": data_mean.tolist(), "std": data_std.tolist()}
        test_normalize_statistic = {"mean": data_mean.tolist(), "std": data_std.tolist()}
    else:
        train_mean = np.mean(train_data, axis=0)
        train_std = np.std(train_data, axis=0)
        train_normalize_statistic = {"mean": train_mean.tolist(), "std": train_std.tolist()}
        val_normalize_statistic = {"mean": train_mean.tolist(), "std": train_std.tolist()}
        test_normalize_statistic = {"mean": train_mean.tolist(), "std": train_std.tolist()}


    if args.optimizer == 'RMSProp':
        my_optim = torch.optim.RMSprop(params=model.parameters(), lr=args.lr, eps=1e-08)
    else:
        my_optim = torch.optim.Adam(params=model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
        # my_optim = torch.optim.AdamW(params=model.parameters(), lr=args.lr) 

    my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=my_optim, gamma=args.decay_rate)

    train_set = ForecastDataset(train_data, window_size=args.window_size, horizon=args.horizon,
                                normalize_method=args.norm_method, norm_statistic=train_normalize_statistic)
    valid_set = ForecastDataset(valid_data, window_size=args.window_size, horizon=args.horizon,
                                normalize_method=args.norm_method, norm_statistic=val_normalize_statistic)
    test_set = ForecastTestDataset(test_data, window_size=args.window_size, horizon=args.horizon,
                                normalize_method=args.norm_method, norm_statistic=test_normalize_statistic)
    train_loader = torch_data.DataLoader(train_set, batch_size=args.batch_size, drop_last=False, shuffle=True,
                                         num_workers=1)
    valid_loader = torch_data.DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=1)
    test_loader = torch_data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=1)

    #forecast_loss = nn.MSELoss(reduction='mean').to(args.device)
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
            #beta = 0.1 #for the threshold of the smooth L1 loss
            #loss = forecast_loss(forecast, target, beta) + forecast_loss(res, target, beta)
            #loss_F = forecast_loss(forecast, target, beta)
            #loss_M = forecast_loss(res, target, beta)
            cnt += 1
            loss.backward()
            my_optim.step()
            loss_total += float(loss)
            loss_total_F  += float(loss_F)
            loss_total_M  += float(loss_M)


        print('| end of epoch {:3d} | time: {:5.2f}s | train_total_loss {:5.4f}, loss_F {:5.4f}, loss_M {:5.4f}  '.format(epoch, (
                time.time() - epoch_start_time), loss_total / cnt, loss_total_F / cnt, loss_total_M / cnt))

        writer.add_scalar('Train_loss_tatal', loss_total / cnt, global_step=epoch)
        writer.add_scalar('Train_loss_Mid', loss_total_F / cnt, global_step=epoch)
        writer.add_scalar('Train_loss_Final', loss_total_M / cnt, global_step=epoch)

        # save_model(model, result_file, epoch)
        if (epoch+1) % args.exponential_decay_step == 0:
            my_lr_scheduler.step()
        if (epoch + 1) % args.validate_freq == 0:
            is_best_for_now = False
            print('------ validate on data: VALIDATE ------')
            performance_metrics = \
                validate(model, epoch, forecast_loss, valid_loader, args.device, args.norm_method, val_normalize_statistic,
                         node_cnt, args.window_size, args.horizon,
                         writer, result_file=None, test=False)
            test_metrics=validate(model, epoch,  forecast_loss, test_loader, args.device, args.norm_method, test_normalize_statistic,
                         node_cnt, args.window_size, args.horizon,
                         writer, result_file=None, test=True)
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
            if is_best_for_now:
                save_model(model, result_file)
            # if epoch%4==0:
            #     save_model(model, result_file,epoch=epoch)
        # early stop
        if args.early_stop and validate_score_non_decrease_count >= args.early_stop_step:
            break
    return performance_metrics, test_normalize_statistic


def test(test_data, train_data, args, result_train_file, result_test_file, epoch):

    test_mean = np.mean(test_data, axis=0)
    test_std = np.std(test_data, axis=0)
    normalize_statistic = {"mean": test_mean.tolist(), "std": test_std.tolist()}


    forecast_loss = nn.L1Loss().to(args.device) #smooth_l1_loss #nn.MSELoss(reduction='mean').to(args.device)
    model = load_model(result_train_file,epoch=epoch)
    node_cnt = test_data.shape[1]
    test_set = ForecastTestDataset(test_data, window_size=args.window_size, horizon=args.horizon,
                               normalize_method=args.norm_method, norm_statistic=normalize_statistic)
    test_loader = torch_data.DataLoader(test_set, batch_size=args.batch_size*10, drop_last=False,
                                        shuffle=False, num_workers=0)
    performance_metrics = validate(model = model, epoch = 100, forecast_loss = forecast_loss, dataloader = test_loader, device =args.device, normalize_method = args.norm_method, statistic = normalize_statistic,
                      node_cnt = node_cnt, window_size = args.window_size, horizon =args.horizon,
                      result_file=result_test_file, writer = None, test=True)
    mae, rmse, mape = performance_metrics['mae'], performance_metrics['rmse'], performance_metrics['mape']
    print('Performance on test set: | MAE: {:5.2f} | MAPE: {:5.2f} | RMSE: {:5.4f}'.format(mae, mape, rmse))

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
                # save model
                # if is_best_for_now:
                #     save_model(model, result_file)
            # if epoch%1==0:
            #     save_model(model, result_file,epoch=epoch)

        # early stop
        if args.early_stop and validate_score_non_decrease_count >= args.early_stop_step:
            break
    return performance_metrics, normalize_statistic


def trainSemi(train_data, valid_data, test_data, args, result_file, writer):
    print("===================Train-Semi=========================")

    node_cnt = train_data.shape[1]

    # part = [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]  # Best model
    # part = [[1, 1], [1,1], [1,1], [0, 0], [0, 0], [0, 0], [0, 0]]  # Best model

    part = [[1, 1], [0, 0], [0, 0]]
    # # part = [[0, 1], [0, 0]]
    # part = [[0, 0]]
    print('level number {}, level details: {}'.format(len(part), part))
    model = WASN(args, num_classes=args.horizon, num_stacks=args.num_stacks, first_conv=args.input_dim,
                 number_levels=len(part),
                 number_level_part=part,
                 haar_wavelet=False)

    print('Parameters of need to grad is:{} M'.format(count_params(model) / 1000000.0))
    in1 = torch.randn(8, 12, 170)
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

    #    forecast_loss = nn.MSELoss(reduction='mean').to(args.device)
    # forecast_loss = nn.L1Loss().to(args.device)
    #    forecast_loss = nn.SmoothL1Loss().to(args.device)
    forecast_loss = smooth_l1_loss
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
            beta = 0.1  # for the threshold of the smooth L1 loss
            loss = forecast_loss(forecast, inputs, beta) + forecast_loss(res, target, beta)
            loss_F = forecast_loss(forecast, inputs, beta)
            loss_M = forecast_loss(res, target, beta)
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
        writer.add_scalar('Train_loss_Mid', loss_total_M / cnt, global_step=epoch)
        writer.add_scalar('Train_loss_Final', loss_total_F / cnt, global_step=epoch)

        # save_model(model, result_file, epoch)
        if (epoch + 1) % args.exponential_decay_step == 0:
            my_lr_scheduler.step()
        if (epoch + 1) % args.validate_freq == 0:
            is_best_for_now = False
            print('------ validate on data: VALIDATE ------')
            performance_metrics = \
                validateSemi(model, epoch, forecast_loss, valid_loader, args.device, args.norm_method, normalize_statistic,
                         node_cnt, args.window_size, args.horizon,
                         writer, result_file=None, test=False)
            test_metrics = validateSemi(model, epoch, forecast_loss, test_loader, args.device, args.norm_method,
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
                print('Best validation model Saved')
            # if epoch%4==0:
            #     save_model(model, result_file,epoch=epoch)
        # early stop
        if args.early_stop and validate_score_non_decrease_count >= args.early_stop_step:
            break
    return performance_metrics, normalize_statistic



def validateSemi(model, epoch, forecast_loss, dataloader, device, normalize_method, statistic,
             node_cnt, window_size, horizon, writer,
             result_file=None,test=False):
    start = datetime.now()
    print("===================Validate-Semi=========================")
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
    input_norm = torch.from_numpy(input_norm).float()
    loss = forecast_loss(forecast_norm, input_norm, beta) + forecast_loss(mid_norm, target_norm, beta)
    loss_F = forecast_loss(forecast_norm, input_norm, beta)
    loss_M = forecast_loss(mid_norm, target_norm, beta)

    # score = evaluate(target, forecast)
    score = evaluate(input, forecast)
    score1 = evaluate(target, mid)

    score_final_detail = evaluate(input, forecast,by_step=True)
    print('by step:MAPE&MAE&RMSE',score_final_detail)


    end = datetime.now()


    if test:
        print(f'TEST: RAW : MAE {score[1]:7.2f}; RMSE {score[2]:7.2f}.')
        print(f'TEST: RAW-Mid : MAE {score1[1]:7.2f}; RMSE {score1[2]:7.2f}.')
        if writer:
            writer.add_scalar('Test MAE_final', score[1], global_step=epoch)
            writer.add_scalar('Test MAE_Mid', score1[1], global_step=epoch)
            writer.add_scalar('Test RMSE_final', score[2], global_step=epoch)
            writer.add_scalar('Test RMSE_Mid', score1[2], global_step=epoch)

        print(f'TEST: Loss final: {loss_F:5.5f}.')
        print(f'TEST: Loss Mid :  {loss_M:5.5f}.')
        if writer:
            writer.add_scalar('Test Loss_final', loss_F, global_step=epoch)
            writer.add_scalar('Test Loss_Mid', loss_M, global_step=epoch)

    else:
        print(f'VAL: RAW : MAE {score[1]:7.2f}; RMSE {score[2]:7.2f}.')
        print(f'VAL: RAW-Mid : MAE {score1[1]:7.2f}; RMSE {score1[2]:7.2f}.')
        if writer:
            writer.add_scalar('VAL MAE_final', score[1], global_step=epoch)
            writer.add_scalar('VAL MAE_Mid', score1[1], global_step=epoch)
            writer.add_scalar('VAL RMSE_final', score[2], global_step=epoch)
            writer.add_scalar('VAL RMSE_Mid', score1[2], global_step=epoch)

        print(f'VAL: Loss final: {loss_F:5.5f}.')
        print(f'VAL: Loss Mid :  {loss_M:5.5f}.')
        if writer:
            writer.add_scalar('VAL Loss_final', loss_F, global_step=epoch)
            writer.add_scalar('VAL Loss_Mid', loss_M, global_step=epoch)

    if result_file:
        if not os.path.exists(result_file):
            os.makedirs(result_file)
        step_to_print = 0
        forcasting_2d = forecast[:, step_to_print, :]
        forcasting_2d_target = target[:, step_to_print, :]
        forcasting_2d_input = input[:, step_to_print, :]
        forcasting_2d_mid = mid[:, step_to_print, :]

        np.savetxt(f'{result_file}/target.csv', forcasting_2d_target, delimiter=",")
        np.savetxt(f'{result_file}/predict.csv', forcasting_2d, delimiter=",")
        np.savetxt(f'{result_file}/predict_abs_error.csv',
                   np.abs(forcasting_2d - forcasting_2d_target), delimiter=",")
        np.savetxt(f'{result_file}/predict_ape.csv',
                   np.abs((forcasting_2d - forcasting_2d_target) / forcasting_2d_target), delimiter=",")

        np.savetxt(f'{result_file}/input.csv', forcasting_2d_input, delimiter=",")
        np.savetxt(f'{result_file}/mid.csv', forcasting_2d_mid, delimiter=",")
        np.savetxt(f'{result_file}/input_forcast_abs_error.csv',
                   np.abs(forcasting_2d_input - forcasting_2d), delimiter=",")
        np.savetxt(f'{result_file}/target_mid_abs_error.csv',
                   np.abs(forcasting_2d_target - forcasting_2d_mid), delimiter=",")

    return dict(mape=score1[0], mae=score1[1], rmse=score1[2])



def trainEco2Deco(train_data, valid_data, test_data, args, result_file, writer):


    node_cnt = train_data.shape[1]

    # part = [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]  # Best model
    # part = [[1, 1], [1,1], [1,1], [0, 0], [0, 0], [0, 0], [0, 0]]  # Best model

    part = [[1, 1], [0, 0], [0, 0]]
    # # part = [[0, 1], [0, 0]]
    # part = [[0, 0]]
    print('level number {}, level details: {}'.format(len(part), part))
    model = WASN(args, num_classes=args.horizon, num_stacks=args.num_stacks, first_conv=args.input_dim,
                 number_levels=len(part),
                 number_level_part=part)

    print('Parameters of need to grad is:{} M'.format(count_params(model) / 1000000.0))
    in1 = torch.randn(8, 12, 170)
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

    #    forecast_loss = nn.MSELoss(reduction='mean').to(args.device)
    # forecast_loss = nn.L1Loss().to(args.device)
    #    forecast_loss = nn.SmoothL1Loss().to(args.device)
    forecast_loss = smooth_l1_loss
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
        loss_total_Final = 0
        loss_total_First = 0
        loss_total_Second = 0
        cnt = 0
        for i, (inputs, target) in enumerate(train_loader):
            inputs = inputs.to(args.device)  # torch.Size([32, 12, 228])
            target = target.to(args.device)  # torch.Size([32, 3, 228])
            model.zero_grad()
            forecast, first, second = model(inputs)
            # loss = forecast_loss(forecast, target) + forecast_loss(res, target)
            # loss1 = forecast_loss(forecast, target)
            # loss2 = forecast_loss(res, target)
            beta = 0.1  # for the threshold of the smooth L1 loss
            loss = forecast_loss(forecast, target, beta) + forecast_loss(first, target, beta) + forecast_loss(second, target, beta)
            loss_Final = forecast_loss(forecast, target, beta)
            loss_First = forecast_loss(first, target, beta)
            loss_Second = forecast_loss(second, target, beta)
            cnt += 1
            loss.backward()
            my_optim.step()
            loss_total += float(loss)
            loss_total_Final += float(loss_Final)
            loss_total_First += float(loss_First)
            loss_total_Second += float(loss_Second)

        print(
            '| end of epoch {:3d} | time: {:5.2f}s | train_total_loss {:5.4f}, loss_Final {:5.4f}, loss_First {:5.4f} , loss_Second {:5.4f} '.format(
                epoch, (
                        time.time() - epoch_start_time), loss_total / cnt, loss_total_Final / cnt, loss_total_First / cnt, loss_total_Second / cnt))

        writer.add_scalar('Train_loss_tatal', loss_total / cnt, global_step=epoch)
        writer.add_scalar('Train_loss_First', loss_total_First / cnt, global_step=epoch)
        writer.add_scalar('Train_loss_Second', loss_total_Second / cnt, global_step=epoch)
        writer.add_scalar('Train_loss_Final', loss_total_Final / cnt, global_step=epoch)

        # save_model(model, result_file, epoch)
        if (epoch + 1) % args.exponential_decay_step == 0:
            my_lr_scheduler.step()
        if (epoch + 1) % args.validate_freq == 0:
            is_best_for_now = False
            print('------ validate on data: VALIDATE ------')
            performance_metrics = \
                validateEco2Deco(model, epoch, forecast_loss, valid_loader, args.device, args.norm_method, normalize_statistic,
                         node_cnt, args.window_size, args.horizon,
                         writer, result_file=None, test=False)
            test_metrics = validateEco2Deco(model, epoch, forecast_loss, test_loader, args.device, args.norm_method,
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
            # if is_best_for_now:
            #     save_model(model, result_file)
            # if epoch%4==0:
            #     save_model(model, result_file,epoch=epoch)
        # early stop
        if args.early_stop and validate_score_non_decrease_count >= args.early_stop_step:
            break
    return performance_metrics, normalize_statistic



def validateEco2Deco(model, epoch, forecast_loss, dataloader, device, normalize_method, statistic,
             node_cnt, window_size, horizon, writer,
             result_file=None,test=False):
    start = datetime.now()
    print("===================ValidateEco2Deco=========================")
    forecast_norm, target_norm, first_norm, second_norm = inferenceEcoDeco(model, dataloader, device,
                                           node_cnt, window_size, horizon)
    if normalize_method and statistic:
        forecast = de_normalized(forecast_norm, normalize_method, statistic)
        target = de_normalized(target_norm, normalize_method, statistic)
        first = de_normalized(first_norm, normalize_method, statistic)
        second = de_normalized(second_norm, normalize_method, statistic)
    else:
        forecast, target = forecast_norm, target_norm
        forecast, target, first, second = forecast_norm, target_norm, first_norm, second_norm



    # score = evaluate(target, forecast)
    score = evaluate(target, forecast)
    score_first = evaluate(target, first)
    score_second = evaluate(target, second)


    end = datetime.now()


    if test:
        print(f'TEST: RAW : MAE {score[1]:7.2f}; RMSE {score[2]:7.2f}.')
        print(f'TEST: RAW-First : MAE {score_first[1]:7.2f}; RMSE {score_first[2]:7.2f}.')
        print(f'TEST: RAW-Second : MAE {score_second[1]:7.2f}; RMSE {score_second[2]:7.2f}.')
        writer.add_scalar('Test MAE_Final', score[1], global_step=epoch)
        writer.add_scalar('Test MAE_First', score_first[1], global_step=epoch)
        writer.add_scalar('Test MAE_Second', score_second[1], global_step=epoch)
        writer.add_scalar('Test RMSE_Final', score[2], global_step=epoch)
        writer.add_scalar('Test RMSE_First', score_first[2], global_step=epoch)
        writer.add_scalar('Test RMSE_Second', score_second[2], global_step=epoch)

    else:
        print(f'Validate: RAW : MAE {score[1]:7.2f}; RMSE {score[2]:7.2f}.')
        print(f'Validate: RAW-First : MAE {score_first[1]:7.2f}; RMSE {score_first[2]:7.2f}.')
        print(f'Validate: RAW-Second : MAE {score_second[1]:7.2f}; RMSE {score_second[2]:7.2f}.')
        writer.add_scalar('Validate MAE_Final', score[1], global_step=epoch)
        writer.add_scalar('Validate MAE_First', score_first[1], global_step=epoch)
        writer.add_scalar('Validate MAE_Second', score_second[1], global_step=epoch)
        writer.add_scalar('Validate RMSE_Final', score[2], global_step=epoch)
        writer.add_scalar('Validate RMSE_First', score_first[2], global_step=epoch)
        writer.add_scalar('Validate RMSE_Second', score_second[2], global_step=epoch)
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


def inferenceEcoDeco(model, dataloader, device, node_cnt, window_size, horizon):
    forecast_set = []
    Second_set = []
    First_set = []
    target_set = []
    input_set = []
    model.eval()
    with torch.no_grad():
        for i, (inputs, target) in enumerate(dataloader):
            inputs = inputs.to(device)
            target = target.to(device)
            input_set.append(inputs.detach().cpu().numpy())
            step = 0
            forecast_steps = np.zeros([inputs.size()[0], horizon, node_cnt], dtype=np.float)
            Second_steps = np.zeros([inputs.size()[0], horizon, node_cnt], dtype=np.float)
            First_steps = np.zeros([inputs.size()[0], horizon, node_cnt], dtype=np.float)
            while step < horizon:
                forecast_result, First_result, Second_result = model(inputs)
                len_model_output = forecast_result.size()[1]
                if len_model_output == 0:
                    raise Exception('Get blank inference result')
                inputs[:, :window_size - len_model_output, :] = inputs[:, len_model_output:window_size,
                                                                   :].clone()
                inputs[:, window_size - len_model_output:, :] = forecast_result.clone()
                forecast_steps[:, step:min(horizon - step, len_model_output) + step, :] = \
                    forecast_result[:, :min(horizon - step, len_model_output), :].detach().cpu().numpy()

                Second_steps[:, step:min(horizon - step, len_model_output) + step, :] = \
                    Second_result[:, :min(horizon - step, len_model_output), :].detach().cpu().numpy()

                First_steps[:, step:min(horizon - step, len_model_output) + step, :] = \
                    First_result[:, :min(horizon - step, len_model_output), :].detach().cpu().numpy()

                step += min(horizon - step, len_model_output)
            forecast_set.append(forecast_steps)
            First_set.append(First_steps)
            Second_set.append(Second_steps)
            target_set.append(target.detach().cpu().numpy())


    return np.concatenate(forecast_set, axis=0), np.concatenate(target_set, axis=0),np.concatenate(First_set, axis=0), np.concatenate(Second_set, axis=0)

def trainOverLap(train_data, valid_data, test_data, args, result_file, writer):
    print("===================Train-Semi=========================")

    node_cnt = train_data.shape[1]


    part = [[1, 1], [0, 0], [0, 0]]
    # # part = [[0, 1], [0, 0]]
    # part = [[0, 0]]
    print('level number {}, level details: {}'.format(len(part), part))
    model = WASN(args, num_classes=args.horizon, num_stacks=args.num_stacks, first_conv=args.input_dim,
                 number_levels=len(part),
                 number_level_part=part)

    print('Parameters of need to grad is:{} M'.format(count_params(model) / 1000000.0))
    in1 = torch.randn(8, 12, 170)
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
    # forecast_loss = nn.L1Loss().to(args.device)
    #    forecast_loss = nn.SmoothL1Loss().to(args.device)
    forecast_loss = smooth_l1_loss
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

        forecast_set = []
        Mid_set = []
        target_set = []
        input_set = []

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
            target_mid = target[:,0:6,:] #torch.cat((inputs[:,6:12,:],target[:,0:6,:]),dim = 1)
            a = inputs[0,:,0]
            b = target[0,:,0]
            c = target_mid[0,:,0]
            model.zero_grad()
            forecast, res = model(inputs)

            beta = 0.1  # for the threshold of the smooth L1 loss
            loss = forecast_loss(forecast, target, beta) + forecast_loss(res, target_mid, beta)
            loss_F = forecast_loss(forecast, target, beta)
            loss_M = forecast_loss(res, target_mid, beta)


            forecast_set.append(forecast.detach().cpu().numpy())
            Mid_set.append(res.detach().cpu().numpy())
            target_set.append(target.detach().cpu().numpy())
            input_set.append(inputs.detach().cpu().numpy())



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

        forecast_total = np.concatenate(forecast_set, axis=0)
        target_total = np.concatenate(target_set, axis=0)
        Mid_total = np.concatenate(Mid_set,axis=0)
        input_total = np.concatenate(input_set, axis=0)
        target_mid_total = target_total[:,0:6,:] #np.concatenate((input_total[:,6:12,:],target_total[:,0:6,:]),axis = 1)

        score_final_detail = evaluate(target_total, forecast_total, by_step=True)
        score_mid_detail = evaluate(target_mid_total, Mid_total, by_step=True)
        print('by Train Mid_step:MAPE&MAE&RMSE', score_mid_detail)
        print('by Train Final_step:MAPE&MAE&RMSE', score_final_detail)

        writer.add_scalar('Train_loss_tatal', loss_total / cnt, global_step=epoch)
        writer.add_scalar('Train_loss_Mid', loss_total_M / cnt, global_step=epoch)
        writer.add_scalar('Train_loss_Final', loss_total_F / cnt, global_step=epoch)

        # save_model(model, result_file, epoch)
        if (epoch + 1) % args.exponential_decay_step == 0:
            my_lr_scheduler.step()
        if (epoch + 1) % args.validate_freq == 0:
            is_best_for_now = False
            print('------ validate on data: VALIDATE ------')
            performance_metrics = \
                validateOverLap(model, epoch, forecast_loss, valid_loader, args.device, args.norm_method, normalize_statistic,
                         node_cnt, args.window_size, args.horizon,
                         writer, result_file=None, test=False)
            test_metrics = validateOverLap(model, epoch, forecast_loss, test_loader, args.device, args.norm_method,
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
                print('Best validation model Saved')
            # if epoch%4==0:
            #     save_model(model, result_file,epoch=epoch)
        # early stop
        if args.early_stop and validate_score_non_decrease_count >= args.early_stop_step:
            break
    return performance_metrics, normalize_statistic



def validateOverLap(model, epoch, forecast_loss, dataloader, device, normalize_method, statistic,
             node_cnt, window_size, horizon, writer,
             result_file=None,test=False):
    start = datetime.now()
    print("===================Validate-Semi=========================")
    forecast_norm, target_norm, mid_norm, input_norm = inferenceOverLap(model, dataloader, device,
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
    input_norm = torch.from_numpy(input_norm).float()
    target_mid_norm = target_norm[:,0:6,:] #torch.cat((input_norm[:,6:12,:],target_norm[:,0:6,:]),dim = 1)

    loss = forecast_loss(forecast_norm, target_norm, beta) + forecast_loss(mid_norm, target_mid_norm, beta)
    loss_F = forecast_loss(forecast_norm, target_norm, beta)
    loss_M = forecast_loss(mid_norm, target_mid_norm, beta)

    # score = evaluate(target, forecast)
    target_mid = target[:,0:6,:] #np.concatenate((input[:,6:12,:],target[:,0:6,:]),axis = 1)
    score = evaluate(target, forecast)
    score1 = evaluate(target_mid, mid)

    score_final_detail = evaluate(target, forecast,by_step=True)
    score_mid_detail = evaluate(target_mid, mid, by_step=True)
    print('by Val/Test Mid_step:MAPE&MAE&RMSE', score_mid_detail)
    print('by Val/Test Final_step:MAPE&MAE&RMSE',score_final_detail)


    end = datetime.now()


    if test:
        print(f'TEST: RAW : MAE {score[1]:7.2f}; RMSE {score[2]:7.2f}.')
        print(f'TEST: RAW-Mid : MAE {score1[1]:7.2f}; RMSE {score1[2]:7.2f}.')
        if writer:
            writer.add_scalar('Test MAE_final', score[1], global_step=epoch)
            writer.add_scalar('Test MAE_Mid', score1[1], global_step=epoch)
            writer.add_scalar('Test RMSE_final', score[2], global_step=epoch)
            writer.add_scalar('Test RMSE_Mid', score1[2], global_step=epoch)

        print(f'TEST: Loss final: {loss_F:5.5f}.')
        print(f'TEST: Loss Mid :  {loss_M:5.5f}.')
        if writer:
            writer.add_scalar('Test Loss_final', loss_F, global_step=epoch)
            writer.add_scalar('Test Loss_Mid', loss_M, global_step=epoch)

    else:
        print(f'VAL: RAW : MAE {score[1]:7.2f}; RMSE {score[2]:7.2f}.')
        print(f'VAL: RAW-Mid : MAE {score1[1]:7.2f}; RMSE {score1[2]:7.2f}.')
        if writer:
            writer.add_scalar('VAL MAE_final', score[1], global_step=epoch)
            writer.add_scalar('VAL MAE_Mid', score1[1], global_step=epoch)
            writer.add_scalar('VAL RMSE_final', score[2], global_step=epoch)
            writer.add_scalar('VAL RMSE_Mid', score1[2], global_step=epoch)

        print(f'VAL: Loss final: {loss_F:5.5f}.')
        print(f'VAL: Loss Mid :  {loss_M:5.5f}.')
        if writer:
            writer.add_scalar('VAL Loss_final', loss_F, global_step=epoch)
            writer.add_scalar('VAL Loss_Mid', loss_M, global_step=epoch)

    if result_file:
        if not os.path.exists(result_file):
            os.makedirs(result_file)
        step_to_print = 0
        forcasting_2d = forecast[:, step_to_print, :]
        forcasting_2d_target = target[:, step_to_print, :]
        forcasting_2d_input = input[:, step_to_print, :]
        forcasting_2d_mid = mid[:, step_to_print, :]

        np.savetxt(f'{result_file}/target.csv', forcasting_2d_target, delimiter=",")
        np.savetxt(f'{result_file}/predict.csv', forcasting_2d, delimiter=",")
        np.savetxt(f'{result_file}/predict_abs_error.csv',
                   np.abs(forcasting_2d - forcasting_2d_target), delimiter=",")
        np.savetxt(f'{result_file}/predict_ape.csv',
                   np.abs((forcasting_2d - forcasting_2d_target) / forcasting_2d_target), delimiter=",")

        np.savetxt(f'{result_file}/input.csv', forcasting_2d_input, delimiter=",")
        np.savetxt(f'{result_file}/mid.csv', forcasting_2d_mid, delimiter=",")
        np.savetxt(f'{result_file}/input_forcast_abs_error.csv',
                   np.abs(forcasting_2d_input - forcasting_2d), delimiter=",")
        np.savetxt(f'{result_file}/target_mid_abs_error.csv',
                   np.abs(forcasting_2d_target - forcasting_2d_mid), delimiter=",")

    return dict(mape=score[0], mae=score[1], rmse=score[2])


def trainBaseline(total, train_data, valid_data, test_data, args, result_file, writer):
    print("===================Train-Semi=========================")

    node_cnt = train_data.shape[1]

    channel_sizes = [args.nhid] * args.levels
    # model = TCN(args.input_dim, args.horizon, channel_sizes, kernel_size=args.kernel, dropout=args.dropout)
    # model = Model(args) #LSTNet
    part = [[1, 1], [0, 0], [0, 0]]
    #
    print('level number {}, level details: {}'.format(len(part), part))
    #
    model = IDCNet(args, num_classes=args.horizon, input_len=args.window_size, input_dim=args.input_dim,
                   number_levels=len(part),
                   number_level_part=part, concat_len=None)
    # model = Transformer(
    #                      n_layers=args.n_layers,
    #                      hidden_size = args.input_dim,
    #                      filter_size=args.filter_size,
    #                      dropout_rate=args.dropout,
    #                      head_size = args.head_size,
    #                      has_inputs=True,
    #                      src_pad_idx=None,
    #                      trg_pad_idx=None)

    # print('Parameters of need to grad is:{} M'.format(count_params(model) / 1000000.0))
    # in1 = torch.randn(32, 12, 170)
    # flops, params = profile(model, inputs=(in1,))
    # macs, params = clever_format([flops, params], "%.3f")
    # print('MACs: {}, Parameters: {}'.format(macs, params))
    #    print_model_parm_flops(model)
    model.to(args.device)
    if len(train_data) == 0:
        raise Exception('Cannot organize enough training data')
    if len(valid_data) == 0:
        raise Exception('Cannot organize enough validation data')
    if len(test_data) == 0:
        raise Exception('Cannot organize enough test data')
    if args.norm_method == 'z_score':
        train_mean = np.mean(total, axis=0)
        train_std = np.std(total, axis=0)
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
    # forecast_loss = nn.L1Loss().to(args.device)
    #    forecast_loss = nn.SmoothL1Loss().to(args.device)
    forecast_loss = smooth_l1_loss
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

        forecast_set = []

        target_set = []


        adjust_learning_rate(my_optim, epoch, args)
        epoch_start_time = time.time()
        model.train()
        loss_total = 0
        loss_total_F = 0
        loss_total_M = 0
        cnt = 0
        for i, (inputs, target) in enumerate(train_loader):
            # print("iter",i)
            inputs = inputs.to(args.device)  # torch.Size([32, 12, 228])
            target = target.to(args.device)  # torch.Size([32, 3, 228])
            # target = inputs
            mask = creatMask(inputs)
            inputs = inputs.masked_fill(mask, 0)
            # inputs = inputs.permute(0, 2, 1)
            model.zero_grad()

            forecast = model(inputs)
            forecast = forecast.squeeze()
            beta = 0.1  # for the threshold of the smooth L1 loss
            loss = forecast_loss(forecast, target, beta)
            loss_F = forecast_loss(forecast, target, beta)



            forecast_set.append(forecast.detach().cpu().numpy())

            target_set.append(target.detach().cpu().numpy())




            cnt += 1
            loss.backward()
            my_optim.step()
            loss_total += float(loss)
            loss_total_F += float(loss_F)

        print(
            '| end of epoch {:3d} | time: {:5.2f}s | train_total_loss {:5.4f}, loss_F {:5.4f}  '.format(
                epoch, (
                        time.time() - epoch_start_time), loss_total / cnt, loss_total_F / cnt))

        forecast_total = np.concatenate(forecast_set, axis=0)
        target_total = np.concatenate(target_set, axis=0)



        score_final_detail = evaluate(target_total, forecast_total, by_step=True)


        print('by Train Final_step:MAPE&MAE&RMSE', score_final_detail)

        writer.add_scalar('Train_loss_tatal', loss_total / cnt, global_step=epoch)

        writer.add_scalar('Train_loss_Final', loss_total_F / cnt, global_step=epoch)

        # save_model(model, result_file, epoch)
        if (epoch + 1) % args.exponential_decay_step == 0:
            my_lr_scheduler.step()
        if (epoch + 1) % args.validate_freq == 0:
            is_best_for_now = False
            print('------ validate on data: VALIDATE ------')
            performance_metrics = \
                validateBaseline(model, epoch, forecast_loss, valid_loader, args.device, args.norm_method, normalize_statistic,
                         node_cnt, args.window_size, args.horizon,
                         writer, result_file=None, test=False)
            test_metrics = validateBaseline(model, epoch, forecast_loss, test_loader, args.device, args.norm_method,
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
                print('Best validation model Saved')
            # if epoch%4==0:
            #     save_model(model, result_file,epoch=epoch)
        # early stop
        if args.early_stop and validate_score_non_decrease_count >= args.early_stop_step:
            break
    return performance_metrics, normalize_statistic



def validateBaseline(model, epoch, forecast_loss, dataloader, device, normalize_method, statistic,
             node_cnt, window_size, horizon, writer,
             result_file=None,test=False):
    start = datetime.now()
    print("===================Validate-Semi=========================")
    forecast_norm, target_norm , input_norm = inferenceBaseline(model, dataloader, device,
                                           node_cnt, window_size, horizon)
    forecast_norm = forecast_norm.squeeze()
    target_norm = target_norm.squeeze()
    if normalize_method and statistic:
        forecast = de_normalized(forecast_norm, normalize_method, statistic)
        target = de_normalized(target_norm, normalize_method, statistic)

    else:
        forecast, target = forecast_norm, target_norm
    forecast_norm = forecast_norm.squeeze()
    target_norm = target_norm.squeeze()
    beta = 0.1
    forecast_norm = torch.from_numpy(forecast_norm).float()

    target_norm = torch.from_numpy(target_norm).float()



    loss = forecast_loss(forecast_norm, target_norm, beta)
    loss_F = forecast_loss(forecast_norm, target_norm, beta)


    # score = evaluate(target, forecast)

    score = evaluate(target, forecast)

    score_final_detail = evaluate(target, forecast,by_step=True)

    print('by Val/Test Final_step:MAPE&MAE&RMSE',score_final_detail)


    end = datetime.now()


    if test:
        print(f'TEST: RAW : MAE {score[1]:7.2f}; RMSE {score[2]:7.2f}.')

        if writer:
            writer.add_scalar('Test MAE_final', score[1], global_step=epoch)

            writer.add_scalar('Test RMSE_final', score[2], global_step=epoch)


        print(f'TEST: Loss final: {loss_F:5.5f}.')

        if writer:
            writer.add_scalar('Test Loss_final', loss_F, global_step=epoch)


    else:
        print(f'VAL: RAW : MAE {score[1]:7.2f}; RMSE {score[2]:7.2f}.')

        if writer:
            writer.add_scalar('VAL MAE_final', score[1], global_step=epoch)

            writer.add_scalar('VAL RMSE_final', score[2], global_step=epoch)


        print(f'VAL: Loss final: {loss_F:5.5f}.')

        if writer:
            writer.add_scalar('VAL Loss_final', loss_F, global_step=epoch)


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


    return dict(mape=score[0], mae=score[1], rmse=score[2])

def inferenceBaseline(model, dataloader, device, node_cnt, window_size, horizon):
    forecast_set = []
    Mid_set = []
    target_set = []
    input_set = []
    model.eval()
    with torch.no_grad():
        for i, (inputs, target) in enumerate(dataloader):
            inputs = inputs.to(device)
            target = target.to(device)

            mask = creatMask(inputs)
            inputs = inputs.masked_fill(mask, 0)

            # target = inputs
            input_set.append(inputs.detach().cpu().numpy())
            target_set.append(target.detach().cpu().numpy())
            step = 0
            forecast_steps = np.zeros([inputs.size()[0], horizon, node_cnt], dtype=np.float)

            while step < horizon:
                # input_tcn = inputs.permute(0, 2, 1)
                forecast_result = model(inputs)
                len_model_output = forecast_result.size()[1]
                if len_model_output == 0:
                    raise Exception('Get blank inference result')
                inputs[:, :window_size - len_model_output, :] = inputs[:, len_model_output:window_size,
                                                                   :].clone()
                inputs[:, window_size - len_model_output:, :] = forecast_result.clone()
                forecast_steps[:, step:min(horizon - step, len_model_output) + step, :] = \
                    forecast_result[:, :min(horizon - step, len_model_output), :].detach().cpu().numpy()



                step += min(horizon - step, len_model_output)
            forecast_set.append(forecast_steps)




    return np.concatenate(forecast_set, axis=0), np.concatenate(target_set, axis=0), np.concatenate(input_set, axis=0)