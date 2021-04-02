import argparse
import math
import time

import torch
import torch.nn as nn

import numpy as np
import importlib

from util_financial import *

# from StackTWNet import WASN
from models.StackTWaveNetTransformerEncoder import WASN

from tensorboardX import SummaryWriter
import torch.optim as optim
import math

import util_financial


parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
parser.add_argument('--data', type=str, default='./dataset/electricity.txt',
                    help='location of the data file')
parser.add_argument('--log_interval', type=int, default=2000, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model/model.pt',
                    help='path to save the final model')
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--L1Loss', type=bool, default=True)
parser.add_argument('--normalize', type=int, default=2)
parser.add_argument('--device',type=str,default='cuda:0',help='')

parser.add_argument('--num_nodes',type=int,default=321,help='number of nodes/variables')


parser.add_argument('--seq_in_len',type=int,default= 48,help='input sequence length') #24*7

parser.add_argument('--horizon', type=int, default=24)


parser.add_argument('--batch_size',type=int,default=32,help='batch size')
parser.add_argument('--lr',type=float,default=0.0003,help='learning rate')
parser.add_argument('--weight_decay',type=float,default=0.00001,help='weight decay rate')


parser.add_argument('--epochs',type=int,default=100,help='')


parser.add_argument('--hidden-size', default=1, type=float, help='hidden channel of module')
parser.add_argument('--INN', default=1, type=int, help='use INN or basic strategy')
parser.add_argument('--kernel', default=3, type=int, help='kernel size')
parser.add_argument('--dilation', default=1, type=int, help='dilation')
parser.add_argument('--window_size', type=int, default=1)
parser.add_argument('--lradj', type=int, default=9,help='adjust learning rate')

parser.add_argument('--model_name', type=str, default='base')

args = parser.parse_args()
args.window_size = args.seq_in_len

device = torch.device(args.device)
torch.set_num_threads(3)







def trainEecoDeco(epoch, data, X, Y, model, criterion, optim, batch_size):
    model.train()
    total_loss = 0
    n_samples = 0
    iter = 0
    final_loss = 0
    min_loss = 0


    for X, Y in data.get_batches(X, Y, batch_size, True):
        model.zero_grad()             #torch.Size([32, 168, 137])


        tx = X#[:, id, :] #torch.Size([32, 1, 137, 168])
        ty = Y#[:, id]       #torch.Size([32, 137])
        # output = model(tx,id) #torch.Size([32, 1, 137, 1])
        # output = model(tx)  # torch.Size([32, 1, 137, 1])
        forecast, res = model(tx)
        forecast = torch.squeeze(forecast)
        scale = data.scale.expand(forecast.size(0), args.horizon, data.m)

        loss = criterion(forecast * scale, ty * scale) + criterion(res * scale, ty * scale)
        loss.backward()
        total_loss += loss.item()
        loss_f =  criterion(forecast * scale, ty * scale)
        loss_m = criterion(res * scale, ty * scale)
        final_loss  += loss_f.item()
        min_loss  += loss_m.item()
        n_samples += (forecast.size(0) * data.m)
        grad_norm = optim.step()

        if iter%100==0:
            print('iter:{:3d} | loss: {:.7f}, loss_final: {:.7f}, loss_mid: {:.7f}'.format(iter,loss.item()/(forecast.size(0) * data.m),
                                                                                           loss_f.item()/(forecast.size(0) * data.m),loss_m.item()/(forecast.size(0) * data.m)))
        iter += 1

    writer.add_scalar('Train_loss_tatal', total_loss / n_samples, global_step=epoch)
    writer.add_scalar('Train_loss_Mid', min_loss / n_samples, global_step=epoch)
    writer.add_scalar('Train_loss_Final', final_loss / n_samples, global_step=epoch)
    print(
        '| Epoch | train_loss {:5.7f} | final_loss {:5.7f} | mid_loss {:5.7f} '.format(
            total_loss / n_samples, final_loss / n_samples, min_loss / n_samples), flush=True)
    return total_loss / n_samples



def evaluateEecoDeco(epoch, data, X, Y, model, evaluateL2, evaluateL1, batch_size):
    model.eval()
    total_loss = 0
    total_loss_l1 = 0

    total_loss_mid = 0
    total_loss_l1_mid = 0
    n_samples = 0
    predict = None
    res_mid = None
    test = None

    for X, Y in data.get_batches(X, Y, batch_size*200, False):
        # print('0')
        # X = torch.unsqueeze(X,dim=1)
        # X = X.transpose(2,3)
        with torch.no_grad():
            forecast, res = model(X) #torch.Size([32, 3, 137])
        forecast = torch.squeeze(forecast)
        res = torch.squeeze(res)
        if len(forecast.shape)==1:
            forecast = forecast.unsqueeze(dim=0)
            res = res.unsqueeze(dim=0)
        if predict is None:
            predict = forecast[:,-1,:].squeeze()
            res_mid = res[:,-1,:].squeeze()
            test = Y[:,-1,:].squeeze() #torch.Size([32, 3, 137])
            true = Y[:, -1, :].squeeze()
        else:
            predict = torch.cat((predict, forecast))
            res_mid = torch.cat((res_mid, res))
            test = torch.cat((test, Y))
        output = forecast[:,-1,:].squeeze()
        output_res = res[:,-1,:].squeeze()
        scale = data.scale.expand(output.size(0),data.m)
        # evaluation_pred =  (output * scale)
        # evaluation_true = (Y * scale)
        # evaluation_pred = evaluation_pred[:,-1,:].squeeze()
        # evaluation_true = evaluation_true[:,-1,:].squeeze()
        #
        # evaluation_res =  (res * scale)
        # evaluation_res = evaluation_res[:,-1,:].squeeze()
        total_loss += evaluateL2(output * scale, true * scale).item()
        total_loss_l1 += evaluateL1(output * scale, true * scale).item()
        total_loss_mid += evaluateL2(output_res * scale, true * scale).item()
        total_loss_l1_mid += evaluateL1(output_res * scale, true * scale).item()

        n_samples += (output.size(0) * data.m)

        #
        # total_loss += evaluateL2(evaluation_pred, evaluation_true).item()
        # total_loss_l1 += evaluateL1(evaluation_pred, evaluation_true).item()
        #
        # total_loss_mid += evaluateL2(evaluation_res, evaluation_true).item()
        # total_loss_l1_mid += evaluateL1(evaluation_res, evaluation_true).item()

        n_samples += (output.size(0) * data.m)

    rse = math.sqrt(total_loss / n_samples) / data.rse
    rae = (total_loss_l1 / n_samples) / data.rae

    predict = predict.data.cpu().numpy()
    Ytest = test.data.cpu().numpy()

    sigma_p = (predict).std(axis=0)
    sigma_g = (Ytest).std(axis=0)
    mean_p = predict.mean(axis=0)
    mean_g = Ytest.mean(axis=0)
    index = (sigma_g != 0)
    correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
    correlation = (correlation[index]).mean()
###############################Middle#######################################################
    rse_mid = math.sqrt(total_loss_mid / n_samples) / data.rse
    rae_mid = (total_loss_l1_mid / n_samples) / data.rae

    res_mid = res_mid.data.cpu().numpy()

    sigma_p = (res_mid).std(axis=0)
    sigma_g = (Ytest).std(axis=0)
    mean_p = res_mid.mean(axis=0)
    mean_g = Ytest.mean(axis=0)
    index = (sigma_g != 0)
    correlation_mid = ((res_mid - mean_p) * (Ytest - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
    correlation_mid = (correlation_mid[index]).mean()

    writer.add_scalar('Validation_final_rse', rse, global_step=epoch)
    writer.add_scalar('Validation_final_rae', rae, global_step=epoch)
    writer.add_scalar('Validation_final_corr', correlation, global_step=epoch)

    writer.add_scalar('Validation_mid_rse', rse_mid, global_step=epoch)
    writer.add_scalar('Validation_mid_rae', rae_mid, global_step=epoch)
    writer.add_scalar('Validation_mid_corr', correlation_mid, global_step=epoch)

    print(
        '|valid_final rse {:5.4f} | valid_final rae {:5.4f} | valid_final corr  {:5.4f}'.format(
            rse, rae, correlation), flush=True)

    print(
        '|valid_mid rse {:5.4f} | valid_mid rae {:5.4f} | valid_mid corr  {:5.4f}'.format(
            rse_mid, rae_mid, correlation_mid), flush=True)
    # if epoch%4==0:
    #     save_model(model, result_file,epoch=epoch)
    return rse, rae, correlation



def testEecoDeco(epoch, data, X, Y, model, evaluateL2, evaluateL1, batch_size):
    model.eval()
    total_loss = 0
    total_loss_l1 = 0

    total_loss_mid = 0
    total_loss_l1_mid = 0
    n_samples = 0
    predict = None
    res_mid = None
    test = None

    for X, Y in data.get_batches(X, Y, batch_size*200, False):
        # print('0')
        # X = torch.unsqueeze(X,dim=1)
        # X = X.transpose(2,3)
        with torch.no_grad():
            forecast, res = model(X) #torch.Size([32, 3, 137])
        forecast = torch.squeeze(forecast)
        res = torch.squeeze(res)
        if len(forecast.shape)==1:
            forecast = forecast.unsqueeze(dim=0)
            res = res.unsqueeze(dim=0)
        if predict is None:
            predict = forecast
            res_mid = res
            test = Y #torch.Size([32, 3, 137])
        else:
            predict = torch.cat((predict, forecast))
            res_mid = torch.cat((res_mid, res))
            test = torch.cat((test, Y))

        scale = data.scale.expand(forecast.size(0), args.horizon, data.m)
        evaluation_pred =  (forecast * scale)
        evaluation_true = (Y * scale)
        evaluation_pred = evaluation_pred[:,-1,:].squeeze()
        evaluation_true = evaluation_true[:,-1,:].squeeze()

        evaluation_res =  (res * scale)
        evaluation_res = evaluation_res[:,-1,:].squeeze()


        total_loss += evaluateL2(evaluation_pred, evaluation_true).item()
        total_loss_l1 += evaluateL1(evaluation_pred, evaluation_true).item()

        total_loss_mid += evaluateL2(evaluation_res, evaluation_true).item()
        total_loss_l1_mid += evaluateL1(evaluation_res, evaluation_true).item()

        n_samples += (forecast.size(0) * data.m)

    rse = math.sqrt(total_loss / n_samples) / data.rse
    rae = (total_loss_l1 / n_samples) / data.rae

    predict = predict.data.cpu().numpy()
    Ytest = test.data.cpu().numpy()

    sigma_p = (predict).std(axis=0)
    sigma_g = (Ytest).std(axis=0)
    mean_p = predict.mean(axis=0)
    mean_g = Ytest.mean(axis=0)
    index = (sigma_g != 0)
    correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
    correlation = (correlation[index]).mean()
###############################Middle#######################################################
    rse_mid = math.sqrt(total_loss_mid / n_samples) / data.rse
    rae_mid = (total_loss_l1_mid / n_samples) / data.rae

    res_mid = res_mid.data.cpu().numpy()

    sigma_p = (res_mid).std(axis=0)
    sigma_g = (Ytest).std(axis=0)
    mean_p = res_mid.mean(axis=0)
    mean_g = Ytest.mean(axis=0)
    index = (sigma_g != 0)
    correlation_mid = ((res_mid - mean_p) * (Ytest - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
    correlation_mid = (correlation_mid[index]).mean()

    writer.add_scalar('Test_final_rse', rse, global_step=epoch)
    writer.add_scalar('Test_final_rae', rae, global_step=epoch)
    writer.add_scalar('Test_final_corr', correlation, global_step=epoch)

    writer.add_scalar('Test_mid_rse', rse_mid, global_step=epoch)
    writer.add_scalar('Test_mid_rae', rae_mid, global_step=epoch)
    writer.add_scalar('Test_mid_corr', correlation_mid, global_step=epoch)

    print(
        '|Test_final rse {:5.4f} | Test_final rae {:5.4f} | Test_final corr  {:5.4f}'.format(
            rse, rae, correlation), flush=True)

    print(
        '|Test_mid rse {:5.4f} | Test_mid rae {:5.4f} | Test_mid corr  {:5.4f}'.format(
            rse_mid, rae_mid, correlation_mid), flush=True)
    # if epoch%4==0:
    #     save_model(model, result_file,epoch=epoch)
    return rse, rae, correlation




def evaluate(data, X, Y, model, evaluateL2, evaluateL1, batch_size):
    model.eval()
    total_loss = 0
    total_loss_l1 = 0
    n_samples = 0
    predict = None
    test = None

    for X, Y in data.get_batches(X, Y, batch_size*200, False):
        print('0')
        # X = torch.unsqueeze(X,dim=1)
        # X = X.transpose(2,3)
        with torch.no_grad():
            output = model(X) #torch.Size([32, 3, 137])
        output = torch.squeeze(output)
        if len(output.shape)==1:
            output = output.unsqueeze(dim=0)
        if predict is None:
            predict = output
            test = Y #torch.Size([32, 3, 137])
        else:
            predict = torch.cat((predict, output))
            test = torch.cat((test, Y))

        scale = data.scale.expand(output.size(0), args.horizon, data.m)
        evaluation_pred =  (output * scale)
        evaluation_true = (Y * scale)
        evaluation_pred = evaluation_pred[:,-1,:].squeeze()
        evaluation_true = evaluation_true[:,-1,:].squeeze()
        # total_loss += evaluateL2(output * scale, Y * scale).item()
        # total_loss_l1 += evaluateL1(output * scale, Y * scale).item()
        total_loss += evaluateL2(evaluation_pred, evaluation_true).item()
        total_loss_l1 += evaluateL1(evaluation_pred, evaluation_true).item()

        n_samples += (output.size(0) * data.m)

    rse = math.sqrt(total_loss / n_samples) / data.rse
    rae = (total_loss_l1 / n_samples) / data.rae

    predict = predict.data.cpu().numpy()
    Ytest = test.data.cpu().numpy()
    # np.save('F:\\school\\Papers\\timeseriesNew\\MTGNN-master\\output\\Ytest_nolinear.npy', Ytest)
    sigma_p = (predict).std(axis=0)
    sigma_g = (Ytest).std(axis=0)
    mean_p = predict.mean(axis=0)
    mean_g = Ytest.mean(axis=0)
    index = (sigma_g != 0)
    correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
    correlation = (correlation[index]).mean()

    # if epoch%4==0:
    #     save_model(model, result_file,epoch=epoch)
    return rse, rae, correlation

def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj== 1:
        lr_adjust = {epoch: args.lr * (0.95 ** ((epoch-1) // 1))}
    elif args.lradj==2:

        lr_adjust = {
            20: 0.0005, 40: 0.0001, 60: 0.00005, 80: 0.00001

        }
    elif args.lradj==3:

        lr_adjust = {
            20: 0.0005, 25: 0.0001, 35: 0.00005, 55: 0.00001
            , 70: 0.000001
        }
    elif args.lradj==4:

        lr_adjust = {
            30: 0.0005, 40: 0.0003, 50: 0.0001, 65: 0.00001
            , 80: 0.000001
        }
    elif args.lradj==5:

        lr_adjust = {
            40: 0.0001, 60: 0.00005
        }
    elif args.lradj==6:

        lr_adjust = {
            0: 0.0001, 5: 0.0005, 10:0.001, 20: 0.0001, 30: 0.00005, 40: 0.00001
            , 70: 0.000001
        }
    elif args.lradj==61:

        lr_adjust = {
            0: 0.0001, 5: 0.0005, 10:0.001, 25: 0.0005, 35: 0.0001, 45: 0.00001
            , 70: 0.000001
        }

    elif args.lradj==7:

        lr_adjust = {
            10: 0.0001, 30: 0.00005, 50: 0.00001
            , 70: 0.000001
        }

    elif args.lradj==8:

        lr_adjust = {
            0: 0.0005, 5: 0.0008, 10:0.001, 20: 0.0001, 30: 0.00005, 40: 0.00001
            , 70: 0.000001
        }
    elif args.lradj==9:

        lr_adjust = {
            0: 0.0001, 10: 0.0005, 20:0.001, 40: 0.0001, 45: 0.00005, 50: 0.00001
            , 70: 0.000001
        }

    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))



def train(epoch, data, X, Y, model, criterion, optim, batch_size):
    model.train()
    total_loss = 0
    n_samples = 0
    iter = 0


    for X, Y in data.get_batches(X, Y, batch_size, True):
        model.zero_grad()             #torch.Size([32, 168, 137])
        # X = torch.unsqueeze(X,dim=1)  #torch.Size([32, 1, 168, 137])
        # X = X.transpose(2,3)          #torch.Size([32, 1, 137, 168])

        # X = X[:, :, 0:-1]
        # Y = Y[:, 0:-1]
        if iter % args.step_size == 0: #100
            perm = np.random.permutation(range(args.num_nodes)) #137
        num_sub = int(args.num_nodes / args.num_split)

        for j in range(args.num_split):
            if j != args.num_split - 1:
                id = perm[j * num_sub:(j + 1) * num_sub]
            else:
                id = perm[j * num_sub:]
            id = torch.tensor(id).to(device)
            id = id.long()
            tx = X#[:, id, :] #torch.Size([32, 1, 137, 168])
            ty = Y#[:, id]       #torch.Size([32, 137])
            # output = model(tx,id) #torch.Size([32, 1, 137, 1])
            output = model(tx)  # torch.Size([32, 1, 137, 1])
            output = torch.squeeze(output)
            scale = data.scale.expand(output.size(0), args.horizon, data.m)
            #scale = scale[:,id]
            loss = criterion(output * scale, ty * scale)
            loss.backward()
            total_loss += loss.item()
            n_samples += (output.size(0) * data.m)
            grad_norm = optim.step()

        if iter%100==0:
            print('iter:{:3d} | loss: {:.3f}'.format(iter,loss.item()/(output.size(0) * data.m)))
        iter += 1
    return total_loss / n_samples




def main_run():

    Data = DataLoaderH(args.data, 0.6, 0.2, device, args.horizon, args.seq_in_len, args.normalize)


    part = [[1, 1], [1, 1], [1, 1], [0, 0], [0, 0], [0, 0], [0, 0]]  # Best model
    # part = [[1, 1], [0, 0], [0, 0]]  # Best model
    model = WASN(args, num_classes=args.horizon, num_stacks = 1, first_conv=args.num_nodes,
                 number_levels=len(part),
                 number_level_part=part).cuda()
    model = model.to(device)



    # print(args)
    # print('The recpetive field size is', model.receptive_field)
    nParams = sum([p.nelement() for p in model.parameters()])
    print('Number of model parameters is', nParams, flush=True)

    if args.L1Loss:
        criterion = smooth_l1_loss #nn.L1Loss(size_average=False).to(device)
    else:
        criterion = nn.MSELoss(size_average=False).to(device)
    evaluateL2 = nn.MSELoss(size_average=False).to(device)
    evaluateL1 = nn.L1Loss(size_average=False).to(device)


    best_val = 10000000

    optim = torch.optim.Adam(params=model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-5)

    infer = 0
    if infer:
        model = load_model('F:\\school\\Papers\\timeseriesNew\\MTGNN-master\\output', epoch=40)
        val_loss, val_rae, val_corr = evaluate(Data, Data.test[0], Data.test[1], model, evaluateL2, evaluateL1,
                                               args.batch_size)
        print(
            '|valid rse {:5.4f} | valid rae {:5.4f} | valid corr  {:5.4f}'.format(
                val_loss, val_rae, val_corr), flush=True)
    # At any point you can hit Ctrl + C to break out of training early.
    else:
        try:
            print('begin training')
            for epoch in range(1, args.epochs + 1):
                epoch_start_time = time.time()
                adjust_learning_rate(optim, epoch, args)
                train_loss = trainEecoDeco(epoch, Data, Data.train[0], Data.train[1], model, criterion, optim, args.batch_size)
                val_loss, val_rae, val_corr = evaluateEecoDeco(epoch, Data, Data.valid[0], Data.valid[1], model, evaluateL2, evaluateL1,
                                                   args.batch_size)
                test_loss, test_rae, test_corr = testEecoDeco(epoch, Data, Data.test[0], Data.test[1], model, evaluateL2,
                                                               evaluateL1,
                                                               args.batch_size)


                print(
                    '| end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.4f} | valid rse {:5.4f} | valid rae {:5.4f} | valid corr  {:5.4f}|'
                    ' test rse {:5.4f} | test rae {:5.4f} | test corr  {:5.4f}'.format(
                        epoch, (time.time() - epoch_start_time), train_loss, val_loss, val_rae, val_corr, test_loss, test_rae, test_corr), flush=True)
                # Save the model if the validation loss is the best we've seen so far.

                if val_loss < best_val:
                    # with open(args.save, 'wb') as f:
                    #     torch.save(model, f)
                    best_val = val_loss
                    print(
                        '--------------| Best Val loss |--------------')
                # if epoch % 5 == 0:
                #     test_acc, test_rae, test_corr = evaluate(Data, Data.test[0], Data.test[1], model, evaluateL2, evaluateL1,
                #                                          args.batch_size)
                #     print("test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f}".format(test_acc, test_rae, test_corr), flush=True)

        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')



    vtest_acc, vtest_rae, vtest_corr = evaluateEecoDeco(Data, Data.valid[0], Data.valid[1], model, evaluateL2, evaluateL1,
                                         args.batch_size)
    test_acc, test_rae, test_corr = evaluateEecoDeco(Data, Data.test[0], Data.test[1], model, evaluateL2, evaluateL1,
                                         args.batch_size)
    print("final test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f}".format(test_acc, test_rae, test_corr))
    return vtest_acc, vtest_rae, vtest_corr, test_acc, test_rae, test_corr

if __name__ == "__main__":

    torch.manual_seed(4321)  # reproducible
    torch.cuda.manual_seed_all(4321)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True  # Can change it to False --> default: False
    torch.backends.cudnn.enabled = True
    writer = SummaryWriter('./run_financial/{}'.format(args.model_name))
    vacc = []
    vrae = []
    vcorr = []
    acc = []
    rae = []
    corr = []
    for i in range(1):
        print('===================================================================')
        print('Num of runs: {}', i)
        val_acc, val_rae, val_corr, test_acc, test_rae, test_corr = main_run()
        vacc.append(val_acc)
        vrae.append(val_rae)
        vcorr.append(val_corr)
        acc.append(test_acc)
        rae.append(test_rae)
        corr.append(test_corr)
    print('\n\n')
    print('1 runs average')
    print('\n\n')
    print("valid\trse\trae\tcorr")
    print("mean\t{:5.4f}\t{:5.4f}\t{:5.4f}".format(np.mean(vacc), np.mean(vrae), np.mean(vcorr)))
    print("std\t{:5.4f}\t{:5.4f}\t{:5.4f}".format(np.std(vacc), np.std(vrae), np.std(vcorr)))
    print('\n\n')
    print("test\trse\trae\tcorr")
    print("mean\t{:5.4f}\t{:5.4f}\t{:5.4f}".format(np.mean(acc), np.mean(rae), np.mean(corr)))
    print("std\t{:5.4f}\t{:5.4f}\t{:5.4f}".format(np.std(acc), np.std(rae), np.std(corr)))

