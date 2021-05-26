import argparse
import time
import torch.nn as nn
from util_financial import *
# from StackTWNet import WASN
# from models.StackTWaveNetTransformerEncoder import WASN
# from models.OriginalStackTWaveNetTransformerEncoder import WASN
from models.IDCN import IDCNet
from models.IDCN_Ecoder import IDCNetEcoder
# from models.TCN import TCN
from tensorboardX import SummaryWriter
import math

parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
parser.add_argument('--data', type=str, default='./dataset/solar_AL.txt',
                    help='location of the data file')
parser.add_argument('--log_interval', type=int, default=2000, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model/model.pt',
                    help='path to save the final model')
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--L1Loss', type=bool, default=True)
parser.add_argument('--normalize', type=int, default=2)
parser.add_argument('--device',type=str,default='cuda:0',help='')

parser.add_argument('--num_nodes',type=int,default=8,help='number of nodes/variables')

parser.add_argument('--batch_size',type=int,default=8,help='batch size')
parser.add_argument('--lr',type=float,default=5e-3,help='learning rate')
parser.add_argument('--weight_decay',type=float,default=0.00001,help='weight decay rate')
parser.add_argument('--epochs',type=int,default=100,help='')
parser.add_argument('--hidden-size', default=1.0, type=float, help='hidden channel of module')
parser.add_argument('--INN', default=1, type=int, help='use INN or basic strategy')
parser.add_argument('--kernel', default=5, type=int, help='kernel size')
parser.add_argument('--dilation', default=1, type=int, help='dilation')
parser.add_argument('--lradj', type=int, default=6,help='adjust learning rate')

parser.add_argument('--model_name', type=str, default='EncoDeco')
parser.add_argument('--model_mode', type=str, default='EncoDeco')
parser.add_argument('--positionalEcoding', type = bool , default=False)

parser.add_argument('--window_size', type=int, default=168) # input size
parser.add_argument('--horizon', type=int, default=3)  # predication

parser.add_argument('--dropout', type=float, default=0.5)

parser.add_argument('--num_concat', type=int, default=165)

parser.add_argument('--single_step', type=int, default=0)
parser.add_argument('--single_step_output_One', type=int, default=0)
parser.add_argument('--lastWeight', type=float, default=1.0)

parser.add_argument('--groups', type=int, default=1)
parser.add_argument('--layers', type=int, default=3)

parser.add_argument('--save_path', type=str, default='./SingleStepCheckpoint')
parser.add_argument('--dataset_name', type=str, default='electricity')

#TCN
parser.add_argument('--levels', type=int, default=1,
                    help='# of levels (default: 8)')
parser.add_argument('--nhid', type=int, default=32,
                    help='number of hidden units per layer (default: 30)')


args = parser.parse_args()
device = torch.device(args.device)
torch.set_num_threads(3)
args.num_concat = args.window_size - args.horizon
#args.window_size = args.window_size - args.horizon
print(args)

if args.data == './dataset/electricity.txt':
    args.num_nodes = 321
    args.dataset_name = 'electricity'

if args.data == './dataset/solar_AL.txt':
    args.num_nodes = 137
    args.dataset_name = 'solar_AL'

if args.data == './dataset/exchange_rate.txt':
    args.num_nodes = 8
    args.dataset_name = 'exchange_rate'

if args.data == './dataset/traffic.txt':
    args.num_nodes = 862
    args.dataset_name = 'traffic'

print('dataset {}, the channel size is {}'.format(args.data, args.num_nodes))







def trainEecoDeco(epoch, data, X, Y, model, criterion, optim, batch_size):
    model.train()
    total_loss = 0
    n_samples = 0
    iter = 0
    final_loss = 0
    min_loss = 0

    for X, Y in data.get_batches(X, Y, batch_size, True):
        model.zero_grad()             #torch.Size([32, 168, 137])


        tx = X
        ty = Y

        forecast, res = model(tx)
        # forecast = torch.squeeze(forecast)
        scale = data.scale.expand(forecast.size(0), args.horizon, data.m)
        bias = data.bias.expand(forecast.size(0), args.horizon, data.m)
        weight = torch.tensor(args.lastWeight).to(device)

        # if args.normalize == 3:
        #     # loss = criterion(forecast, ty) + criterion(res, ty)
        # else:
            # loss = criterion(forecast * scale + bias, ty * scale + bias) + criterion(res * scale + bias, ty * scale + bias)



        if args.normalize == 3:
            if args.lastWeight == 1.0:
                loss_f = criterion(forecast, ty)
                loss_m = criterion(res, ty)
            else:

                loss_f = criterion(forecast[:, :-1, :] ,
                                   ty[:, :-1, :] ) \
                         + weight * criterion(forecast[:, -1:, :],
                                              ty[:, -1:, :] )
                loss_m = criterion(res[:, :-1, :] ,
                                   ty[:, :-1, :] ) \
                         + weight * criterion(res[:, -1:, :],
                                              ty[:, -1:, :] )
        else:
            if args.lastWeight == 1.0:
                loss_f = criterion(forecast * scale + bias, ty * scale + bias)
                loss_m = criterion(res * scale + bias, ty * scale + bias)
            else:

                loss_f = criterion(forecast[:, :-1, :] * scale[:, :-1, :] + bias[:, :-1, :],
                                 ty[:, :-1, :] * scale[:, :-1, :] + bias[:, :-1, :]) \
                       + weight * criterion(forecast[:, -1:, :] * scale[:, -1:, :] + bias[:, -1:, :],
                                            ty[:, -1:, :] * scale[:, -1:, :] + bias[:, -1:, :])
                print(criterion(forecast[:, :-1, :] * scale[:, :-1, :] + bias[:, :-1, :],ty[:, :-1, :] * scale[:, :-1, :] + bias[:, :-1, :]),weight * criterion(forecast[:, -1:, :] * scale[:, -1:, :] + bias[:, -1:, :],ty[:, -1:, :] * scale[:, -1:, :] + bias[:, -1:, :]))
                loss_m = criterion(res[:, :-1, :] * scale[:, :-1, :] + bias[:, :-1, :],
                                 ty[:, :-1, :] * scale[:, :-1, :] + bias[:, :-1, :]) \
                       + weight * criterion(res[:, -1:, :] * scale[:, -1:, :] + bias[:, -1:, :],
                                            ty[:, -1:, :] * scale[:, -1:, :] + bias[:, -1:, :])

        loss = loss_f+loss_m

        loss.backward()
        total_loss += loss.item()

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



def evaluateEecoDeco(epoch, data, X, Y, model, evaluateL2, evaluateL1, batch_size,writer):
    model.eval()
    total_loss = 0
    total_loss_l1 = 0

    total_loss_mid = 0
    total_loss_l1_mid = 0
    n_samples = 0
    predict = None
    res_mid = None
    test = None

    forecast_set = []
    Mid_set = []
    target_set = []

    for X, Y in data.get_batches(X, Y, batch_size*200, False):
        # print('0')
        # X = torch.unsqueeze(X,dim=1)
        # X = X.transpose(2,3)
        with torch.no_grad():
            forecast, res = model(X) #torch.Size([32, 3, 137])
        # forecast = torch.squeeze(forecast)
        # res = torch.squeeze(res)
        true = Y[:, -1, :].squeeze()


        forecast_set.append(forecast)
        Mid_set.append(res)
        target_set.append(Y)

        if len(forecast.shape)==1:
            forecast = forecast.unsqueeze(dim=0)
            res = res.unsqueeze(dim=0)
        if predict is None:
            predict = forecast[:,-1,:].squeeze()
            res_mid = res[:,-1,:].squeeze()
            test = Y[:,-1,:].squeeze() #torch.Size([32, 3, 137])

        else:
            predict = torch.cat((predict, forecast[:,-1,:].squeeze()))
            res_mid = torch.cat((res_mid, res[:,-1,:].squeeze()))
            test = torch.cat((test, Y[:, -1, :].squeeze()))
        output = forecast[:,-1,:].squeeze()
        output_res = res[:,-1,:].squeeze()
        scale = data.scale.expand(output.size(0),data.m)
        bias = data.bias.expand(output.size(0), data.m)


        total_loss += evaluateL2(output * scale + bias, true * scale+ bias).item()
        total_loss_l1 += evaluateL1(output * scale+ bias, true * scale+ bias).item()
        total_loss_mid += evaluateL2(output_res * scale+ bias, true * scale+ bias).item()
        total_loss_l1_mid += evaluateL1(output_res * scale+ bias, true * scale+ bias).item()

        n_samples += (output.size(0) * data.m)

    forecast_Norm = torch.cat(forecast_set, axis=0)
    target_Norm = torch.cat(target_set, axis=0)
    Mid_Norm = torch.cat(Mid_set, axis=0)


    rse_final_each = []
    rae_final_each = []
    corr_final_each = []
    Scale = data.scale.expand(forecast_Norm.size(0),data.m)
    bias = data.bias.expand(forecast_Norm.size(0),data.m)
    for i in range(forecast_Norm.shape[1]):
        lossL2_F = evaluateL2(forecast_Norm[:,i,:] * Scale + bias, target_Norm[:,i,:] * Scale+ bias).item()
        lossL1_F = evaluateL1(forecast_Norm[:,i,:] * Scale+ bias, target_Norm[:,i,:] * Scale+ bias).item()
        lossL2_M = evaluateL2(Mid_Norm[:, i, :] * Scale+ bias, target_Norm[:, i, :] * Scale+ bias).item()
        lossL1_M = evaluateL1(Mid_Norm[:, i, :] * Scale+ bias, target_Norm[:, i, :] * Scale+ bias).item()
        rse_F = math.sqrt(lossL2_F / forecast_Norm.shape[0]/ data.m) / data.rse
        rae_F = (lossL1_F / forecast_Norm.shape[0]/ data.m) / data.rae
        rse_final_each.append(rse_F.item())
        rae_final_each.append(rae_F.item())

        pred = forecast_Norm[:,i,:].data.cpu().numpy()
        y_true = target_Norm[:,i,:].data.cpu().numpy()

        sig_p = (pred).std(axis=0)
        sig_g = (y_true).std(axis=0)
        m_p = pred.mean(axis=0)
        m_g = y_true.mean(axis=0)
        ind = (sig_g != 0)
        corr = ((pred - m_p) * (y_true - m_g)).mean(axis=0) / (sig_p * sig_g)
        corr = (corr[ind]).mean()
        corr_final_each.append(corr)

    # print('Valid_Each_final Rse:', rse_final_each)
    # print('Valid_Each_final Rae:', rae_final_each)
    # print('Valid_Each_final Corr:', corr_final_each)

    # rse = math.sqrt(total_loss / n_samples) / data.rse
    # rae = (total_loss_l1 / n_samples) / data.rae
    #
    # predict = predict.data.cpu().numpy()
    # Ytest = test.data.cpu().numpy()
    #
    # sigma_p = (predict).std(axis=0)
    # sigma_g = (Ytest).std(axis=0)
    # mean_p = predict.mean(axis=0)
    # mean_g = Ytest.mean(axis=0)
    # index = (sigma_g != 0)
    # correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
    # correlation = (correlation[index]).mean()
###############################Middle#######################################################
    rse = math.sqrt(total_loss / n_samples) / data.rse
    rae = (total_loss_l1 / n_samples) / data.rae

    rse_mid = math.sqrt(total_loss_mid / n_samples) / data.rse
    rae_mid = (total_loss_l1_mid / n_samples) / data.rae

    predict = forecast_Norm.cpu().numpy()
    Ytest = target_Norm.cpu().numpy()

    sigma_p = (predict).std(axis=0)
    sigma_g = (Ytest).std(axis=0)
    mean_p = predict.mean(axis=0)
    mean_g = Ytest.mean(axis=0)
    index = (sigma_g != 0)
    correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
    correlation = (correlation[index]).mean()

    mid_pred = Mid_Norm.cpu().numpy()
    sigma_p = (mid_pred).std(axis=0)
    mean_p = mid_pred.mean(axis=0)
    correlation_mid = ((mid_pred - mean_p) * (Ytest - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
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



def testEecoDeco(epoch, data, X, Y, model, evaluateL2, evaluateL1, batch_size, writer):
    model.eval()
    total_loss = 0
    total_loss_l1 = 0

    total_loss_mid = 0
    total_loss_l1_mid = 0
    n_samples = 0
    predict = None
    res_mid = None
    test = None

    forecast_set = []
    Mid_set = []
    target_set = []

    for X, Y in data.get_batches(X, Y, batch_size*10, False):
        # print('0')
        # X = torch.unsqueeze(X,dim=1)
        # X = X.transpose(2,3)
        with torch.no_grad():
            forecast, res = model(X) #torch.Size([32, 3, 137])
        # forecast = torch.squeeze(forecast)
        # res = torch.squeeze(res)
        true = Y[:, -1, :].squeeze()

        forecast_set.append(forecast)
        Mid_set.append(res)
        target_set.append(Y)

        if len(forecast.shape)==1:
            forecast = forecast.unsqueeze(dim=0)
            res = res.unsqueeze(dim=0)
        if predict is None:
            predict = forecast[:, -1, :].squeeze()
            res_mid = res[:, -1, :].squeeze()
            test = Y[:, -1, :].squeeze()  # torch.Size([32, 3, 137])

        else:
            predict = torch.cat((predict, forecast[:, -1, :].squeeze()))
            res_mid = torch.cat((res_mid, res[:, -1, :].squeeze()))
            test = torch.cat((test, Y[:, -1, :].squeeze()))
        output = forecast[:, -1, :].squeeze()
        output_res = res[:, -1, :].squeeze()
        scale = data.scale.expand(output.size(0), data.m)
        bias = data.bias.expand(output.size(0), data.m)

        total_loss += evaluateL2(output * scale + bias, true * scale+ bias).item()
        total_loss_l1 += evaluateL1(output * scale + bias, true * scale+ bias).item()
        total_loss_mid += evaluateL2(output_res * scale + bias, true * scale+ bias).item()
        total_loss_l1_mid += evaluateL1(output_res * scale + bias, true * scale+ bias).item()

        n_samples += (output.size(0) * data.m)

    forecast_Norm = torch.cat(forecast_set, axis=0)
    target_Norm = torch.cat(target_set, axis=0)
    Mid_Norm = torch.cat(Mid_set, axis=0)

    rse_final_each = []
    rae_final_each = []
    corr_final_each = []
    Scale = data.scale.expand(forecast_Norm.size(0), data.m)
    bias = data.bias.expand(forecast_Norm.size(0), data.m)
    for i in range(forecast_Norm.shape[1]):
        lossL2_F = evaluateL2(forecast_Norm[:, i, :] * Scale + bias, target_Norm[:, i, :] * Scale + bias).item()
        lossL1_F = evaluateL1(forecast_Norm[:, i, :] * Scale+ bias, target_Norm[:, i, :] * Scale+ bias).item()
        lossL2_M = evaluateL2(Mid_Norm[:, i, :] * Scale + bias, target_Norm[:, i, :] * Scale+ bias).item()
        lossL1_M = evaluateL1(Mid_Norm[:, i, :] * Scale + bias, target_Norm[:, i, :] * Scale+ bias).item()
        rse_F = math.sqrt(lossL2_F / forecast_Norm.shape[0] / data.m) / data.rse
        rae_F = (lossL1_F / forecast_Norm.shape[0] / data.m) / data.rae
        rse_final_each.append(rse_F.item())
        rae_final_each.append(rae_F.item())

        pred = forecast_Norm[:, i, :].data.cpu().numpy()
        y_true = target_Norm[:, i, :].data.cpu().numpy()

        sig_p = (pred).std(axis=0)
        sig_g = (y_true).std(axis=0)
        m_p = pred.mean(axis=0)
        m_g = y_true.mean(axis=0)
        ind = (sig_g != 0)
        corr = ((pred - m_p) * (y_true - m_g)).mean(axis=0) / (sig_p * sig_g)
        corr = (corr[ind]).mean()
        corr_final_each.append(corr)

    # print('TEST_Each_final Rse:', rse_final_each)
    # print('TEST_Each_final Rae:', rae_final_each)
    # print('TEST_Each_final Corr:', corr_final_each)


    # rse = math.sqrt(total_loss / n_samples) / data.rse
    # rae = (total_loss_l1 / n_samples) / data.rae
    #
    # predict = predict.data.cpu().numpy()
    # Ytest = test.data.cpu().numpy()
    #
    # sigma_p = (predict).std(axis=0)
    # sigma_g = (Ytest).std(axis=0)
    # mean_p = predict.mean(axis=0)
    # mean_g = Ytest.mean(axis=0)
    # index = (sigma_g != 0)
    # correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
    # correlation = (correlation[index]).mean()
    # ###############################Middle#######################################################
    # rse_mid = math.sqrt(total_loss_mid / n_samples) / data.rse
    # rae_mid = (total_loss_l1_mid / n_samples) / data.rae
    #
    # res_mid = res_mid.data.cpu().numpy()
    #
    # sigma_p = (res_mid).std(axis=0)
    # sigma_g = (Ytest).std(axis=0)
    # mean_p = res_mid.mean(axis=0)
    # mean_g = Ytest.mean(axis=0)
    # index = (sigma_g != 0)
    # correlation_mid = ((res_mid - mean_p) * (Ytest - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
    # correlation_mid = (correlation_mid[index]).mean()
#===================
    rse = math.sqrt(total_loss / n_samples) / data.rse
    rae = (total_loss_l1 / n_samples) / data.rae

    rse_mid = math.sqrt(total_loss_mid / n_samples) / data.rse
    rae_mid = (total_loss_l1_mid / n_samples) / data.rae

    predict = forecast_Norm.cpu().numpy()
    Ytest = target_Norm.cpu().numpy()

    sigma_p = (predict).std(axis=0)
    sigma_g = (Ytest).std(axis=0)
    mean_p = predict.mean(axis=0)
    mean_g = Ytest.mean(axis=0)
    index = (sigma_g != 0)
    correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
    correlation = (correlation[index]).mean()

    mid_pred = Mid_Norm.cpu().numpy()
    sigma_p = (mid_pred).std(axis=0)
    mean_p = mid_pred.mean(axis=0)
    correlation_mid = ((mid_pred - mean_p) * (Ytest - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
    correlation_mid = (correlation_mid[index]).mean()



    writer.add_scalar('Test_final_rse', rse, global_step=epoch)
    writer.add_scalar('Test_final_rae', rae, global_step=epoch)
    writer.add_scalar('Test_final_corr', correlation, global_step=epoch)

    writer.add_scalar('Test_mid_rse', rse_mid, global_step=epoch)
    writer.add_scalar('Test_mid_rae', rae_mid, global_step=epoch)
    writer.add_scalar('Test_mid_corr', correlation_mid, global_step=epoch)

    print(
        '|Test_final rse {:5.4f} | Test_final rae {:5.4f} | Test_final corr   {:5.4f}'.format(
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
            1: 0.0005, 5: 0.0008, 10:0.001, 20: 0.0005, 30: 0.0001, 40: 0.00005,50: 0.00001
        }
    elif args.lradj==6:

        lr_adjust = {
            1: 0.0001, 5: 0.0005, 10:0.001, 20: 0.0001, 30: 0.00005, 40: 0.00001
            , 70: 0.000001
        }
    elif args.lradj==61:

        lr_adjust = {
            1: 0.0001, 5: 0.0005, 10:0.001, 25: 0.0005, 35: 0.0001, 45: 0.00001
            , 70: 0.000001
        }

    elif args.lradj==7:

        lr_adjust = {
            10: 0.0001, 30: 0.00005, 50: 0.00001
            , 70: 0.000001
        }

    elif args.lradj==8:

        lr_adjust = {
            1: 0.0005, 5: 0.0008, 10:0.001, 20: 0.0001, 30: 0.00005, 40: 0.00001
            , 70: 0.000001
        }
    elif args.lradj==9:

        lr_adjust = {
            1: 0.0001, 10: 0.0005, 20:0.001, 40: 0.0001, 45: 0.00005, 50: 0.00001
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


def trainEeco(epoch, data, X, Y, model, criterion, optim, batch_size):
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
        forecast = model(tx) #torch.Size([64, 3, 8])
        # forecast = torch.squeeze(forecast)
        scale = data.scale.expand(forecast.size(0), args.horizon, data.m)
        bias = data.bias.expand(forecast.size(0), args.horizon, data.m)

        # loss = criterion(forecast * scale+ bias, ty * scale+ bias)
        weight = torch.tensor(args.lastWeight).to(device)
        # a = criterion(forecast[:, -1:, :] * scale[:, -1:, :] + bias[:, -1:, :],
        #           ty[:, -1:, :] * scale[:, -1:, :] + bias[:, -1:, :])

        # if args.normalize == 3:
#        if args.lastWeight == 1.0:
#        loss = criterion(forecast, ty)
#        else:
#<<<<<<< HEAD
        if args.lastWeight == 1.0:
             loss = criterion(forecast * scale + bias, ty * scale + bias)
#             loss = criterion(forecast, ty)
        else:
             loss = criterion(forecast[:,:-1,:] * scale[:,:-1,:] + bias[:,:-1,:], ty[:,:-1,:] * scale[:,:-1,:] + bias[:,:-1,:])\
                       +  weight * criterion(forecast[:,-1:,:] * scale[:,-1:,:] + bias[:,-1:,:], ty[:,-1:,:] * scale[:,-1:,:] + bias[:,-1:,:])
             print(criterion(forecast[:,:-1,:] * scale[:,:-1,:] + bias[:,:-1,:], ty[:,:-1,:] * scale[:,:-1,:] + bias[:,:-1,:]),weight * criterion(forecast[:,-1:,:] * scale[:,-1:,:] + bias[:,-1:,:], ty[:,-1:,:] * scale[:,-1:,:] + bias[:,-1:,:]))
#             loss = criterion(forecast[:, :-1, :],
#                             ty[:, :-1, :])  \
#                   + weight * criterion(forecast[:, -1:, :] ,
#                                             ty[:, -1:, :] )
        # else:
        #     if args.lastWeight == 1.0:
        #         loss = criterion(forecast * scale + bias, ty * scale + bias)
        #
        #     # loss2 = criterion(forecast[:, :-1, :] * scale[:, :-1, :] + bias[:, :-1, :],
        #     #                  ty[:, :-1, :] * scale[:, :-1, :] + bias[:, :-1, :])
        #     # loss3 = criterion(forecast[:,-1:,:] * scale[:,-1:,:] + bias[:,-1:,:], ty[:,-1:,:] * scale[:,-1:,:] + bias[:,-1:,:])
        #     else:
        #         loss = criterion(forecast[:,:-1,:] * scale[:,:-1,:] + bias[:,:-1,:], ty[:,:-1,:] * scale[:,:-1,:] + bias[:,:-1,:])\
        #                +  weight * criterion(forecast[:,-1:,:] * scale[:,-1:,:] + bias[:,-1:,:], ty[:,-1:,:] * scale[:,-1:,:] + bias[:,-1:,:])

        loss.backward()
        total_loss += loss.item()
        #final_loss  += loss_f.item()

        n_samples += (forecast.size(0) * data.m)
        grad_norm = optim.step()

        if iter%100==0:
            print('iter:{:3d} | loss: {:.7f}, loss_final: {:.7f}'.format(iter,loss.item()/(forecast.size(0) * data.m),
                                                                                           loss.item()/(forecast.size(0) * data.m)))
        iter += 1

    writer.add_scalar('Train_loss_tatal', total_loss / n_samples, global_step=epoch)
    writer.add_scalar('Train_loss_Final', final_loss / n_samples, global_step=epoch)
    print(
        '| Epoch | train_loss {:5.7f} | final_loss {:5.7f} | mid_loss {:5.7f} '.format(
            total_loss / n_samples, final_loss / n_samples, min_loss / n_samples), flush=True)
    return total_loss / n_samples



def evaluateEeco(epoch, data, X, Y, model, evaluateL2, evaluateL1, batch_size,writer):
    model.eval()
    total_loss = 0
    total_loss_l1 = 0

    total_loss_mid = 0
    total_loss_l1_mid = 0
    n_samples = 0
    predict = None
    res_mid = None
    test = None

    forecast_set = []
    Mid_set = []
    target_set = []

    for X, Y in data.get_batches(X, Y, batch_size*200, False):
        # print('0')
        # X = torch.unsqueeze(X,dim=1)
        # X = X.transpose(2,3)
        with torch.no_grad():
            forecast = model(X) #torch.Size([32, 3, 137])
        # forecast = torch.squeeze(forecast)
        # res = torch.squeeze(res)
        true = Y[:, -1, :].squeeze()


        forecast_set.append(forecast)

        target_set.append(Y)

        if len(forecast.shape)==1:
            forecast = forecast.unsqueeze(dim=0)

        if predict is None:
            predict = forecast[:,-1,:].squeeze()

            test = Y[:,-1,:].squeeze() #torch.Size([32, 3, 137])

        else:
            predict = torch.cat((predict, forecast[:,-1,:].squeeze()))

            test = torch.cat((test, Y[:, -1, :].squeeze()))
        output = forecast[:,-1,:].squeeze()

        scale = data.scale.expand(output.size(0),data.m)
        bias = data.bias.expand(output.size(0), data.m)


        total_loss += evaluateL2(output * scale+bias, true * scale+bias).item()
        total_loss_l1 += evaluateL1(output * scale+bias, true * scale+bias).item()


        n_samples += (output.size(0) * data.m)

    forecast_Norm = torch.cat(forecast_set, axis=0)
    target_Norm = torch.cat(target_set, axis=0)



    rse_final_each = []
    rae_final_each = []
    corr_final_each = []
    Scale = data.scale.expand(forecast_Norm.size(0),data.m)
    bias = data.bias.expand(forecast_Norm.size(0), data.m)
    for i in range(forecast_Norm.shape[1]):
        lossL2_F = evaluateL2(forecast_Norm[:,i,:] * Scale+bias, target_Norm[:,i,:] * Scale+bias).item()
        lossL1_F = evaluateL1(forecast_Norm[:,i,:] * Scale+bias, target_Norm[:,i,:] * Scale+bias).item()

        rse_F = math.sqrt(lossL2_F / forecast_Norm.shape[0]/ data.m) / data.rse
        rae_F = (lossL1_F / forecast_Norm.shape[0]/ data.m) / data.rae
        rse_final_each.append(rse_F.item())
        rae_final_each.append(rae_F.item())

        pred = forecast_Norm[:,i,:].data.cpu().numpy()
        y_true = target_Norm[:,i,:].data.cpu().numpy()

        sig_p = (pred).std(axis=0)
        sig_g = (y_true).std(axis=0)
        m_p = pred.mean(axis=0)
        m_g = y_true.mean(axis=0)
        ind = (sig_g != 0)
        corr = ((pred - m_p) * (y_true - m_g)).mean(axis=0) / (sig_p * sig_g)
        corr = (corr[ind]).mean()
        corr_final_each.append(corr)

    # print('Valid_Each_final Rse:', rse_final_each)
    # print('Valid_Each_final Rae:', rae_final_each)
    # print('Valid_Each_final Corr:', corr_final_each)

    rse = math.sqrt(total_loss / n_samples) / data.rse
    rae = (total_loss_l1 / n_samples) / data.rae

    # predict = predict.data.cpu().numpy()
    # Ytest = test.data.cpu().numpy()
    predict = forecast_Norm.cpu().numpy()
    Ytest = target_Norm.cpu().numpy()

    sigma_p = (predict).std(axis=0)
    sigma_g = (Ytest).std(axis=0)
    mean_p = predict.mean(axis=0)
    mean_g = Ytest.mean(axis=0)
    index = (sigma_g != 0)
    correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
    correlation = (correlation[index]).mean()
###############################Middle#######################################################


    writer.add_scalar('Validation_final_rse', rse, global_step=epoch)
    writer.add_scalar('Validation_final_rae', rae, global_step=epoch)
    writer.add_scalar('Validation_final_corr', correlation, global_step=epoch)



    print(
        '|valid_final rse {:5.4f} | valid_final rae {:5.4f} | valid_final corr  {:5.4f}'.format(
            rse, rae, correlation), flush=True)


    # if epoch%4==0:
    #     save_model(model, result_file,epoch=epoch)
    return rse, rae, correlation



def testEeco(epoch, data, X, Y, model, evaluateL2, evaluateL1, batch_size, writer):
    model.eval()
    total_loss = 0
    total_loss_l1 = 0

    total_loss_mid = 0
    total_loss_l1_mid = 0
    n_samples = 0
    predict = None
    res_mid = None
    test = None

    forecast_set = []
    Mid_set = []
    target_set = []

    for X, Y in data.get_batches(X, Y, batch_size*10, False):
        # print('0')
        # X = torch.unsqueeze(X,dim=1)
        # X = X.transpose(2,3)
        with torch.no_grad():
            forecast = model(X) #torch.Size([32, 3, 137])
        # forecast = torch.squeeze(forecast)
        # res = torch.squeeze(res)
        true = Y[:, -1, :].squeeze()

        forecast_set.append(forecast)

        target_set.append(Y)

        if len(forecast.shape)==1:
            forecast = forecast.unsqueeze(dim=0)

        if predict is None:
            predict = forecast[:, -1, :].squeeze()

            test = Y[:, -1, :].squeeze()  # torch.Size([32, 3, 137])

        else:
            predict = torch.cat((predict, forecast[:, -1, :].squeeze()))

            test = torch.cat((test, Y[:, -1, :].squeeze()))
        output = forecast[:, -1, :].squeeze()

        scale = data.scale.expand(output.size(0), data.m)
        bias = data.bias.expand(output.size(0), data.m)

        total_loss += evaluateL2(output * scale+bias, true * scale+bias).item()
        total_loss_l1 += evaluateL1(output * scale+bias, true * scale+bias).item()


        n_samples += (output.size(0) * data.m)

    forecast_Norm = torch.cat(forecast_set, axis=0)
    target_Norm = torch.cat(target_set, axis=0)


    rse_final_each = []
    rae_final_each = []
    corr_final_each = []
    Scale = data.scale.expand(forecast_Norm.size(0), data.m)
    bias = data.bias.expand(forecast_Norm.size(0), data.m)
    for i in range(forecast_Norm.shape[1]):
        lossL2_F = evaluateL2(forecast_Norm[:, i, :] * Scale + bias, target_Norm[:, i, :] * Scale+ bias).item()
        lossL1_F = evaluateL1(forecast_Norm[:, i, :] * Scale+ bias, target_Norm[:, i, :] * Scale+ bias).item()

        rse_F = math.sqrt(lossL2_F / forecast_Norm.shape[0] / data.m) / data.rse
        rae_F = (lossL1_F / forecast_Norm.shape[0] / data.m) / data.rae
        rse_final_each.append(rse_F.item())
        rae_final_each.append(rae_F.item())

        pred = forecast_Norm[:, i, :].data.cpu().numpy()
        y_true = target_Norm[:, i, :].data.cpu().numpy()

        sig_p = (pred).std(axis=0)
        sig_g = (y_true).std(axis=0)
        m_p = pred.mean(axis=0)
        m_g = y_true.mean(axis=0)
        ind = (sig_g != 0)
        corr = ((pred - m_p) * (y_true - m_g)).mean(axis=0) / (sig_p * sig_g)
        corr = (corr[ind]).mean()
        corr_final_each.append(corr)

    # print('TEST_Each_final Rse:', rse_final_each)
    # print('TEST_Each_final Rae:', rae_final_each)
    # print('TEST_Each_final Corr:', corr_final_each)


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




#===================

    predict = forecast_Norm.cpu().numpy()
    Ytest = target_Norm.cpu().numpy()

    sigma_p = (predict).std(axis=0)
    sigma_g = (Ytest).std(axis=0)
    mean_p = predict.mean(axis=0)
    mean_g = Ytest.mean(axis=0)
    index = (sigma_g != 0)
    correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
    correlation = (correlation[index]).mean()

    writer.add_scalar('Test_final_rse', rse, global_step=epoch)
    writer.add_scalar('Test_final_rae', rae, global_step=epoch)
    writer.add_scalar('Test_final_corr', correlation, global_step=epoch)

    print(
        '|Test_final rse {:5.4f} | Test_final rae {:5.4f} | Test_final corr   {:5.4f}'.format(
            rse, rae, correlation), flush=True)


    # if epoch%4==0:
    #     save_model(model, result_file,epoch=epoch)
    return rse, rae, correlation


def trainSingleEeco(epoch, data, X, Y, model, criterion, optim, batch_size, writer):
    model.train()
    total_loss = 0
    n_samples = 0
    iter = 0
    for X, Y in data.get_batches(X, Y, batch_size, True):
        model.zero_grad()

        tx = X  # [:, id, :] #torch.Size([32, 1, 137, 168])
        ty = Y[:,-1,:]  # [:, id]


        output = model(tx)
        output = output[:,-1,:] #torch.squeeze(output)
        scale = data.scale.expand(output.size(0), data.m)

        loss = criterion(output * scale, ty * scale)
        loss.backward()
        total_loss += loss.item()
        n_samples += (output.size(0) * data.m)
        grad_norm = optim.step()

        if iter%100==0:
            print('iter:{:3d} | loss: {:.3f}'.format(iter,loss.item()/(output.size(0) * data.m)))
        iter += 1

    writer.add_scalar('TrainSingleEeco loss', total_loss / n_samples, global_step=epoch)
    return total_loss / n_samples

def evaluateSingleEeco(epoch, data, X, Y, model, evaluateL2, evaluateL1, batch_size, writer):
    model.eval()
    total_loss = 0
    total_loss_l1 = 0
    n_samples = 0
    predict = None
    test = None

    for X, Y in data.get_batches(X, Y, batch_size, False):

        with torch.no_grad():
            output = model(X)
        output = output[:,-1,:] #torch.squeeze(output)
        Y = Y[:,-1,:]
        if len(output.shape)==1:
            output = output.unsqueeze(dim=0)
        if predict is None:
            predict = output
            test = Y
        else:
            predict = torch.cat((predict, output))
            test = torch.cat((test, Y))

        scale = data.scale.expand(output.size(0), data.m)
        total_loss += evaluateL2(output * scale, Y * scale).item()
        total_loss_l1 += evaluateL1(output * scale, Y * scale).item()
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

    writer.add_scalar('Validation_final_rse', rse, global_step=epoch)
    writer.add_scalar('Validation_final_rae', rae, global_step=epoch)
    writer.add_scalar('Validation_final_corr', correlation, global_step=epoch)

    print(
        '|valid_final rse {:5.4f} | valid_final rae {:5.4f} | valid_final corr  {:5.4f}'.format(
            rse, rae, correlation), flush=True)

    return rse, rae, correlation

def testSingleEeco(epoch, data, X, Y, model, evaluateL2, evaluateL1, batch_size, writer):
    model.eval()
    total_loss = 0
    total_loss_l1 = 0
    n_samples = 0
    predict = None
    test = None

    for X, Y in data.get_batches(X, Y, batch_size, False):

        with torch.no_grad():
            output = model(X)
        output = output[:, -1, :]  # torch.squeeze(output)
        Y = Y[:, -1, :]
        if len(output.shape)==1:
            output = output.unsqueeze(dim=0)
        if predict is None:
            predict = output
            test = Y
        else:
            predict = torch.cat((predict, output))
            test = torch.cat((test, Y))

        scale = data.scale.expand(output.size(0), data.m)
        total_loss += evaluateL2(output * scale, Y * scale).item()
        total_loss_l1 += evaluateL1(output * scale, Y * scale).item()
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

    writer.add_scalar('Test_final_rse', rse, global_step=epoch)
    writer.add_scalar('Test_final_rae', rae, global_step=epoch)
    writer.add_scalar('Test_final_corr', correlation, global_step=epoch)

    print(
        '|Test_final rse {:5.4f} | Test_final rae {:5.4f} | Test_final corr  {:5.4f}'.format(
            rse, rae, correlation), flush=True)

    return rse, rae, correlation


def trainSingleEecoDeco(epoch, data, X, Y, model, criterion, optim, batch_size):
    model.train()
    total_loss = 0
    n_samples = 0
    iter = 0
    final_loss = 0
    min_loss = 0

    for X, Y in data.get_batches(X, Y, batch_size, True):
        model.zero_grad()             #torch.Size([32, 168, 137])


        tx = X  #torch.Size([64, 24, 8])
        ty = Y  #torch.Size([64, 24, 8])
        ty_last = Y[:,-1,:] #torch.Size([64, 8])

        forecast, res = model(tx)
        # forecast = torch.squeeze(forecast)
        scale_mid = data.scale.expand(forecast.size(0), args.horizon, data.m)
        bias_mid = data.bias.expand(forecast.size(0), args.horizon, data.m)

        scale = data.scale.expand(forecast.size(0), data.m)
        bias = data.bias.expand(forecast.size(0), data.m)
        if args.normalize == 3:
            loss = criterion(forecast[:,-1,:], ty_last) + criterion(res, ty)/res.shape[1]
        else:
            loss = criterion(forecast[:,-1,:] * scale + bias, ty_last * scale + bias) + criterion(res * scale_mid + bias_mid, ty * scale_mid + bias_mid)/res.shape[1]


        loss.backward()
        total_loss += loss.item()

        if args.normalize == 3:
            loss_f = criterion(forecast[:,-1,:], ty_last)
            loss_m = criterion(res, ty)/res.shape[1]
        else:
            loss_f =  criterion(forecast[:,-1,:] * scale + bias, ty_last * scale + bias)
            loss_m = criterion(res * scale_mid + bias_mid, ty * scale_mid + bias_mid)/res.shape[1]



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


def evaluateSingleEecoDeco(epoch, data, X, Y, model, evaluateL2, evaluateL1, batch_size,writer):
    model.eval()
    total_loss = 0
    total_loss_l1 = 0

    total_loss_mid = 0
    total_loss_l1_mid = 0
    n_samples = 0
    predict = None
    res_mid = None
    test = None


    for X, Y in data.get_batches(X, Y, batch_size, False):
        # print('0')
        # X = torch.unsqueeze(X,dim=1)
        # X = X.transpose(2,3)
        Y = Y[:, -1, :]

        with torch.no_grad():
            forecast, res = model(X) #torch.Size([32, 3, 137])

        if len(forecast.shape)==1:
            forecast = forecast.unsqueeze(dim=0)
            res = res.unsqueeze(dim=0)
        if predict is None:
            predict = forecast[:,-1,:].squeeze()
            res_mid = res[:,-1,:].squeeze()
            test = Y
        else:
            predict = torch.cat((predict, forecast[:,-1,:]))
            res_mid = torch.cat((res_mid, res[:,-1,:]))
            test = torch.cat((test, Y))


        scale = data.scale.expand(forecast.size(0),data.m)
        bias = data.bias.expand(forecast.size(0), data.m)


        total_loss += evaluateL2(forecast[:,-1,:] * scale + bias, Y * scale+ bias).item()
        total_loss_l1 += evaluateL1(forecast[:,-1,:] * scale+ bias, Y * scale+ bias).item()
        total_loss_mid += evaluateL2(res[:,-1,:] * scale+ bias, Y * scale+ bias).item()
        total_loss_l1_mid += evaluateL1(res[:,-1,:] * scale+ bias, Y * scale+ bias).item()

        n_samples += (forecast[:,-1,:].size(0) * data.m)

    rse = math.sqrt(total_loss / n_samples) / data.rse
    rae = (total_loss_l1 / n_samples) / data.rae

    rse_mid = math.sqrt(total_loss_mid / n_samples) / data.rse
    rae_mid = (total_loss_l1_mid / n_samples) / data.rae

    predict = predict.data.cpu().numpy()
    Ytest = test.data.cpu().numpy()
    sigma_p = (predict).std(axis=0)
    sigma_g = (Ytest).std(axis=0)
    mean_p = predict.mean(axis=0)
    mean_g = Ytest.mean(axis=0)
    index = (sigma_g != 0)
    correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
    correlation = (correlation[index]).mean()

    mid_pred = res_mid.cpu().numpy()
    sigma_p = (mid_pred).std(axis=0)
    mean_p = mid_pred.mean(axis=0)
    correlation_mid = ((mid_pred - mean_p) * (Ytest - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
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


def testSingleEecoDeco(epoch, data, X, Y, model, evaluateL2, evaluateL1, batch_size, writer):
    model.eval()
    total_loss = 0
    total_loss_l1 = 0

    total_loss_mid = 0
    total_loss_l1_mid = 0
    n_samples = 0
    predict = None
    res_mid = None
    test = None

    for X, Y in data.get_batches(X, Y, batch_size * 200, False):
        # print('0')
        # X = torch.unsqueeze(X,dim=1)
        # X = X.transpose(2,3)
        Y = Y[:, -1, :]
        with torch.no_grad():
            forecast, res = model(X)  # torch.Size([32, 3, 137])

        if len(forecast.shape) == 1:
            forecast = forecast.unsqueeze(dim=0)
            res = res.unsqueeze(dim=0)
        if predict is None:
            predict = forecast[:, -1, :].squeeze()
            res_mid = res[:, -1, :].squeeze()
            test = Y
        else:
            predict = torch.cat((predict, forecast[:,-1,:]))
            res_mid = torch.cat((res_mid, res[:,-1,:]))
            test = torch.cat((test, Y))

        scale = data.scale.expand(forecast.size(0), data.m)
        bias = data.bias.expand(forecast.size(0), data.m)

        total_loss += evaluateL2(forecast[:, -1, :] * scale + bias, Y * scale + bias).item()
        total_loss_l1 += evaluateL1(forecast[:, -1, :] * scale + bias, Y * scale + bias).item()
        total_loss_mid += evaluateL2(res[:, -1, :] * scale + bias, Y * scale + bias).item()
        total_loss_l1_mid += evaluateL1(res[:, -1, :] * scale + bias, Y * scale + bias).item()

        n_samples += (forecast[:, -1, :].size(0) * data.m)



    rse = math.sqrt(total_loss / n_samples) / data.rse
    rae = (total_loss_l1 / n_samples) / data.rae

    rse_mid = math.sqrt(total_loss_mid / n_samples) / data.rse
    rae_mid = (total_loss_l1_mid / n_samples) / data.rae

    predict = predict.data.cpu().numpy()
    Ytest = test.data.cpu().numpy()
    sigma_p = (predict).std(axis=0)
    sigma_g = (Ytest).std(axis=0)
    mean_p = predict.mean(axis=0)
    mean_g = Ytest.mean(axis=0)
    index = (sigma_g != 0)
    correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
    correlation = (correlation[index]).mean()

    mid_pred = res_mid.cpu().numpy()
    sigma_p = (mid_pred).std(axis=0)
    mean_p = mid_pred.mean(axis=0)
    correlation_mid = ((mid_pred - mean_p) * (Ytest - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
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



def main_run():

    Data = DataLoaderH(args.data, 0.6, 0.2, device, args.horizon, args.window_size, args.normalize)

    if args.layers == 2:
        part = [[1, 1], [0, 0], [0, 0]]
    if args.layers == 3:
        part = [[1, 1], [1, 1], [1, 1], [0, 0], [0, 0], [0, 0], [0, 0]]
    if args.layers == 4:
        part = [[1, 1],  [1, 1], [1, 1],  [1, 1], [1, 1], [1, 1], [1, 1], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]

    # part = [[1, 1],  [1, 1], [1, 1],   [1, 1], [1, 1], [1, 1], [1, 1],   [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1],
    #           [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]] #5

    if args.model_mode =="Enco":
        model = IDCNetEcoder(args, num_classes=args.horizon, input_len=args.window_size, input_dim=args.num_nodes,
                       number_levels=len(part),
                       number_level_part=part, num_layers = 3, concat_len=args.num_concat)
        #
        # channel_sizes = [args.nhid] * 1
        # model = TCN(args.num_nodes, args.window_size, args.horizon, channel_sizes, kernel_size=args.kernel,
        #             dropout=args.dropout)

    else:
        model = IDCNet(args, num_classes = args.horizon, input_len=args.window_size, input_dim = args.num_nodes,
                     number_levels=len(part),
                     number_level_part=part, num_layers = 3, concat_len= args.num_concat)
    model = model.to(device)
    print(model)


    # print(args)
    # print('The recpetive field size is', model.receptive_field)
    nParams = sum([p.nelement() for p in model.parameters()])
    print('Number of model parameters is', nParams, flush=True)

    if args.L1Loss:
        criterion = smooth_l1_loss #nn.L1Loss(size_average=False).to(device)  nn.L1Loss().to(args.device)
    #    criterion =  nn.L1Loss().to(args.device)
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

                if args.model_mode =="Enco":
                    if args.single_step:
                        train_loss = trainSingleEeco(epoch, Data, Data.train[0], Data.train[1], model, criterion, optim,
                                               args.batch_size, writer)
                        val_loss, val_rae, val_corr = evaluateSingleEeco(epoch, Data, Data.valid[0], Data.valid[1], model,
                                                                   evaluateL2, evaluateL1,
                                                                   args.batch_size, writer)
                        test_loss, test_rae, test_corr = testSingleEeco(epoch, Data, Data.test[0], Data.test[1], model,
                                                                  evaluateL2,
                                                                  evaluateL1,
                                                                  args.batch_size, writer)
                    else:
                        train_loss = trainEeco(epoch, Data, Data.train[0], Data.train[1], model, criterion, optim,
                                                   args.batch_size)
                        val_loss, val_rae, val_corr = evaluateEeco(epoch, Data, Data.valid[0], Data.valid[1], model,
                                                                       evaluateL2, evaluateL1,
                                                                       args.batch_size, writer)
                        test_loss, test_rae, test_corr = testEeco(epoch, Data, Data.test[0], Data.test[1], model,
                                                                      evaluateL2,
                                                                      evaluateL1,
                                                                      args.batch_size, writer)

                    print(
                        '| EncoOnly: end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.4f} | valid rse {:5.4f} | valid rae {:5.4f} | valid corr  {:5.4f}|'
                        ' test rse {:5.4f} | test rae {:5.4f} | test corr  {:5.4f}'.format(
                            epoch, (time.time() - epoch_start_time), train_loss, val_loss, val_rae, val_corr, test_loss,
                            test_rae, test_corr), flush=True)
                    # Save the model if the validation loss is the best we've seen so far.

                    if val_loss < best_val:
                        # with open(args.save, 'wb') as f:
                        #     torch.save(model, f)
                        best_val = val_loss
                        save_model(model, args.dataset_name,args.save_path,  epoch=epoch)
                        print(
                            '--------------| Best Val loss |--------------')

                else:
                    if args.single_step:
                        train_loss = trainSingleEecoDeco(epoch, Data, Data.train[0], Data.train[1], model, criterion, optim,
                                                   args.batch_size)
                        val_loss, val_rae, val_corr = evaluateSingleEecoDeco(epoch, Data, Data.valid[0], Data.valid[1], model,
                                                                       evaluateL2, evaluateL1,
                                                                       args.batch_size, writer)
                        test_loss, test_rae, test_corr = testSingleEecoDeco(epoch, Data, Data.test[0], Data.test[1], model,
                                                                      evaluateL2,
                                                                      evaluateL1,
                                                                      args.batch_size, writer)
                    else:
                        train_loss = trainEecoDeco(epoch, Data, Data.train[0], Data.train[1], model, criterion, optim, args.batch_size)
                        val_loss, val_rae, val_corr = evaluateEecoDeco(epoch, Data, Data.valid[0], Data.valid[1], model, evaluateL2, evaluateL1,
                                                           args.batch_size, writer)
                        test_loss, test_rae, test_corr = testEecoDeco(epoch, Data, Data.test[0], Data.test[1], model, evaluateL2,
                                                                       evaluateL1,
                                                                       args.batch_size, writer)


                    print(
                        '| EncoDeco: end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.4f} | valid rse {:5.4f} | valid rae {:5.4f} | valid corr  {:5.4f}|'
                        ' test rse {:5.4f} | test rae {:5.4f} | test corr  {:5.4f}'.format(
                            epoch, (time.time() - epoch_start_time), train_loss, val_loss, val_rae, val_corr, test_loss, test_rae, test_corr), flush=True)
                    # Save the model if the validation loss is the best we've seen so far.

                    if val_loss < best_val:
                        # with open(args.save, 'wb') as f:
                        #     torch.save(model, f)
                        save_model(model, args.dataset_name,args.save_path,  epoch=epoch)
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



    vtest_acc, vtest_rae, vtest_corr = evaluateEecoDeco(args.epochs, Data, Data.valid[0], Data.valid[1], model, evaluateL2, evaluateL1,
                                         args.batch_size,writer)
    test_acc, test_rae, test_corr = evaluateEecoDeco(args.epochs, Data, Data.test[0], Data.test[1], model, evaluateL2, evaluateL1,
                                         args.batch_size,writer)
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

