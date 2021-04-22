import numpy as np

import torch.nn.functional as F

import numpy as np

from time import time
import os
import math

import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix


from torch import nn
from torch import optim
import torch


from sklearn.metrics import f1_score
import joblib
from torch.utils.data import Dataset, DataLoader
from models.IDCN_Ecoder import IDCNet

import argparse

def prepare_x(data):
    df1 = data[:40, :].T
    return np.array(df1)

def get_label(data):
    lob = data[-5:, :].T
    return lob

def data_classification(X, Y, T):
    [N, D] = X.shape
    df = np.array(X)

    dY = np.array(Y)

    dataY = dY[T - 1:N]

    dataX = np.zeros((N - T + 1, T, D))
    for i in range(T, N + 1):
        dataX[i - T] = df[i - T:i, :]

    return dataX.reshape(dataX.shape + (1,)), dataY








parser = argparse.ArgumentParser()

parser.add_argument('--window_size', type=int, default=96)
parser.add_argument('--horizon', type=int, default=3)

parser.add_argument('--epoch', type=int, default=80)
parser.add_argument('--lr', type=float, default=0.0001)

parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--norm_method', type=str, default='z_score')
parser.add_argument('--optimizer', type=str, default='N') #

parser.add_argument('--exponential_decay_step', type=int, default=5)
parser.add_argument('--decay_rate', type=float, default=0.5)
parser.add_argument('--dropout_rate', type=float, default=0.5)
parser.add_argument('--leakyrelu_rate', type=int, default=0.2)
parser.add_argument('--lradj', type=str, default='type1',help='adjust learning rate')
# Action Part

parser.add_argument('--input_dim', type=int, default=40)
parser.add_argument('--num_stacks', type=int, default=1)
parser.add_argument('--share-weight', default=0, type=int, help='share weight or not in attention q,k,v')
parser.add_argument('--temp', default=0, type=int, help='Use temporature weights or not, if false, temp=1')
parser.add_argument('--hidden-size', default=1, type=int, help='hidden channel of module')
parser.add_argument('--INN', default=1, type=int, help='use INN or basic strategy')
parser.add_argument('--head_size', default=16, type=int, help='hidden channel of module')
parser.add_argument('--kernel', default=5, type=int, help='kernel size')
parser.add_argument('--dilation', default=1, type=int, help='dilation')
parser.add_argument('--positionalEcoding', type=bool, default=True)
args = parser.parse_args()
print(f'Training configs: {args}')


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    lr_adjust = {epoch: args.lr * (0.95 ** ((epoch - 1) // 1))}

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





def checkpoint(epoch, model, optimizer):
    model_out_path = os.path.join(os.getcwd(), r'results', "model_best_adjust.pth")
    states = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(states, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """

    def __init__(self, smoothing=0.3):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()




def get_variable(x):
    x = Variable(x)
    return x.cuda() if torch.cuda.is_available() else x


def train_epoch_action(epoch, train_loader, model, loss_function, optimizer, f1_train_epoch_weighted,
                       f1_train_epoch_macro, result):
    print('train at epoch {}'.format(epoch + 1))

    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    train_pred = np.empty((0))
    train_true = np.empty((0))

    end_time = time()
    adjust_learning_rate(optimizer, epoch, args)

    for i, (seqs, labels) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time() - end_time)

        seqs = get_variable(seqs.float())  # [Batch,1,window,113]
        labels = get_variable(labels.long())  # [B]
        seqs = seqs.squeeze()

        output = model(seqs)
        loss = loss_function(output, labels)
        loss_total = loss



        losses.update(loss_total)

        _, preds = torch.max(output.data, 1)

        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()

        f1_train_weighted = f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='weighted')
        f1_train_macro = f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro')

        batch_time.update(time() - end_time)
        end_time = time()

        train_pred = np.append(train_pred, preds.cpu().numpy(), axis=0)
        train_true = np.append(train_true, labels.cpu().numpy(), axis=0)

        if (i + 1) % 500 == 0:
            print(
                'Each Train_Iter [%d/%d] Loss: %.6f, Loss_avg: %.6f, F1-score_weighted: %.3f, F1-score_macro: %.3f'
                % (i + 1, len(train_loader), loss.item(), losses.avg, f1_train_weighted, f1_train_macro))

    f1_train_epoch_weighted.update(f1_score(train_true, train_pred, average='weighted'))
    f1_train_epoch_macro.update(f1_score(train_true, train_pred, average='macro'))
    macroF1_each = [f1_score(train_true, train_pred, labels=[i], average='macro') for i in range(3)]


    print(
        'Train--> Final: Epoch [%d/%d], Loss: %.6f,  Time: %.3f, F1-score_weighted.avg: %.3f,'
        ' F1-score_macro.avg: %.3f,lr: %.7f '
        % (epoch + 1, 100, losses.avg,
           batch_time.val, f1_train_epoch_weighted.avg, f1_train_epoch_macro.avg, optimizer.param_groups[0]['lr']))
    print(
        'Epoch Each class f1 macro Train_Iter', macroF1_each)

    print(
        '=================================================================')
# In[20]:


def val_epoch_action(epoch, valition_loader, model, optimizer, loss_function, f1_epoch_test_weighted,
                     f1_epoch_test_macro, result):
    print('validation at epoch {}'.format(epoch + 1))

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end_time = time()
    test_pred = np.empty((0))
    test_true = np.empty((0))
    event_acc, event_miF, event_maF, frame_acc, frame_miF, frame_maF = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    predicted_label_segment, lengths_varying_segment, true_label_segment = torch.LongTensor(), torch.LongTensor(), torch.LongTensor()
    with torch.no_grad():
        correct, total = 0, 0
        for i, (seqs, labels) in enumerate(valition_loader):
            # measure data loading time
            data_time.update(time() - end_time)

            seqs = get_variable(seqs.float())  # 64 48 77
            labels = get_variable(labels.long())

            seqs = seqs.squeeze()
            output = model(seqs)
            loss = loss_function(output, labels)
            loss_total = loss


            losses.update(loss_total)

            labels = labels.squeeze()
            _, preds = torch.max(output.data, 1)
            total += labels.size(0)
            duide = (preds == labels).sum()
            # print(duide)
            correct += (preds == labels).sum()

            # losses.update(loss.data, seqs.size(0))

            f1_test_weighted = f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='weighted')
            f1_test_macro = f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro')

            batch_time.update(time() - end_time)
            end_time = time()
            test_pred = np.append(test_pred, preds.cpu().numpy(), axis=0)
            test_true = np.append(test_true, labels.cpu().numpy(), axis=0)



            if (i + 1) % 100 == 0:
                print(
                    'Validation_Iter [%d/%d], F1-weighted-score: %.3f, F1-macro-score: %.3f'
                    % (i + 1, len(valition_loader), f1_test_weighted, f1_test_macro))


        macroF1_each = [f1_score(test_true, test_pred, labels=[i], average='macro') for i in
                        range(3)]
        f1_epoch_test_weighted.update(f1_score(test_true, test_pred, average='weighted'))
        f1_epoch_test_macro.update(f1_score(test_true, test_pred, average='macro'))

        re = np.concatenate((macroF1_each, [f1_epoch_test_weighted.val], [f1_epoch_test_macro.val], [losses.avg],
                             [optimizer.param_groups[0]['lr']]))
        result.append(re)

        acc_test = float(correct) * 100 / total

        print(
            'Test--> Final Epoch [%d/%d], Time: %.3f, Accuracy: %.5f, F1-score_weighted.avg: %.5f, F1-score_macro.avg: %.5f, F1-score_micro.avg: %.5f'
            % (epoch + 1, 100, batch_time.val, acc_test, f1_score(test_true, test_pred, average='weighted'),
               f1_score(test_true, test_pred, average='macro'), f1_score(test_true, test_pred, average='micro')))

        print(
            'Each class f1 macro Test_Iter', macroF1_each)



class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    torch.manual_seed(4321)  # reproducible
    np.random.seed(4321)
    torch.cuda.manual_seed_all(4321)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True  # Can change it to False --> default: False
    torch.backends.cudnn.enabled = True

    Labels = [0, 1, 2]

    pretrain = 0
    result_train = []
    result_test = []
    print("Loading data...")
    # please change the data_path to your local path
    data_path = 'H:/LOBData'

    dec_train = np.loadtxt(data_path + '/Train_Dst_NoAuction_DecPre_CF_7.txt')
    dec_test1 = np.loadtxt(data_path + '/Test_Dst_NoAuction_DecPre_CF_7.txt')
    dec_test2 = np.loadtxt(data_path + '/Test_Dst_NoAuction_DecPre_CF_8.txt')
    dec_test3 = np.loadtxt(data_path + '/Test_Dst_NoAuction_DecPre_CF_9.txt')
    dec_test = np.hstack((dec_test1, dec_test2, dec_test3))

    # extract limit order book data from the FI-2010 dataset
    train_lob = prepare_x(dec_train)
    test_lob = prepare_x(dec_test)

    # extract label from the FI-2010 dataset
    train_label = get_label(dec_train)
    test_label = get_label(dec_test)

    # prepare training data. We feed past 100 observations into our algorithms and choose the prediction horizon.
    trainX_CNN, trainY_CNN = data_classification(train_lob, train_label, T=96)
    trainY_CNN = trainY_CNN[:, 3] - 1  #(254655, 96, 40, 1)
    # trainY_CNN = np_utils.to_categorical(trainY_CNN, 3)

    # prepare test data.
    testX_CNN, testY_CNN = data_classification(test_lob, test_label, T=96)
    testY_CNN = testY_CNN[:, 3] - 1  #(139492, 96, 40, 1)
    # testY_CNN = np_utils.to_categorical(testY_CNN, 3)
    # trainX_CNN = trainX_CNN[100000:150000]
    # trainY_CNN = trainY_CNN[100000:150000]
    # testX_CNN = testX_CNN[80000:105000]
    # testY_CNN = testY_CNN[80000:105000]

    # # np.save(data_path + '/trainX.npy', trainX_CNN)
    # # np.save(data_path + '/trainY.npy', trainY_CNN)
    # # np.save(data_path + '/testX.npy', testX_CNN)
    # # np.save(data_path + '/testY.npy', testY_CNN)
    # trainX_CNN = np.load(data_path + '/trainX.npy')
    # trainY_CNN = np.load(data_path + '/trainY.npy')
    # testX_CNN = np.load(data_path + '/testX.npy')
    # testY_CNN = np.load(data_path + '/testY.npy')

    training_set = [(trainX_CNN[i], trainY_CNN[i]) for i in range(len(trainY_CNN))]
    testing_set = [(testX_CNN[i], testY_CNN[i]) for i in range(len(testY_CNN))]

    # part = [[1, 1], [1, 1], [1, 1], [0, 0], [0, 0], [0, 0], [0, 0]]  # Best model
    part = [[1, 1], [0, 0], [0, 0]]  # Best model


    print('level number {}, level details: {}'.format(len(part), part))
    model = IDCNet(args, num_classes=3, input_len=args.window_size, input_dim=args.input_dim,
                   number_levels=len(part),
                   number_level_part=part, concat_len=None).cuda()

    # print(model)
    model_params = 0
    for parameter in model.parameters():
        model_params += parameter.numel()
    # print('INFO: Trainable parameter count: {:.2f}M'.format(model_params / 1000000.0))
    # for k, v in model.state_dict().items():
    #     print(k, v.shape)
    loss_function = LabelSmoothing(smoothing=0.3)  # nn.CrossEntropyLoss()
    # loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4, amsgrad=True)
    Feature = []
    Feature_label = []



    train_loader = DataLoader(dataset = training_set, batch_size=args.batch_size, shuffle=True)  # , sampler = sampler
    test_loader = DataLoader(dataset = testing_set, batch_size=args.batch_size, shuffle=False)

    for epoch in range(args.epoch):
        # In[11]:
        # Sensor data is segmented using a sliding window mechanism




        f1_train_weighted = AverageMeter()
        f1_train_macro = AverageMeter()
        f1_test_weighted = AverageMeter()
        f1_test_macro = AverageMeter()
        train_epoch_action(epoch, train_loader, model, loss_function, optimizer, f1_train_weighted, f1_train_macro,
                           result_train)
        val_epoch_action(epoch, test_loader, model, optimizer, loss_function, f1_test_weighted, f1_test_macro,
                         result_test)


