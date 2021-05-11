from ETTH_util.data.data_loader import Dataset_ETT_hour, Dataset_ETT_minute
from ETTH_util.exp.exp_basic import Exp_Basic

from models.IDCN import IDCNet
# from models.TCN import TCN

from ETTH_util.utils.tools import EarlyStopping, adjust_learning_rate
from ETTH_util.utils.metrics import metric

from tensorboardX import SummaryWriter
import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import os
import time

import warnings
warnings.filterwarnings('ignore')

class Exp_Informer(Exp_Basic):
    def __init__(self, args):
        super(Exp_Informer, self).__init__(args)
    
    def _build_model(self):
        # model_dict = {
        #     'informer':Informer,
        # }
        if self.args.model=='informer':
            a = 0

        else:
            if self.args.layers == 2:
                part = [[1, 1], [0, 0], [0, 0]]
            if self.args.layers == 3:
                part = [[1, 1], [1, 1], [1, 1], [0, 0], [0, 0], [0, 0], [0, 0]]
            if self.args.layers == 4:
                part = [[1, 1],  [1, 1], [1, 1],  [1, 1], [1, 1], [1, 1], [1, 1], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]

            model = IDCNet(self.args, num_classes=self.args.pred_len, input_len=self.args.seq_len, input_dim=7,
                           number_levels=len(part),number_level_part=part, num_layers = self.args.layers, concat_len=self.args.num_concat)

            # channel_sizes = [self.args.nhid] * self.args.levels
            # model = TCN(7, self.args.seq_len, channel_sizes, kernel_size=self.args.kernel, dropout=self.args.dropout)
        
        return model.double()

    def _get_data(self, flag):
        args = self.args

        data_dict = {
            'ETTh1':Dataset_ETT_hour,
            'ETTh2':Dataset_ETT_hour,
            'ETTm1':Dataset_ETT_minute,
        }
        Data = data_dict[self.args.data]

        if flag == 'test':
            shuffle_flag = False; drop_last = True; batch_size = args.batchSize
        else:
            shuffle_flag = True; drop_last = True; batch_size = args.batchSize
        
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.lr)
        return model_optim
    
    def _select_criterion(self):
        criterion =  nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        preds = []
        trues = []
        mids = []
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(vali_loader):
            batch_x = batch_x.double().to(self.device)
            batch_y = batch_y.double()

            outputs, mid = self.model(batch_x) ###################################################################################
            batch_y = batch_y[:,-self.args.pred_len:,:].to(self.device)

            pred = outputs.detach().cpu()
            mid_pred = mid.detach().cpu()
            true = batch_y.detach().cpu()

            preds.append(pred.numpy())
            trues.append(true.numpy())
            mids.append(mid_pred.numpy())


            loss = criterion(pred, true) + criterion(mid_pred, true)

            total_loss.append(loss)
        total_loss = np.average(total_loss)

        preds = np.array(preds)
        mids = np.array(mids)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        mids = mids.reshape(-1, mids.shape[-2], mids.shape[-1])
        print('test shape:', preds.shape, trues.shape)


        mae, mse, rmse, mape, mspe, corr = metric(mids, trues)
        print('Mid: mse:{}, mae:{}, rmse:{}, mape:{}, corr:{}'.format(mse, mae, rmse, mape, corr))
        mae, mse, rmse, mape, mspe, corr = metric(preds, trues)
        print('Final: mse:{}, mae:{}, rmse:{}, mape:{}, corr:{}'.format(mse, mae, rmse, mape, corr))

        self.model.train()
        return total_loss
        
    def train(self, setting):
        train_data, train_loader = self._get_data(flag = 'train')
        vali_data, vali_loader = self._get_data(flag = 'val')
        test_data, test_loader = self._get_data(flag = 'test')

        writer = SummaryWriter('./run_ETTh/{}'.format(self.args.model))


        path = './checkpoints/'+setting
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)######################################
        
        model_optim = self._select_optimizer()
        criterion =  self._select_criterion()

        for epoch in range(self.args.epochs):
            iter_count = 0
            train_loss = []
            
            self.model.train()
            for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(train_loader):

                iter_count += 1
                
                model_optim.zero_grad()
                
                batch_x = batch_x.double().to(self.device)
                batch_y = batch_y.double()  #                     torch.Size([32, 96, 7])

                outputs, mid = self.model(batch_x)
                batch_y = batch_y[:,-self.args.pred_len:,:].to(self.device)

                loss = criterion(outputs, batch_y) + criterion(mid, batch_y)
                train_loss.append(loss.item())
                
                if (i+1) % 100==0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed*((self.args.epochs - epoch)*train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                
                loss.backward()
                model_optim.step()

            train_loss = np.average(train_loss)
            print("Epoch: {0}, Steps: {1} | Valiation Results =====>")
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            print("Epoch: {0}, Steps: {1} | Test Results =====>")
            test_loss = self.vali(test_data, test_loader, criterion)


            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))

            writer.add_scalar('train_loss', train_loss, global_step=epoch)
            writer.add_scalar('vali_loss', vali_loss, global_step=epoch)
            writer.add_scalar('test_loss', test_loss, global_step=epoch)

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch+1, self.args)
            
        best_model_path = path+'/'+'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        
        return self.model

    def test(self, setting):
        test_data, test_loader = self._get_data(flag='test')
        
        self.model.eval()
        
        preds = []
        trues = []
        
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(test_loader):
            batch_x = batch_x.double().to(self.device)
            batch_y = batch_y.double()


            outputs, res = self.model(batch_x)
            batch_y = batch_y[:,-self.args.pred_len:,:].to(self.device)
            
            pred = outputs.detach().cpu().numpy()
            true = batch_y.detach().cpu().numpy()
            
            preds.append(pred)
            trues.append(true)

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, corr = metric(preds, trues)
        print('mse:{}, mae:{}, rmse:{}, mape:{}, corr:{}'.format(mse, mae, rmse, mape, corr))

        np.save(folder_path+'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path+'pred.npy', preds)
        np.save(folder_path+'true.npy', trues)

        return
