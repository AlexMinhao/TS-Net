from ETTH_util.data.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from ETTH_util.exp.exp_basic import Exp_Basic
# from models.model import Informer, InformerStack

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

from models.IDCN import IDCNet
from models.IDCN_Ecoder import IDCNetEcoder

class Exp_Informer(Exp_Basic):
    def __init__(self, args):
        super(Exp_Informer, self).__init__(args)
    
    def _build_model(self):
        # model_dict = {
        #     'informer':Informer,
        #     'informerstack':InformerStack,
        # }
        if self.args.model=='informer' or self.args.model=='informerstack':
            print('No Informer')
            # e_layers = self.args.e_layers if self.args.model=='informer' else self.args.s_layers
            # model = model_dict[self.args.model](
            #     self.args.enc_in,
            #     self.args.dec_in,
            #     self.args.c_out,
            #     self.args.seq_len,
            #     self.args.label_len,
            #     self.args.pred_len,
            #     self.args.factor,
            #     self.args.d_model,
            #     self.args.n_heads,
            #     e_layers, # self.args.e_layers,
            #     self.args.d_layers,
            #     self.args.d_ff,
            #     self.args.dropout,
            #     self.args.attn,
            #     self.args.embed,
            #     self.args.freq,
            #     self.args.activation,
            #     self.args.output_attention,
            #     self.args.distil,
            #     self.args.mix,
            #     self.device
            # ).float()
        else:
            if self.args.layers == 2:
                part = [[1, 1], [0, 0], [0, 0]]
            if self.args.layers == 3:
                part = [[1, 1], [1, 1], [1, 1], [0, 0], [0, 0], [0, 0], [0, 0]]
            if self.args.layers == 4:
                part = [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0],
                        [0, 0], [0, 0], [0, 0]]
            if self.args.layers == 5:
                part = [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1],
                        [1, 1], [1, 1], [1, 1],[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]

            if self.args.stacks==2:
                model = IDCNet(self.args, num_classes=self.args.pred_len, input_len=self.args.seq_len, input_dim=7,
                               number_levels=len(part),
                               number_level_part=part, num_layers=self.args.layers, concat_len=None)
            elif self.args.stacks==1:
                model = IDCNetEcoder(self.args, num_classes=self.args.pred_len, input_len=self.args.seq_len, input_dim=7,
                               number_levels=len(part),
                               number_level_part=part, num_layers = self.args.layers, concat_len=None)
            else:
                print('Error!')
        print(model)
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model.double()

    def _get_data(self, flag):
        args = self.args

        data_dict = {
            'ETTh1':Dataset_ETT_hour,
            'ETTh2':Dataset_ETT_hour,
            'ETTm1':Dataset_ETT_minute,
            'ETTm2':Dataset_ETT_minute,
            'WTH':Dataset_Custom,
            'ECL':Dataset_Custom,
            'Solar':Dataset_Custom,
            'custom':Dataset_Custom,
        }
        Data = data_dict[self.args.data]
        timeenc = 0 if args.embed!='timeF' else 1

        if flag == 'test':
            shuffle_flag = False; drop_last = True; batch_size = args.batch_size; freq=args.freq
        elif flag=='pred':
            shuffle_flag = False; drop_last = False; batch_size = 1; freq=args.detail_freq
            Data = Dataset_Pred
        else:
            shuffle_flag = True; drop_last = True; batch_size = args.batch_size; freq=args.freq
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            timeenc=timeenc,
            freq=freq,
            cols=args.cols
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
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    
    def _select_criterion(self, losstype):
        if losstype == "mse":
            criterion = nn.MSELoss()
        elif losstype == "mae":
            criterion = nn.L1Loss()
        else:
            criterion = nn.L1Loss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []

        preds = []
        trues = []
        mids = []
        pred_scales = []
        true_scales = []
        mid_scales = []

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            pred, pred_scale, mid, mid_scale, true, true_scale = self._process_one_batch_IDCN(
                vali_data, batch_x, batch_y)

            if self.args.stacks == 1:
                loss = criterion(pred.detach().cpu(), true.detach().cpu())

                preds.append(pred.detach().cpu().numpy())
                trues.append(true.detach().cpu().numpy())
                pred_scales.append(pred_scale.detach().cpu().numpy())
                true_scales.append(true_scale.detach().cpu().numpy())

            elif self.args.stacks == 2:
                loss = criterion(pred.detach().cpu(), true.detach().cpu()) + criterion(mid.detach().cpu(), true.detach().cpu())

                preds.append(pred.detach().cpu().numpy())
                trues.append(true.detach().cpu().numpy())
                mids.append(mid.detach().cpu().numpy())
                pred_scales.append(pred_scale.detach().cpu().numpy())
                mid_scales.append(mid_scale.detach().cpu().numpy())
                true_scales.append(true_scale.detach().cpu().numpy())

            else:
                print('Error!')


            total_loss.append(loss)
        total_loss = np.average(total_loss)

        if self.args.stacks == 1:
            preds = np.array(preds)
            trues = np.array(trues)
            pred_scales = np.array(pred_scales)
            true_scales = np.array(true_scales)

            preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
            trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
            true_scales = true_scales.reshape(-1, true_scales.shape[-2], true_scales.shape[-1])
            pred_scales = pred_scales.reshape(-1, pred_scales.shape[-2], pred_scales.shape[-1])

            mae, mse, rmse, mape, mspe, corr = metric(preds, trues)
            maes, mses, rmses, mapes, mspes, corrs = metric(pred_scales, true_scales)
            print('normed mse:{:.4f}, mae:{:.4f}, rmse:{:.4f}, mape:{:.4f}, mspe:{:.4f}, corr:{:.4f}'.format(mse, mae, rmse, mape, mspe, corr))
            print('denormed mse:{:.4f}, mae:{:.4f}, rmse:{:.4f}, mape:{:.4f}, mspe:{:.4f}, corr:{:.4f}'.format(mses, maes, rmses, mapes, mspes, corrs))
        elif self.args.stacks == 2:
            preds = np.array(preds)
            trues = np.array(trues)
            mids = np.array(mids)
            pred_scales = np.array(pred_scales)
            true_scales = np.array(true_scales)
            mid_scales = np.array(mid_scales)

            preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
            trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
            mids = mids.reshape(-1, mids.shape[-2], mids.shape[-1])
            true_scales = true_scales.reshape(-1, true_scales.shape[-2], true_scales.shape[-1])
            pred_scales = pred_scales.reshape(-1, pred_scales.shape[-2], pred_scales.shape[-1])
            mid_scales = mid_scales.reshape(-1, mid_scales.shape[-2], mid_scales.shape[-1])
            # print('test shape:', preds.shape, mids.shape, trues.shape)

            mae, mse, rmse, mape, mspe, corr = metric(mids, trues)
            maes, mses, rmses, mapes, mspes, corrs = metric(mid_scales, true_scales)
            print('mid --> normed mse:{:.4f}, mae:{:.4f}, rmse:{:.4f}, mape:{:.4f}, mspe:{:.4f}, corr:{:.4f}'.format(mse, mae, rmse, mape, mspe, corr))
            print('mid --> denormed mse:{:.4f}, mae:{:.4f}, rmse:{:.4f}, mape:{:.4f}, mspe:{:.4f}, corr:{:.4f}'.format(mses, maes, rmses, mapes, mspes, corrs))

            mae, mse, rmse, mape, mspe, corr = metric(preds, trues)
            maes, mses, rmses, mapes, mspes, corrs = metric(pred_scales, true_scales)
            print('final --> normed mse:{:.4f}, mae:{:.4f}, rmse:{:.4f}, mape:{:.4f}, mspe:{:.4f}, corr:{:.4f}'.format(mse, mae, rmse, mape, mspe, corr))
            print('final --> denormed mse:{:.4f}, mae:{:.4f}, rmse:{:.4f}, mape:{:.4f}, mspe:{:.4f}, corr:{:.4f}'.format(mses, maes, rmses, mapes, mspes, corrs))
        else:
            print('Error!')

        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag = 'train')
        vali_data, vali_loader = self._get_data(flag = 'val')
        test_data, test_loader = self._get_data(flag = 'test')

        writer = SummaryWriter('./run_ETTh/{}'.format(self.args.model))

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        model_optim = self._select_optimizer()
        criterion =  self._select_criterion(self.args.loss)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                
                model_optim.zero_grad()
                pred, pred_scale, mid, mid_scale, true, true_scale = self._process_one_batch_IDCN(
                    train_data, batch_x, batch_y)

                if self.args.stacks == 1:
                    loss = criterion(pred, true)
                elif self.args.stacks == 2:
                    loss = criterion(pred, true) + criterion(mid, true)
                else:
                    print('Error!')

                train_loss.append(loss.item())
                
                if (i+1) % 100==0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                
                if self.args.use_amp:
                    print('use amp')
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
            train_loss = np.average(train_loss)
            print('--------start to validate-----------')
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            print('--------start to test-----------')
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
        mids = []
        pred_scales = []
        true_scales = []
        mid_scales = []
        
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(test_loader):
            pred, pred_scale, mid, mid_scale, true, true_scale = self._process_one_batch_IDCN(
                test_data, batch_x, batch_y)

            if self.args.stacks == 1:
                preds.append(pred.detach().cpu().numpy())
                trues.append(true.detach().cpu().numpy())
                pred_scales.append(pred_scale.detach().cpu().numpy())
                true_scales.append(true_scale.detach().cpu().numpy())
            elif self.args.stacks == 2:
                preds.append(pred.detach().cpu().numpy())
                trues.append(true.detach().cpu().numpy())
                mids.append(mid.detach().cpu().numpy())
                pred_scales.append(pred_scale.detach().cpu().numpy())
                mid_scales.append(mid_scale.detach().cpu().numpy())
                true_scales.append(true_scale.detach().cpu().numpy())

            else:
                print('Error!')

        if self.args.stacks == 1:
            preds = np.array(preds)
            trues = np.array(trues)

            pred_scales = np.array(pred_scales)
            true_scales = np.array(true_scales)

            preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
            trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
            true_scales = true_scales.reshape(-1, true_scales.shape[-2], true_scales.shape[-1])
            pred_scales = pred_scales.reshape(-1, pred_scales.shape[-2], pred_scales.shape[-1])

            mae, mse, rmse, mape, mspe, corr = metric(preds, trues)
            maes, mses, rmses, mapes, mspes, corrs = metric(pred_scales, true_scales)
            print('normed mse:{:.4f}, mae:{:.4f}, rmse:{:.4f}, mape:{:.4f}, mspe:{:.4f}, corr:{:.4f}'.format(mse, mae, rmse, mape, mspe, corr))
            print('TTTT denormed mse:{:.4f}, mae:{:.4f}, rmse:{:.4f}, mape:{:.4f}, mspe:{:.4f}, corr:{:.4f}'.format(mses, maes, rmses, mapes, mspes, corrs))

            # result save
            # folder_path = './results/' + setting + '/'
            # if not os.path.exists(folder_path):
            #     os.makedirs(folder_path)

            # mae, mse, rmse, mape, mspe, corr = metric(preds, trues)
            # print('Test:mse:{:.4f}, mae:{:.4f}, rmse:{:.4f}, mape:{:.4f}, mspe:{:.4f}, corr:{:.4f}'.format(mse, mae, rmse, mape, mspe, corr))

            # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
            # np.save(folder_path + 'pred.npy', preds)
            # np.save(folder_path + 'true.npy', trues)

        elif self.args.stacks == 2:
            preds = np.array(preds)
            trues = np.array(trues)
            mids = np.array(mids)

            pred_scales = np.array(pred_scales)
            true_scales = np.array(true_scales)
            mid_scales = np.array(mid_scales)

            preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
            trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
            mids = mids.reshape(-1, mids.shape[-2], mids.shape[-1])
            true_scales = true_scales.reshape(-1, true_scales.shape[-2], true_scales.shape[-1])
            pred_scales = pred_scales.reshape(-1, pred_scales.shape[-2], pred_scales.shape[-1])
            mid_scales = mid_scales.reshape(-1, mid_scales.shape[-2], mid_scales.shape[-1])
            # print('test shape:', preds.shape, mids.shape, trues.shape)

            mae, mse, rmse, mape, mspe, corr = metric(mids, trues)
            maes, mses, rmses, mapes, mspes, corrs = metric(mid_scales, true_scales)
            print('Mid --> normed mse:{:.4f}, mae:{:.4f}, rmse:{:.4f}, mape:{:.4f}, mspe:{:.4f}, corr:{:.4f}'.format(mse, mae, rmse, mape, mspe, corr))

            mae, mse, rmse, mape, mspe, corr = metric(preds, trues)
            maes, mses, rmses, mapes, mspes, corrs = metric(pred_scales, true_scales)
            print('TTTT Final --> denormed mse:{:.4f}, mae:{:.4f}, rmse:{:.4f}, mape:{:.4f}, mspe:{:.4f}, corr:{:.4f}'.format(mse, mae, rmse, mape, mspe, corr))


            # result save
            # folder_path = './results/' + setting + '/'
            # if not os.path.exists(folder_path):
            #     os.makedirs(folder_path)

            # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
            # np.save(folder_path + 'pred.npy', preds)
            # np.save(folder_path + 'true.npy', trues)
        else:
            print('Error!')
        return mae, maes, mse, mses

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')
        
        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path+'/'+'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        self.model.eval()
        
        preds = []
        
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(pred_loader):
            pred, true = self._process_one_batch(
                pred_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            preds.append(pred.detach().cpu().numpy())

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        
        # result save
        # folder_path = './results/' + setting +'/'
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)
        
        # np.save(folder_path+'real_prediction.npy', preds)
        
        return

    def _process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float()

        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        # decoder input
        if self.args.padding==0:
            dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        elif self.args.padding==1:
            dec_inp = torch.ones([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        dec_inp = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).float().to(self.device)
        # encoder - decoder
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:
            if self.args.output_attention:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        if self.args.inverse:
            outputs = dataset_object.inverse_transform(outputs)
        f_dim = -1 if self.args.features=='MS' else 0
        batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)
        print('xxxx',outputs.shape,batch_y.shape,batch_x.shape,batch_y.shape)
        return outputs, batch_y

    def _process_one_batch_IDCN(self, dataset_object, batch_x, batch_y):
        batch_x = batch_x.double().to(self.device)
        batch_y = batch_y.double()


        # decoder input
        # if self.args.padding==0:
        #     dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        # elif self.args.padding==1:
        #     dec_inp = torch.ones([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        # dec_inp = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).float().to(self.device)
        # # encoder - decoder
        # if self.args.use_amp:
        #     with torch.cuda.amp.autocast():
        #         if self.args.output_attention:
        #             outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
        #         else:
        #             outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        # else:
        #     if self.args.output_attention:
        #         outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
        #     else:
        #         outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        if self.args.stacks == 1:
            outputs = self.model(batch_x)
        elif self.args.stacks == 2:
            outputs, mid = self.model(batch_x)
        else:
            print('Error!')

#        if self.args.inverse:
        outputs_scaled = dataset_object.inverse_transform(outputs)
        if self.args.stacks == 2:
            mid_scaled = dataset_object.inverse_transform(mid)
        f_dim = -1 if self.args.features=='MS' else 0
        batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)
        batch_y_scaled = dataset_object.inverse_transform(batch_y)
    #    print('yyy',outputs.shape,batch_y.shape,batch_x.shape)
        if self.args.stacks == 1:
            return outputs, outputs_scaled, 0,0, batch_y, batch_y_scaled
        elif self.args.stacks == 2:
            return outputs, outputs_scaled, mid,mid_scaled, batch_y, batch_y_scaled
        else:
            print('Error!')
