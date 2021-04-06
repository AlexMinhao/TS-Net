
import numpy as np

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


import json
from numpy.lib.stride_tricks import as_strided as ast

from torch import nn
from torch import optim
import torch

import pickle as cp
import pywt



import argparse







class Splitting(nn.Module):
    def __init__(self):
        super(Splitting, self).__init__()
        # Deciding the stride base on the direction
        # self.conv_even = lambda x: x[:, ::2, :]
        # self.conv_odd = lambda x: x[:, 1::2, :]

    def even(self, x):
        return x[:, ::2, :]
    def odd(self, x):
        return x[:, 1::2, :]

    # def forward(self, x):
    #     '''Returns the odd and even part'''
    #     return (self.conv_even(x), self.conv_odd(x))

    def forward(self, x):
        '''Returns the odd and even part'''
        return (self.even(x), self.odd(x))






class LiftingScheme(nn.Module):
    def __init__(self, args, in_planes, modified=False, size=[], splitting=True, k_size=4, dropout=0.5, simple_lifting=False):
        super(LiftingScheme, self).__init__()
        self.modified = args.INN

        kernel_size = args.kernel
        dilation = args.dilation
        pad = dilation * (kernel_size - 1) // 2 + 1  # 2 1 0 0
        # pad = k_size // 2
        self.splitting = splitting
        self.split = Splitting()

        # Dynamic build sequential network
        modules_P = []
        modules_U = []
        modules_psi = []
        modules_phi = []
        prev_size = 1

        # HARD CODED Architecture
        if simple_lifting:
            modules_P += [
                nn.ReplicationPad1d(pad),
                nn.Conv2d(in_planes, in_planes,
                          kernel_size=kernel_size, stride=1),
                nn.Dropout(dropout),
                nn.Tanh()
            ]
            modules_U += [
                nn.ReplicationPad1d(pad),
                nn.Conv2d(in_planes, in_planes,
                          kernel_size=kernel_size, stride=1),

                nn.Dropout(dropout),
                nn.Tanh()
            ]
        else:
            size_hidden = args.hidden_size
            modules_P += [
                nn.ReplicationPad1d(pad),
                nn.Conv1d(in_planes * prev_size, int(in_planes * size_hidden),
                          kernel_size=kernel_size, dilation=dilation, stride=1),
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
                nn.Dropout(dropout),
                nn.Conv1d(int(in_planes * size_hidden), in_planes,
                          kernel_size=3, stride=1),
                nn.Tanh()
            ]
            modules_U += [
                nn.ReplicationPad1d(pad),
                nn.Conv1d(in_planes * prev_size, int(in_planes * size_hidden),
                          kernel_size=kernel_size, dilation=dilation, stride=1),
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
                nn.Dropout(dropout),
                nn.Conv1d(int(in_planes * size_hidden), in_planes,
                          kernel_size=3, stride=1),
                nn.Tanh()
            ]
            if self.modified:
                modules_phi += [
                    nn.ReplicationPad1d(pad),
                    nn.Conv1d(in_planes * prev_size, int(in_planes * size_hidden),
                              kernel_size=kernel_size, dilation=dilation, stride=1),
                    nn.LeakyReLU(negative_slope=0.01, inplace=True),
                    nn.Dropout(dropout),
                    nn.Conv1d(int(in_planes * size_hidden), in_planes,
                              kernel_size=3, stride=1),
                    nn.Tanh()
                ]
                modules_psi += [
                    nn.ReplicationPad1d(pad),
                    nn.Conv1d(in_planes * prev_size, int(in_planes * size_hidden),
                              kernel_size=kernel_size, dilation=dilation, stride=1),
                    nn.LeakyReLU(negative_slope=0.01, inplace=True),
                    nn.Dropout(dropout),
                    nn.Conv1d(int(in_planes * size_hidden), in_planes,
                              kernel_size=3, stride=1),
                    nn.Tanh()
                ]
                self.phi = nn.Sequential(*modules_phi)
                self.psi = nn.Sequential(*modules_psi)

        self.P = nn.Sequential(*modules_P)
        self.U = nn.Sequential(*modules_U)

    def forward(self, x):
        if self.splitting:
            # 3  224  112
            # 3  112  112
            (x_even, x_odd) = self.split(x)
        else:
            (x_even, x_odd) = x

        if self.modified:
            x_even = x_even.permute(0, 2, 1)
            x_odd = x_odd.permute(0, 2, 1)
            d = x_odd.mul(torch.exp(self.phi(x_even))) - self.P(x_even)
            c = x_even.mul(torch.exp(self.psi(d))) + self.U(d)

            return (c, d)

        else:

            x_even = x_even.permute(0, 2, 1)
            x_odd = x_odd.permute(0, 2, 1)

            d = x_odd - self.P(x_even)
            c = x_even + self.U(d)

            return (c, d)


class LiftingSchemeLevel(nn.Module):
    def __init__(self, args, in_planes, share_weights, modified=False, size=[2, 1], kernel_size=4, simple_lifting=False):
        super(LiftingSchemeLevel, self).__init__()
        self.level = LiftingScheme(args,
            in_planes=in_planes, modified=modified,
            size=size, k_size=kernel_size, simple_lifting=simple_lifting)

    def forward(self, x):
        '''Returns (LL, LH, HL, HH)'''
        (L, H) = self.level(x)  # 10 3 224 224

        return (L, H)


class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes, disable_conv):
        super(BottleneckBlock, self).__init__()
        self.bn1 = nn.BatchNorm1d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        # self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)

        self.disable_conv = disable_conv  # in_planes == out_planes
        if not self.disable_conv:
            self.conv1 = nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=1,
                                   padding=0, bias=False)

    def forward(self, x):
        if self.disable_conv:
            return self.relu(self.bn1(x))
        else:
            return self.conv1(self.relu(self.bn1(x)))


class LevelWASN(nn.Module):
    def __init__(self, args, in_planes, lifting_size, kernel_size, no_bottleneck,
                 share_weights, simple_lifting, regu_details, regu_approx):
        super(LevelWASN, self).__init__()
        self.regu_details = regu_details
        self.regu_approx = regu_approx
        if self.regu_approx + self.regu_details > 0.0:
            self.loss_details = nn.SmoothL1Loss()

        self.wavelet = LiftingSchemeLevel(args, in_planes, share_weights,
                                          size=lifting_size, kernel_size=kernel_size,
                                          simple_lifting=simple_lifting)
        self.share_weights = share_weights
        if no_bottleneck:
            # We still want to do a BN and RELU, but we will not perform a conv
            # as the input_plane and output_plare are the same
            self.bootleneck = BottleneckBlock(in_planes, in_planes, disable_conv=True)
        else:
            self.bootleneck = BottleneckBlock(in_planes, in_planes, disable_conv=False)

    def forward(self, x):
        (L, H) = self.wavelet(x)  # 10 9 128
        approx = L
        details = H
        r = None
        if (self.regu_approx + self.regu_details != 0.0):  # regu_details=0.01, regu_approx=0.01

            if self.regu_details:
                rd = self.regu_details * \
                     details.abs().mean()

            # Constrain on the approximation
            if self.regu_approx:
                rc = self.regu_approx * torch.dist(approx.mean(), x.mean(), p=2)

            if self.regu_approx == 0.0:
                # Only the details
                r = rd
            elif self.regu_details == 0.0:
                # Only the approximation
                r = rc
            else:
                # Both
                r = rd + rc

        if self.bootleneck:
            return self.bootleneck(approx).permute(0, 2, 1), r, details
        else:
            return approx.permute(0, 2, 1), r, details



class EncoderTree(nn.Module):
    def __init__(self, level_layers,  level_parts, norm_layer=None):
        super(EncoderTree, self).__init__()
        self.level_layers = nn.ModuleList(level_layers)
        self.conv_layers = None #nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer
        # self.level_part = [[1, 1], [0, 0], [0, 0]]
        self.level_part = level_parts #[[0, 1], [0, 0]]

        self.count_levels = 0
    def forward(self, x, attn_mask=None):
        # x [B, L, D] torch.Size([16, 336, 512])
        rs = []  # List of constrains on details and mean
        det = []  # List of averaged pooled details

        input = [x, ]
        for l in self.level_layers:
            low, r, details = l(input[0])

            if self.level_part[self.count_levels][0]:
                input.append(low)
            else:
                low = low.permute(0, 2, 1)
                det += [low]  ##############################################################################
            if self.level_part[self.count_levels][1]:
                details = details.permute(0, 2, 1)
                input.append(details)
            else:
                det += [details]  ##############################################################################
            del input[0]
            rs += [r]
            self.count_levels = self.count_levels + 1

        for aprox in input:
            aprox = aprox.permute(0, 2, 1)  # b 77 1
            # aprox = self.avgpool(aprox) ##############################################################################
            det += [aprox]

        self.count_levels = 0
        # We add them inside the all GAP detail coefficients
        x = torch.cat(det, 2)  # [b, 77, 8]
        x = x.permute(0, 2, 1)
        if self.norm is not None:
            x = self.norm(x)  #torch.Size([16, 512, 336])

        return x



class WASN(nn.Module):
    def __init__(self, args, num_classes, num_stacks = 3 , first_conv=9,
                 number_levels=4, number_level_part=[[1, 0], [1, 0], [1, 0]],
                 no_bootleneck=True):
        super(WASN, self).__init__()




        # First convolution


        in_planes = first_conv
        out_planes = first_conv * (number_levels + 1)
        self.pe = args.positionalEcoding

        self.blocks1 = EncoderTree(
            [
            LevelWASN(args = args, in_planes=in_planes,
                      lifting_size=[2, 1], kernel_size=4, no_bottleneck=True,
                      share_weights=False, simple_lifting=False, regu_details=0.01, regu_approx=0.01)

            for l in range(number_levels)
            ],


            level_parts = number_level_part
        )

        self.blocks2 = EncoderTree(
            [
                LevelWASN(args=args, in_planes=in_planes,
                          lifting_size=[2, 1], kernel_size=4, no_bottleneck=True,
                          share_weights=False, simple_lifting=False, regu_details=0.01, regu_approx=0.01)

                for l in range(number_levels)
            ],


            level_parts= number_level_part
        )

        if no_bootleneck:
            in_planes *= 1

        self.num_planes = out_planes



        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                # nn.init.xavier_uniform_(m.weight.data)
                # if m.bias is not None:
                m.bias.data.zero_()


        self.projection1 = nn.Conv1d(args.window_size, num_classes,
                                     kernel_size=1, stride=1, bias=False)

        self.projection2 = nn.Conv1d(args.window_size+num_classes, num_classes,
                                     kernel_size=1, stride=1, bias=False)


        
        self.hidden_size = in_planes 
        # For positional encoding
        if self.hidden_size%2 == 1:
           self.hidden_size += 1

        num_timescales = self.hidden_size // 2  # 词维度除以2,因为词维度一半要求sin,一半要求cos
        max_timescale = 10000.0
        min_timescale = 1.0
        # min_timescale: 将应用于每个位置的最小尺度
        # max_timescale: 在每个位置应用的最大尺度
        log_timescale_increment = (
                math.log(float(max_timescale) / float(min_timescale)) /
                max(num_timescales - 1, 1))  # 因子log(max/min) / (256-1)
        temp = torch.arange(num_timescales, dtype=torch.float32)
        inv_timescales = min_timescale * torch.exp(
            torch.arange(num_timescales, dtype=torch.float32) *
            -log_timescale_increment)  # 将log(max/min)均分num_timescales份数(词维度一半)
        self.register_buffer('inv_timescales', inv_timescales)
    def get_position_encoding(self, x):
        max_length = x.size()[1]
        position = torch.arange(max_length, dtype=torch.float32,
                                device=x.device) #tensor([0., 1., 2., 3., 4.], device='cuda:0')
        temp1 = position.unsqueeze(1) #5 1
        temp2 = self.inv_timescales.unsqueeze(0) #1 256
        scaled_time = position.unsqueeze(1) * self.inv_timescales.unsqueeze(0) #5 256
        signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)],
                           dim=1) #5 512 [T, C]
        signal = F.pad(signal, (0, 0, 0, self.hidden_size % 2))
        signal = signal.view(1, max_length, self.hidden_size)



        # signal = F.pad(signal, (1, self.hidden_size % 2), "constant", 0)
        # if self.hidden_size % 2==1:
        #     signal = signal[:,1:]
        # signal = signal.view(1, max_length, self.hidden_size)
       
        return signal

    def forward(self, x):
        if self.pe:
            pe = self.get_position_encoding(x)
            if pe.shape[2]>x.shape[2]:
                x += pe[:,:,:-1]
            else:
                x += self.get_position_encoding(x)
        res1 = x

        x = self.blocks1(x, attn_mask=None)

        x += res1

        x = self.projection1(x)
        MidOutPut = x

        x = torch.cat((res1, x), dim=1)
        res2 = x
        x = self.blocks2(x, attn_mask=None)
        x += res2
        x = self.projection2(x)
        return x, MidOutPut









def get_variable(x):
    x = Variable(x)
    return x.cuda() if torch.cuda.is_available() else x




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=bool, default=True)
    parser.add_argument('--evaluate', type=bool, default=False)
    parser.add_argument('--dataset', type=str, default='PeMS03_data')  # PeMS07
    parser.add_argument('--window_size', type=int, default=12)
    parser.add_argument('--horizon', type=int, default=12)
    parser.add_argument('--train_length', type=float, default=7)
    parser.add_argument('--valid_length', type=float, default=2)
    parser.add_argument('--test_length', type=float, default=1)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--lr', type=float, default=3 * 1e-5)
    parser.add_argument('--multi_layer', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--validate_freq', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--norm_method', type=str, default='z_score')
    parser.add_argument('--optimizer', type=str, default='RMSProp')
    parser.add_argument('--early_stop', type=bool, default=False)
    parser.add_argument('--exponential_decay_step', type=int, default=5)
    parser.add_argument('--decay_rate', type=float, default=0.5)
    parser.add_argument('--dropout_rate', type=float, default=0.5)
    parser.add_argument('--leakyrelu_rate', type=int, default=0.2)

    # Action Part

    parser.add_argument('--share-weight', default=0, type=int, help='share weight or not in attention q,k,v')
    parser.add_argument('--temp', default=0, type=int, help='Use temporature weights or not, if false, temp=1')
    parser.add_argument('--hidden-size', default=1, type=int, help='hidden channel of module')
    parser.add_argument('--INN', default=1, type=int, help='use INN or basic strategy')
    parser.add_argument('--kernel', default=3, type=int, help='kernel size')
    parser.add_argument('--dilation', default=1, type=int, help='dilation')
    parser.add_argument('--positionalEcoding', type=bool, default=True)

    args = parser.parse_args()
    # part = [[1, 1], [1, 1], [1, 1], [0, 0], [0, 0], [0, 0], [0, 0]]  # Best model
    part = [[1, 1], [0, 0], [0, 0]]
    # part = [ [0, 0]]

    print('level number {}, level details: {}'.format(len(part), part))
    model = WASN(args, num_classes=12, first_conv=307,
                 number_levels=len(part),
                 number_level_part=part).cuda()
    x = torch.randn(32, 12, 307).cuda()
    y,res = model(x)
    print(y.shape)