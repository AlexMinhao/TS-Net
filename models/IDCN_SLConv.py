
import math
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn
import torch
from torch.nn.utils import weight_norm
import argparse
import numpy as np

class sliding_window(nn.Module):
    def __init__(self, window, stride):
        super(sliding_window, self).__init__()

        self.window_size = window
        self.stride = stride

        # （n-f+2p）/s+1
    def forward(self, x):
        length = x.shape[1]
        partition = int((length-self.window_size)/self.stride + 1)
        start = 0
        end = 0
        x_new = []
        for _ in range(length):
            if end < length:
                end = start + self.window_size
                x_new.append(x[:,start:end,:])
                start = start + self.stride

        return x_new









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

    def forward(self, x):
        '''Returns the odd and even part'''
        return (self.even(x), self.odd(x))


class Interactor(nn.Module):
    def __init__(self, args, in_planes, splitting=True, dropout=0.5,
                 simple_lifting=False):
        super(Interactor, self).__init__()
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
            # x_odd_update = x_odd.mul(torch.exp(self.phi(x_even))) - self.P(x_even)
            # x_even_update = x_even.mul(torch.exp(self.psi(x_odd_update))) + self.U(x_odd_update)
            d = x_odd.mul(torch.exp(self.phi(x_even)))
            c = x_even.mul(torch.exp(self.psi(x_odd)))
            x_even_update = c + self.U(d)
            x_odd_update = d - self.P(c)

            return (x_even_update, x_odd_update)

        else:

            x_even = x_even.permute(0, 2, 1)
            x_odd = x_odd.permute(0, 2, 1)

            d = x_odd - self.P(x_even)
            c = x_even + self.U(d)

            return (c, d)


class InteractorLevel(nn.Module):
    def __init__(self, args, in_planes,
                 simple_lifting=False):
        super(InteractorLevel, self).__init__()
        self.level = Interactor(args,
                                   in_planes=in_planes,
                                   simple_lifting=simple_lifting)

    def forward(self, x):
        '''Returns (LL, LH, HL, HH)'''
        # (L, H)
        (x_even_update, x_odd_update) = self.level(x)  # 10 3 224 224

        return (x_even_update, x_odd_update)


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


class LevelIDCN(nn.Module):
    def __init__(self, args, in_planes, lifting_size, kernel_size, no_bottleneck,
                 share_weights, simple_lifting, regu_details, regu_approx):
        super(LevelIDCN, self).__init__()
        self.regu_details = regu_details
        self.regu_approx = regu_approx
        if self.regu_approx + self.regu_details > 0.0:
            self.loss_details = nn.SmoothL1Loss()

        self.interact = InteractorLevel(args, in_planes,
                                          simple_lifting=simple_lifting)
        self.share_weights = share_weights
        if no_bottleneck:
            # We still want to do a BN and RELU, but we will not perform a conv
            # as the input_plane and output_plare are the same
            self.bootleneck = BottleneckBlock(in_planes, in_planes, disable_conv=True)
        else:
            self.bootleneck = BottleneckBlock(in_planes, in_planes, disable_conv=False)

    def forward(self, x):
        (x_even_update, x_odd_update) = self.interact(x)  # 10 9 128


        if self.bootleneck:
            return self.bootleneck(x_even_update).permute(0, 2, 1), x_odd_update
        else:
            return x_even_update.permute(0, 2, 1),x_odd_update


class EncoderTree(nn.Module):
    def __init__(self, level_layers, level_parts, num_layers, Encoder=True, norm_layer=None):
        super(EncoderTree, self).__init__()
        self.level_layers = nn.ModuleList(level_layers)
        self.conv_layers = None  # nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer
        # self.level_part = [[1, 1], [0, 0], [0, 0]]
        self.level_part = level_parts  # [[0, 1], [0, 0]]
        self.layers = num_layers
        self.count_levels = 0
        self.ecoder = Encoder

    def reOrder(self, num_of_length, layer=2):
        N = num_of_length
        n = list(range(1, N + 1, 1))
        remain = [i % 2 for i in n]

        n_1 = []
        for i in range(N):
            if remain[i] > 0:
                n_1.append((n[i] + 1) / 2 + N / 2)
            else:
                n_1.append(n[i] / 2)

        remain = [i % 2 for i in n_1]

        n_2 = []
        rem4 = [i % 4 for i in n]

        for i in range(N):
            if rem4[i] == 0:
                n_2.append(int(n[i] / 4))

            elif rem4[i] == 1:

                n_2.append(int((3 * N + 3) / 4 + n[i] / 4))
            elif rem4[i] == 2:
                n_2.append(int((1 * N + 2) / 4 + n[i] / 4))
            elif rem4[i] == 3:
                n_2.append(int((2 * N + 1) / 4 + n[i] / 4))
            else:
                print("Error!")

        n_3 = []
        rem8 = [i % 8 for i in n]
        for i in range(N):
            if rem8[i] == 0:
                n_3.append(int(n[i] / 8))
            elif rem8[i] == 1:
                n_3.append(int(n[i] / 8 + (7 * N + 7) / 8))
            elif rem8[i] == 2:
                n_3.append(int(n[i] / 8 + (3 * N + 6) / 8))
            elif rem8[i] == 3:
                n_3.append(int(n[i] / 8 + (5 * N + 5) / 8))
            elif rem8[i] == 4:
                n_3.append(int(n[i] / 8 + (1 * N + 4) / 8))
            elif rem8[i] == 5:
                n_3.append(int(n[i] / 8 + (6 * N + 3) / 8))
            elif rem8[i] == 6:
                n_3.append(int(n[i] / 8 + (2 * N + 2) / 8))
            elif rem8[i] == 7:
                n_3.append(int(n[i] / 8 + (4 * N + 1) / 8))

            else:
                print("Error!")

        n_4 = []
        rem16 = [i % 16 for i in n]
        for i in range(N):
            if rem16[i] == 0:
                n_4.append(int(n[i] / 16))
            elif rem16[i] == 1:
                n_4.append(int(n[i] / 16 + (15 * N + 15) / 16))
            elif rem16[i] == 2:
                n_4.append(int(n[i] / 16 + (7 * N + 14) / 16))
            elif rem16[i] == 3:
                n_4.append(int(n[i] / 16 + (11 * N + 13) / 16))
            elif rem16[i] == 4:
                n_4.append(int(n[i] / 16 + (3 * N + 12) / 16))
            elif rem16[i] == 5:
                n_4.append(int(n[i] / 16 + (13 * N + 11) / 16))
            elif rem16[i] == 6:
                n_4.append(int(n[i] / 16 + (5 * N + 10) / 16))
            elif rem16[i] == 7:
                n_4.append(int(n[i] / 16 + (9 * N + 9) / 16))
            elif rem16[i] == 8:
                n_4.append(int(n[i] / 16 + (1 * N + 8) / 16))
            elif rem16[i] == 9:
                n_4.append(int(n[i] / 16 + (14 * N + 7) / 16))
            elif rem16[i] == 10:
                n_4.append(int(n[i] / 16 + (6 * N + 6) / 16))
            elif rem16[i] == 11:
                n_4.append(int(n[i] / 16 + (10 * N + 5) / 16))
            elif rem16[i] == 12:
                n_4.append(int(n[i] / 16 + (2 * N + 4) / 16))
            elif rem16[i] == 13:
                n_4.append(int(n[i] / 16 + (12 * N + 3) / 16))
            elif rem16[i] == 14:
                n_4.append(int(n[i] / 16 + (4 * N + 2) / 16))
            elif rem16[i] == 15:
                n_4.append(int(n[i] / 16 + (8 * N + 1) / 16))

            else:
                print("Error!")

            n_5 = []
            rem32 = [i % 32 for i in n]
            for i in range(N):
                if rem32[i] == 0:
                    n_5.append(int(n[i] / 32))
                elif rem32[i] == 1:
                    n_5.append(int(n[i] / 32 + (31 * N + 31) / 32))
                elif rem32[i] == 2:
                    n_5.append(int(n[i] / 32 + (15 * N + 30) / 16))
                elif rem32[i] == 3:
                    n_5.append(int(n[i] / 32 + (23 * N + 29) / 16))
                elif rem32[i] == 4:
                    n_5.append(int(n[i] / 32 + (7 * N + 28) / 32))
                elif rem32[i] == 5:
                    n_5.append(int(n[i] / 32 + (27 * N + 27) / 32))
                elif rem32[i] == 6:
                    n_5.append(int(n[i] / 32 + (11 * N + 26) / 32))
                elif rem32[i] == 7:
                    n_5.append(int(n[i] / 32 + (19 * N + 25) / 32))
                elif rem32[i] == 8:
                    n_5.append(int(n[i] / 32 + (3 * N + 24) / 32))
                elif rem32[i] == 9:
                    n_5.append(int(n[i] / 32 + (29 * N + 23) / 32))
                elif rem32[i] == 10:
                    n_5.append(int(n[i] / 32 + (13 * N + 22) / 32))
                elif rem32[i] == 11:
                    n_5.append(int(n[i] / 32 + (21 * N + 21) / 32))
                elif rem32[i] == 12:
                    n_5.append(int(n[i] / 32 + (5 * N + 20) / 32))
                elif rem32[i] == 13:
                    n_5.append(int(n[i] / 32 + (25 * N + 19) / 32))
                elif rem32[i] == 14:
                    n_5.append(int(n[i] / 32 + (9 * N + 18) / 16))
                elif rem32[i] == 15:
                    n_5.append(int(n[i] / 32 + (17 * N + 17) / 32))
                elif rem32[i] == 16:
                    n_5.append(int(n[i] / 32 + (1 * N + 16) / 32))
                elif rem32[i] == 17:
                    n_5.append(int(n[i] / 32 + (30 * N + 15) / 32))
                elif rem32[i] == 18:
                    n_5.append(int(n[i] / 32 + (14 * N + 14) / 32))
                elif rem32[i] == 19:
                    n_5.append(int(n[i] / 32 + (22 * N + 13) / 32))
                elif rem32[i] == 20:
                    n_5.append(int(n[i] / 32 + (6 * N + 12) / 32))
                elif rem32[i] == 21:
                    n_5.append(int(n[i] / 32 + (26 * N + 11) / 32))
                elif rem32[i] == 22:
                    n_5.append(int(n[i] / 32 + (10 * N + 10) / 32))
                elif rem32[i] == 23:
                    n_5.append(int(n[i] / 32 + (18 * N + 9) / 16))
                elif rem32[i] == 24:
                    n_5.append(int(n[i] / 32 + (2 * N + 8) / 32))
                elif rem32[i] == 25:
                    n_5.append(int(n[i] / 32 + (28 * N + 7) / 32))
                elif rem32[i] == 26:
                    n_5.append(int(n[i] / 32 + (12 * N + 6) / 32))
                elif rem32[i] == 27:
                    n_5.append(int(n[i] / 32 + (20 * N + 5) / 32))
                elif rem32[i] == 28:
                    n_5.append(int(n[i] / 32 + (4 * N + 4) / 32))
                elif rem32[i] == 29:
                    n_5.append(int(n[i] / 32 + (24 * N + 3) / 32))
                elif rem32[i] == 30:
                    n_5.append(int(n[i] / 32 + (8 * N + 2) / 32))
                elif rem32[i] == 31:
                    n_5.append(int(n[i] / 32 + (16 * N + 1) / 32))
                else:
                    print("Error!")


        if layer == 2:
            return [i - 1 for i in n_2]
        if layer == 3:
            return [i - 1 for i in n_3]
        if layer == 4:
            return [i - 1 for i in n_4]
        if layer == 5:
            return [i - 1 for i in n_5]

    def forward(self, x, attn_mask=None):

        # x [B, L, D] torch.Size([16, 336, 512])

        det = []  # List of averaged pooled details
        input = [x, ]
        for l in self.level_layers:
            x_even_update, x_odd_update = l(input[0])

            if self.level_part[self.count_levels][0]:
                input.append(x_even_update)
            else:
                x_even_update = x_even_update.permute(0, 2, 1)
                det += [x_even_update]  ##############################################################################
            if self.level_part[self.count_levels][1]:
                x_odd_update = x_odd_update.permute(0, 2, 1)
                input.append(x_odd_update)
            else:
                det += [x_odd_update]  ##############################################################################
            del input[0]

            self.count_levels = self.count_levels + 1

        for aprox in input:
            aprox = aprox.permute(0, 2, 1)  # b 77 1
            # aprox = self.avgpool(aprox) ##############################################################################
            det += [aprox]

        self.count_levels = 0
        # We add them inside the all GAP detail coefficients

        x = torch.cat(det, 2)  # torch.Size([32, 307, 12])
        index = self.reOrder(x.shape[2], layer=self.layers)
        x_reorder = [x[:, :, i].unsqueeze(2) for i in index]

        x_reorder = torch.cat(x_reorder, 2)

        x = x_reorder.permute(0, 2, 1)
        # x = x.permute(0, 2, 1)
        if self.norm is not None:
            x = self.norm(x)  # torch.Size([16, 512, 336])

        return x


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class IDCNet(nn.Module):
    def __init__(self, args, num_classes, input_len, input_dim=9,
                 number_levels=4, number_level_part=[[1, 0], [1, 0], [1, 0]], num_layers = 3,
                 concat_len = None, no_bootleneck=True):
        super(IDCNet, self).__init__()

        # First convolution

        # self.first_conv = True
        # self.conv_first = nn.Sequential(
        #     weight_norm(nn.Conv1d(input_dim, int(args.hidden_size * input_dim),
        #                           kernel_size=2, stride=1, padding=1, bias=False)),
        #     # nn.BatchNorm1d(extend_channel),
        #     Chomp1d(1),
        #     nn.LeakyReLU(negative_slope=0.01, inplace=True),
        #     nn.Dropout(0.5),
        #     # weight_norm(nn.Conv1d(args.hidden_size * first_conv, first_conv,
        #     #           kernel_size=2, stride=1, padding=1, bias=False)),
        #     #
        #     # nn.LeakyReLU(negative_slope=0.01, inplace=True),
        #     # nn.Dropout(0.5),
        # )
        # self.conv_Second = nn.Sequential(
        #     weight_norm(nn.Conv1d(input_dim, int(args.hidden_size * input_dim),
        #                           kernel_size=2, stride=1, padding=1, bias=False)),
        #     # nn.BatchNorm1d(extend_channel),
        #     Chomp1d(1),
        #     nn.LeakyReLU(negative_slope=0.01, inplace=True),
        #     nn.Dropout(0.5),
        #     # weight_norm(nn.Conv1d(args.hidden_size * first_conv, first_conv,
        #     #           kernel_size=2, stride=1, padding=1, bias=False)),
        #     #
        #     # nn.LeakyReLU(negative_slope=0.01, inplace=True),
        #     # nn.Dropout(0.5),
        # )

        in_planes = input_dim
        out_planes = input_dim * (number_levels + 1)
        self.pe = args.positionalEcoding
        self.horizon = num_classes

        self.blocks1 = EncoderTree(
            [
                LevelIDCN(args=args, in_planes=in_planes,
                          lifting_size=[2, 1], kernel_size=4, no_bottleneck=True,
                          share_weights=False, simple_lifting=False, regu_details=0.01, regu_approx=0.01)

                for l in range(number_levels)
            ],

            level_parts=number_level_part,
            num_layers=num_layers,
            Encoder=True
        )

        self.blocks2 = EncoderTree(
            [
                LevelIDCN(args=args, in_planes=in_planes,
                          lifting_size=[2, 1], kernel_size=4, no_bottleneck=True,
                          share_weights=False, simple_lifting=False, regu_details=0.01, regu_approx=0.01)

                for l in range(number_levels)
            ],

            level_parts=number_level_part,
            num_layers=num_layers,
            Encoder=False
        )

        self.concat_len = concat_len

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

        self.projection1 = nn.Conv1d(input_len, num_classes,
                                     kernel_size=1, stride=1, bias=False)

        if self.concat_len:
            self.projection2 = nn.Conv1d(concat_len + num_classes, num_classes,
                                     kernel_size=1, stride=1, bias=False)
        else:
            self.projection2 = nn.Conv1d(input_len + num_classes, num_classes,
                                         kernel_size=1, stride=1, bias=False)

        self.projection3 = nn.Conv1d(input_dim, input_dim,
                                     kernel_size=1, stride=1, bias=False)

        self.projection4 = nn.Conv1d(input_dim, input_dim,
                                     kernel_size=1, stride=1, bias=False)

        self.projection5 = nn.Conv1d(input_dim, input_dim,
                                     kernel_size=7, stride=1, bias=False, padding=4)

        self.projection6 = nn.Conv1d(input_dim, input_dim,
                                     kernel_size=7, stride=1, bias=False, padding=4)

        self.projection7 = nn.Conv2d(1, 1,
                                     kernel_size=(5,1), stride=1, bias=False, padding=(2,0))

        self.projection8 = nn.Conv2d(1, 1,
                                     kernel_size=(5, 1), stride=1, bias=False, padding=(2, 0))


        self.sliding = sliding_window(window = 24, stride = 12)

        conv_block1 = [
                nn.Conv1d(24, 12,
                          kernel_size=1, stride=1, bias=False)
                for i in range(13)
            ]
        self.projection9 = nn.ModuleList(conv_block1)
        self.projection99 = nn.Conv1d(12*13, num_classes,
                                     kernel_size=1, stride=1, bias=False)
        conv_block2 = [
            nn.Conv1d(24, 12,
                      kernel_size=1, stride=1, bias=False)
            for i in range(13)
        ]
        self.projection10 = nn.ModuleList(conv_block2)
        self.projection1010 = nn.Conv1d(12*13, num_classes,
                                      kernel_size=1, stride=1, bias=False)

        self.projection11 = nn.Conv1d(2*num_classes, num_classes,
                                     kernel_size=1, stride=1, bias=False)
        self.projection12 = nn.Conv1d(2*num_classes, num_classes,
                                     kernel_size=1, stride=1, bias=False)


        self.hidden_size = in_planes
        # For positional encoding
        if self.hidden_size % 2 == 1:
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
                                device=x.device)  # tensor([0., 1., 2., 3., 4.], device='cuda:0')
        temp1 = position.unsqueeze(1)  # 5 1
        temp2 = self.inv_timescales.unsqueeze(0)  # 1 256
        scaled_time = position.unsqueeze(1) * self.inv_timescales.unsqueeze(0)  # 5 256
        signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)],
                           dim=1)  # 5 512 [T, C]
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
            if pe.shape[2] > x.shape[2]:
                x += pe[:, :, :-1]
            else:
                x += self.get_position_encoding(x)

#1  ##############baseline######################
        # res1 = x
        # x = self.blocks1(x, attn_mask=None)
        # x += res1
        # x = self.projection1(x)
        # MidOutPut = x
        # if self.concat_len:
        #     x = torch.cat((res1[:, -self.concat_len:,:], x), dim=1)
        # else:
        #     x = torch.cat((res1, x), dim=1)
        # res2 = x
        # x = self.blocks2(x, attn_mask=None)
        # x += res2
        # x = self.projection2(x)

#2  ##############channel direction######################

        # res1 = x
        # x = self.blocks1(x, attn_mask=None)
        # x += res1
        # x = x.permute(0,2,1)
        # x = self.projection3(x)
        # x = x.permute(0, 2, 1)
        # MidOutPut = x[:, -self.horizon:, :]
        #
        # if self.concat_len:
        #     x = torch.cat((res1[:, -self.concat_len:, :], MidOutPut), dim=1)
        # else:
        #     x = torch.cat((res1, x), dim=1)
        # res2 = x
        # x = self.blocks2(x, attn_mask=None)
        # x += res2
        # x = x.permute(0, 2, 1)
        # x = self.projection4(x)
        # x = x.permute(0, 2, 1)
        # x = x[:, -self.horizon:, :]

#3  ##############channel direction kernal = 3######################

        # res1 = x
        # x = self.blocks1(x, attn_mask=None)
        # x += res1
        # x = x.permute(0, 2, 1)
        # x = self.projection5(x)
        # x = x.permute(0, 2, 1)
        # MidOutPut = x[:, -self.horizon:, :]
        # if self.concat_len:
        #     x = torch.cat((res1[:, -self.concat_len:, :], MidOutPut), dim=1)
        # else:
        #     x = torch.cat((res1, x), dim=1)
        # res2 = x
        # x = self.blocks2(x, attn_mask=None)
        # x += res2
        # x = x.permute(0, 2, 1)
        # x = self.projection6(x)
        # x = x.permute(0, 2, 1)
        # x = x[:, -self.horizon:, :]

#4  ##############channel direction Conv2d k = 3######################

        res1 = x
        x = self.blocks1(x, attn_mask=None)
        x += res1
        x = x.unsqueeze(1)
        x = self.projection7(x)
        x = x.squeeze()
        MidOutPut = x[:, -self.horizon:, :]
        if self.concat_len:
            x = torch.cat((res1[:, -self.concat_len:, :], MidOutPut), dim=1)
        else:
            x = torch.cat((res1, x), dim=1)
        res2 = x
        x = self.blocks2(x, attn_mask=None)
        x += res2
        x = x.unsqueeze(1)
        x = self.projection8(x)
        x = x.squeeze()
        x = x[:, -self.horizon:, :]

#5  ############## split ######################

        # res1 = x
        # x = self.blocks1(x, attn_mask=None)
        # x += res1
        #
        # x_split = self.sliding(x)
        # x_new = []
        # for i, x_ in enumerate(x_split):
        #     x_ = self.projection9[i](x_)
        #     x_new.append(x_)
        # x_new = torch.cat(x_new,dim=1)
        # x = self.projection99(x_new)
        #
        # MidOutPut = x
        # if self.concat_len:
        #     x = torch.cat((res1[:, -self.concat_len:,:], x), dim=1)
        # else:
        #     x = torch.cat((res1, x), dim=1)
        # res2 = x
        # x = self.blocks2(x, attn_mask=None)
        # x += res2
        #
        #
        # x_split2 = self.sliding(x)
        # x_new = []
        # for i, x2_ in enumerate(x_split2):
        #     x2_ = self.projection10[i](x2_)
        #     x_new.append(x2_)
        # x_new = torch.cat(x_new, dim=1)
        # x = self.projection1010(x_new)



#6  ############## split concat linear ######################

        # res1 = x
        # x = self.blocks1(x, attn_mask=None)
        # x += res1
        #
        # x_split = self.sliding(x)
        # x_new = []
        # for i, x_ in enumerate(x_split):
        #     x_ = self.projection9[i](x_)
        #     x_new.append(x_)
        # x_new = torch.cat(x_new, dim=1)
        #
        # x_normal = self.projection1(x)
        # x_split_result = self.projection99(x_new)
        # x = torch.cat([x_normal,x_split_result], dim=1)
        # x = self.projection11(x)
        #
        # MidOutPut = x
        #
        # if self.concat_len:
        #     x = torch.cat((res1[:, -self.concat_len:,:], x), dim=1)
        # else:
        #     x = torch.cat((res1, x), dim=1)
        # res2 = x
        # x = self.blocks2(x, attn_mask=None)
        # x += res2
        # #
        # #
        # x_split = self.sliding(x)
        # x_new = []
        # for i, x_ in enumerate(x_split):
        #     x_ = self.projection10[i](x_)
        #     x_new.append(x_)
        # x_new = torch.cat(x_new, dim=1)
        #
        # x_normal2 = self.projection2(x)
        # x_split_result2 = self.projection1010(x_new)
        # x = torch.cat([x_normal2, x_split_result2], dim=1)
        # x = self.projection12(x)
        #
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
    part = [[1, 1], [1, 1], [1, 1], [0, 0], [0, 0], [0, 0], [0, 0]]  # Best model
    # part = [[1, 1], [0, 0], [0, 0]]
    # part = [ [0, 0]]

    print('level number {}, level details: {}'.format(len(part), part))
    model = IDCNet(args, num_classes=24, input_len= 168, input_dim=7,
                 number_levels=len(part),
                 number_level_part=part, concat_len = 144).cuda()
    x = torch.randn(32, 168, 7).cuda()
    y,res = model(x)
    print(y.shape)