import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        # self.use_cuda = args.cuda
        self.P = args.window_size
        self.m = args.input_dim
        self.hidR = args.hidRNN
        self.hidC = args.hidCNN
        self.hidS = args.hidSkip
        self.Ck = args.CNN_kernel
        self.skip = args.skip
        self.pt = int((self.P - self.Ck) / self.skip)
        self.hw = args.highway_window
        self.conv1 = nn.Conv2d(1, self.hidC, kernel_size=(self.Ck, self.m))
        self.GRU1 = nn.GRU(self.hidC, self.hidR)
        self.dropout = nn.Dropout(p=args.dropout)
        if (self.skip > 0):
            self.GRUskip = nn.GRU(self.hidC, self.hidS)
            self.linear1 = nn.Linear(self.hidR + self.skip * self.hidS, self.m)
        else:
            self.linear1 = nn.Linear(self.hidR, self.m)
        if (self.hw > 0):
            self.highway = nn.Linear(self.hw, 1)
        self.output = None
        if (args.output_fun == 'sigmoid'):
            self.output = F.sigmoid
        if (args.output_fun == 'tanh'):
            self.output = F.tanh

        self.projection = nn.Linear(1, self.P)

    def creatMask(self, x):
        b, l, c = x.shape
        mask_ratio = nn.Dropout(p=0.8)
        Mask = torch.ones(b, l, c, device=x.device)
        Mask = mask_ratio(Mask)
        Mask = Mask > 0  # torch.Size([8, 1, 48, 48])
        Mask = Mask
        x.masked_fill_(Mask, 0)
        return x

    def forward(self, x):
        batch_size = x.size(0)

        x = self.creatMask(x)

        # CNN
        c = x.view(-1, 1, self.P, self.m)
        c = F.relu(self.conv1(c))
        c = self.dropout(c)
        c = torch.squeeze(c, 3)

        # RNN
        r = c.permute(2, 0, 1).contiguous()
        _, r = self.GRU1(r)
        r = self.dropout(torch.squeeze(r, 0))

        # skip-rnn

        if (self.skip > 0):
            s = c[:, :, int(-self.pt * self.skip):].contiguous()
            s = s.view(batch_size, self.hidC, self.pt, self.skip)
            s = s.permute(2, 0, 3, 1).contiguous()
            s = s.view(self.pt, batch_size * self.skip, self.hidC)
            _, s = self.GRUskip(s)
            s = s.view(batch_size, self.skip * self.hidS)
            s = self.dropout(s)
            r = torch.cat((r, s), 1)

        res = self.linear1(r)

        # highway
        if (self.hw > 0):
            z = x[:, -self.hw:, :]
            z = z.permute(0, 2, 1).contiguous().view(-1, self.hw)
            z = self.highway(z)
            z = z.view(-1, self.m)
            res = res + z

        res = res.unsqueeze(2)
        res = self.projection(res)
        res = res.permute(0,2,1).contiguous()
        # if (self.output):
        #     res = self.output(res)
        return res

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
    # parser.add_argument('--data', type=str, required=True,
    #                     help='location of the data file')
    parser.add_argument('--model', type=str, default='LSTNet',
                        help='')
    parser.add_argument('--hidCNN', type=int, default=100, # 32 64 128
                        help='number of CNN hidden units')
    parser.add_argument('--hidRNN', type=int, default=100,  #64 128 256
                        help='number of RNN hidden units')
    parser.add_argument('--window', type=int, default=12,
                        help='window size')
    parser.add_argument('--CNN_kernel', type=int, default=3,
                        help='the kernel size of the CNN layers')
    parser.add_argument('--highway_window', type=int, default=6,
                        help='The window size of the highway component')
    parser.add_argument('--clip', type=float, default=10.,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=100,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='batch size')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--seed', type=int, default=54321,
                        help='random seed')
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--log_interval', type=int, default=2000, metavar='N',
                        help='report interval')
    parser.add_argument('--save', type=str,  default='model/model.pt',
                        help='path to save the final model')
    parser.add_argument('--cuda', type=str, default=True)
    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--horizon', type=int, default=12)
    parser.add_argument('--skip', type=float, default=2)
    parser.add_argument('--hidSkip', type=int, default=5)
    parser.add_argument('--L1Loss', type=bool, default=True)
    parser.add_argument('--normalize', type=int, default=2)
    parser.add_argument('--output_fun', type=str, default='sigmoid')
    parser.add_argument('--input_dim', default=170, type=int)
    parser.add_argument('--window_size', default=12, type=int)
    args = parser.parse_args()


    model = Model(args)
    x = torch.randn(64, 12, 170)
    y= model(x)
    print(y.shape)