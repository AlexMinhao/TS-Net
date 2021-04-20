import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse



def initialize_weight(x):
    nn.init.xavier_uniform_(x.weight)
    if x.bias is not None:
        nn.init.constant_(x.bias, 0)


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, filter_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, filter_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.layer2 = nn.Linear(filter_size, hidden_size)

        initialize_weight(self.layer1)
        initialize_weight(self.layer2)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, dropout_rate, head_size=8):
        super(MultiHeadAttention, self).__init__()

        self.head_size = head_size

        self.att_size = att_size = hidden_size // head_size
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, head_size * att_size, bias=False)
        self.linear_k = nn.Linear(hidden_size, head_size * att_size, bias=False)
        self.linear_v = nn.Linear(hidden_size, head_size * att_size, bias=False)
        initialize_weight(self.linear_q)
        initialize_weight(self.linear_k)
        initialize_weight(self.linear_v)

        self.att_dropout = nn.Dropout(dropout_rate)

        self.output_layer = nn.Linear(head_size * att_size, hidden_size,
                                      bias=False)
        initialize_weight(self.output_layer)

    def forward(self, q, k, v, mask, cache=None):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(batch_size, -1, self.head_size, d_k)
        if cache is not None and 'encdec_k' in cache:
            k, v = cache['encdec_k'], cache['encdec_v']
        else:
            k = self.linear_k(k).view(batch_size, -1, self.head_size, d_k)
            v = self.linear_v(v).view(batch_size, -1, self.head_size, d_v)

            if cache is not None:
                cache['encdec_k'], cache['encdec_v'] = k, v

        q = q.transpose(1, 2)                  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)                  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q.mul_(self.scale)
        x = torch.matmul(q, k) # [b, h, q_len, k_len]
        if mask is not None:
            x.masked_fill_(mask.unsqueeze(1), -1e9)
        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.head_size * d_v)

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, filter_size, dropout_rate, head_size):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.self_attention = MultiHeadAttention(hidden_size, dropout_rate, head_size= head_size)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.ffn = FeedForwardNetwork(hidden_size, filter_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, mask = None):  # pylint: disable=arguments-differ
        y = self.self_attention_norm(x)
        y = self.self_attention(y, y, y, mask)  #torch.Size([10, 5, 512])
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x


class DecoderLayer(nn.Module):
    def __init__(self, hidden_size, filter_size, dropout_rate):
        super(DecoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.self_attention = MultiHeadAttention(hidden_size, dropout_rate)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.enc_dec_attention_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.enc_dec_attention = MultiHeadAttention(hidden_size, dropout_rate)
        self.enc_dec_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.ffn = FeedForwardNetwork(hidden_size, filter_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, enc_output, self_mask, i_mask, cache):
        y = self.self_attention_norm(x)
        y = self.self_attention(y, y, y, self_mask)
        y = self.self_attention_dropout(y)
        x = x + y

        if enc_output is not None:
            y = self.enc_dec_attention_norm(x)
            y = self.enc_dec_attention(y, enc_output, enc_output, i_mask,
                                       cache)
            y = self.enc_dec_attention_dropout(y)
            x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x


class Encoder(nn.Module):
    def __init__(self, hidden_size, filter_size, dropout_rate, n_layers, head_size):
        super(Encoder, self).__init__()

        encoders = [EncoderLayer(hidden_size, filter_size, dropout_rate, head_size=head_size)
                    for _ in range(n_layers)]
        self.layers = nn.ModuleList(encoders)

        self.last_norm = nn.LayerNorm(hidden_size, eps=1e-6)

    def forward(self, inputs, mask):
        encoder_output = inputs
        for enc_layer in self.layers:
            encoder_output = enc_layer(encoder_output, mask)
        return self.last_norm(encoder_output)


class Decoder(nn.Module):
    def __init__(self, hidden_size, filter_size, dropout_rate, n_layers):
        super(Decoder, self).__init__()

        decoders = [DecoderLayer(hidden_size, filter_size, dropout_rate)
                    for _ in range(n_layers)]
        self.layers = nn.ModuleList(decoders)

        self.last_norm = nn.LayerNorm(hidden_size, eps=1e-6)

    def forward(self, targets, enc_output, i_mask, t_self_mask, cache):
        decoder_output = targets
        for i, dec_layer in enumerate(self.layers):
            layer_cache = None
            if cache is not None:
                if i not in cache:
                    cache[i] = {}
                layer_cache = cache[i]
            decoder_output = dec_layer(decoder_output, enc_output,
                                       t_self_mask, i_mask, layer_cache)
        return self.last_norm(decoder_output)


class Transformer(nn.Module):
    def __init__(self,
                 n_layers=3,
                 hidden_size=256,
                 filter_size=512,
                 dropout_rate=0.1,
                 head_size=8,
                 has_inputs=True,
                 src_pad_idx=None,
                 trg_pad_idx=None):
        super(Transformer, self).__init__()

        self.hidden_size = hidden_size
        self.emb_scale = hidden_size ** 0.5
        self.has_inputs = has_inputs
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

        # self.t_vocab_embedding = nn.Embedding(t_vocab_size, hidden_size)   #####################################
        # nn.init.normal_(self.t_vocab_embedding.weight, mean=0,
        #                 std=hidden_size**-0.5)
        self.t_emb_dropout = nn.Dropout(dropout_rate)
        self.decoder = Decoder(hidden_size, filter_size,
                               dropout_rate, n_layers)

        if has_inputs:
            # if not share_target_embedding:
            #     self.i_vocab_embedding = nn.Embedding(i_vocab_size, #####################################
            #                                           hidden_size)
            #     nn.init.normal_(self.i_vocab_embedding.weight, mean=0,
            #                     std=hidden_size**-0.5)
            # else:
            #     self.i_vocab_embedding = self.t_vocab_embedding
            #
            # self.i_emb_dropout = nn.Dropout(dropout_rate)

            self.encoder = Encoder(hidden_size, filter_size,
                                   dropout_rate, n_layers, head_size= head_size)

        # For positional encoding
        num_timescales = self.hidden_size // 2 #词维度除以2,因为词维度一半要求sin,一半要求cos
        max_timescale = 10000.0
        min_timescale = 1.0
        # min_timescale: 将应用于每个位置的最小尺度
        # max_timescale: 在每个位置应用的最大尺度
        log_timescale_increment = (
            math.log(float(max_timescale) / float(min_timescale)) /
            max(num_timescales - 1, 1)) #因子log(max/min) / (256-1)
        temp = torch.arange(num_timescales, dtype=torch.float32)
        inv_timescales = min_timescale * torch.exp(
            torch.arange(num_timescales, dtype=torch.float32) *
            -log_timescale_increment) #将log(max/min)均分num_timescales份数(词维度一半)
        self.register_buffer('inv_timescales', inv_timescales)
#       向模块添加持久缓冲区
    def forward(self, inputs):
        enc_output, i_mask = None, None
        if self.has_inputs:
            i_mask = 0#utils.create_pad_mask(inputs, [2,3,4])
            enc_output = self.encode(inputs, i_mask) # torch.Size([10, 5, 512])

        # t_mask = 0#utils.create_pad_mask(targets, self.trg_pad_idx)
        # target_size = targets.size()[1]
        # t_self_mask = utils.create_trg_self_mask(target_size,
        #                                          device=targets.device) #
        # tensor([[[0, 1, 1, 1, 1],
        #          [0, 0, 1, 1, 1],
        #          [0, 0, 0, 1, 1],
        #          [0, 0, 0, 0, 1],
        #          [0, 0, 0, 0, 0]]], device='cuda:0', dtype=torch.uint8)

        return enc_output #self.decode(targets, enc_output, i_mask, t_self_mask, t_mask)

    def encode(self, inputs, i_mask):
        # Input embedding
        # input_embedded = self.i_vocab_embedding(inputs)
        # input_embedded.masked_fill_(i_mask.squeeze(1).unsqueeze(-1), 0)
        # input_embedded *= self.emb_scale
        inputs += self.get_position_encoding(inputs)
        # input_embedded = self.i_emb_dropout(input_embedded)
        i_mask = None
        return self.encoder(inputs, i_mask)

    def decode(self, targets, enc_output, i_mask, t_self_mask, t_mask,
               cache=None):
        # # target embedding
        # target_embedded = self.t_vocab_embedding(targets)
        # # target_embedded.masked_fill_(t_mask.squeeze(1).unsqueeze(-1), 0)
        #
        # # Shifting
        # target_embedded = target_embedded[:, :-1]
        # target_embedded = F.pad(target_embedded, (0, 0, 1, 0))
        #
        # target_embedded *= self.emb_scale
        targets += self.get_position_encoding(targets)
        # target_embedded = self.t_emb_dropout(target_embedded)

        # decoder
        decoder_output = self.decoder(targets, enc_output, i_mask,
                                      t_self_mask, cache)
        # linear
        output = torch.matmul(decoder_output,
                              self.t_vocab_embedding.weight.transpose(0, 1))
  # torch.Size([10, 5, 100])
        return output

    def get_position_encoding(self, x):
        max_length = x.size()[1]
        position = torch.arange(max_length, dtype=torch.float32,
                                device=x.device) #tensor([0., 1., 2., 3., 4.], device='cuda:0')
        temp1 = position.unsqueeze(1) #5 1
        temp2 = self.inv_timescales.unsqueeze(0) #1 256
        scaled_time = position.unsqueeze(1) * self.inv_timescales.unsqueeze(0) #5 256
        signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)],
                           dim=1) #5 512
        signal = F.pad(signal, (0, 0, 0, self.hidden_size % 2))
        signal = signal.view(1, max_length, self.hidden_size)
        return signal

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--problem', default='wmt32k')  # required=True
    # parser.add_argument('--train_step', type=int, default=200)
    # parser.add_argument('--batch_size', type=int, default=12)
    # parser.add_argument('--max_length', type=int, default=100)
    # parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--hidden_size', type=int, default=170)
    # parser.add_argument('--filter_size', type=int, default=256)
    # parser.add_argument('--warmup', type=int, default=16000)
    # parser.add_argument('--val_every', type=int, default=5)
    parser.add_argument('--dropout', type=float, default=0.1)
    # parser.add_argument('--label_smoothing', type=float, default=0.1)
    # parser.add_argument('--model', type=str, default='transformer')
    # parser.add_argument('--output_dir', type=str, default='./output')
    # parser.add_argument('--data_dir', type=str, default='./data')
    # parser.add_argument('--no_cuda', action='store_true')
    # parser.add_argument('--parallel', action='store_true')
    # parser.add_argument('--summary_grad', action='store_true')
    # parser.add_argument('--head_size', type=int, default=8)

    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--filter_size', type=int, default=256)
    parser.add_argument('--head_size', type=int, default=8)

    opt = parser.parse_args()

    model = Transformer(
                         n_layers=opt.n_layers,
                         hidden_size=opt.hidden_size,
                         filter_size=opt.filter_size,
                         dropout_rate=opt.dropout,
                         head_size = opt.head_size,
                         has_inputs=True,
                         src_pad_idx=None,
                         trg_pad_idx=None)

    x = torch.randn(32, 12, 170)

    y = model(x)
    print(y.shape)