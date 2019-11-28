import math
import logging
from collections import namedtuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)


class TFLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, proj_size,
                 peephole=True, forget_bias=1.0):
        super(TFLSTMCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.proj_size = proj_size
        self.peephole = peephole
        self.forget_bias = forget_bias

        self.weight = nn.Parameter(torch.empty(4 * hidden_size, input_size + proj_size))
        self.bias = nn.Parameter(torch.empty(4 * hidden_size))
        if peephole:
            self.w_f_diag = nn.Parameter(torch.empty(hidden_size))
            self.w_i_diag = nn.Parameter(torch.empty(hidden_size))
            self.w_o_diag = nn.Parameter(torch.empty(hidden_size))
        else:
            self.register_parameter('w_f_diag', None)
            self.register_parameter('w_i_diag', None)
            self.register_parameter('w_o_diag', None)
        self.projection = nn.Linear(hidden_size, proj_size, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, hx):
        cat = torch.cat([input, hx[0]], dim=-1)
        multiplied = F.linear(cat, self.weight, self.bias)
        i, j, f, o = torch.split(multiplied, self.hidden_size, dim=-1)

        if self.peephole:
            c = (torch.sigmoid(f + self.w_f_diag * hx[1] + self.forget_bias) * hx[1]
                 + torch.sigmoid(i + self.w_i_diag * hx[1]) * torch.tanh(j))
            m = torch.sigmoid(o + self.w_o_diag * c) * torch.tanh(c)
        else:
            c = (torch.sigmoid(f + self.forget_bias) * hx[1]
                 + torch.sigmoid(i) * torch.tanh(j))
            m = torch.sigmoid(o) * torch.tanh(c)

        r = self.projection(m)
        return r, c


class TFLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, proj_size, num_layers,
                 peephole=True, forget_bias=1.0, batch_first=False, dropout=0.):
        super(TFLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.proj_size = proj_size
        self.num_layers = num_layers
        self.peephole = peephole
        self.forget_bias = forget_bias
        self.batch_first = batch_first

        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            cell_in_size = input_size if i == 0 else proj_size
            cell = TFLSTMCell(cell_in_size, hidden_size, proj_size,
                              peephole, forget_bias)
            self.layers.append(cell)

    def forward(self, input, hx=None):
        batch_dim, seq_dim = (0, 1) if self.batch_first else (1, 0)
        if hx is None:
            bsz = input.size(batch_dim)
            hx = []
            for __ in range(self.num_layers):
                h = torch.zeros(bsz, self.proj_size,
                                dtype=input.dtype, device=input.device)
                c = torch.zeros(bsz, self.hidden_size,
                                dtype=input.dtype, device=input.device)
                hx.append((h, c))

        output = []
        for i in range(input.size(seq_dim)):
            x = input[:, i] if self.batch_first else input[i]
            for j, layer in enumerate(self.layers):
                if j > 0:
                    x = self.dropout(x)
                h, c = layer(x, hx[j])
                hx[j] = (h, c)
                x = h
            output.append(h)
        return torch.stack(output, dim=batch_dim), hx


def init_forget(lstm, forget_bias=1.0):
    fb = forget_bias
    for i in range(lstm.num_layers):
        nhid = lstm.hidden_size
        getattr(lstm, 'bias_ih_l%d' % i)[nhid:2 * nhid].data.fill_(fb)
        getattr(lstm, 'bias_hh_l%d' % i)[nhid:2 * nhid].data.fill_(fb)


class WordRNN(nn.Module):
    def __init__(self, in_tokens, out_tokens, cell, din, dout, nhid,
                 nlayers=1, bidirec=False, autoenc=False, decoder_bias=False,
                 forget_bias=1.0, dropout=0., pad_idx=None):
        super(WordRNN, self).__init__()

        self.drop = nn.Dropout(dropout)
        demb = din * 2 if bidirec else din
        self.encoder = nn.Embedding(in_tokens, demb, padding_idx=pad_idx)
        self.decoder = nn.Linear(dout, out_tokens, decoder_bias)

        cell_type = cell.upper()

        def rnn():
            if cell_type == 'TFLSTM':
                return TFLSTM(din, nhid, dout, nlayers,
                              forget_bias=forget_bias,
                              batch_first=True, dropout=dropout)
            else:
                return getattr(nn, cell_type)(
                        din, nhid, nlayers, batch_first=True, dropout=dropout)

        self.rnn = rnn()
        d = dout if cell_type == 'TFLSTM' else nhid
        self.fc = nn.Linear(d, dout, bias=False)
        if bidirec:
            self.bw_rnn = rnn()
            self.bw_fc = nn.Linear(d, dout, bias=False)
        else:
            self.register_parameter('bw_rnn', None)
            self.register_parameter('bw_fc', None)

        self.reset_parameters()
        if forget_bias is not False and cell_type == 'LSTM':
            init_forget(self.rnn, forget_bias=forget_bias)
            if bidirec:
                init_forget(self.bw_rnn, forget_bias=forget_bias)

        self.cell_type = cell_type
        self.in_tokens = in_tokens
        self.out_tokens = out_tokens
        self.din = din
        self.dout = dout
        self.bidirec = bidirec
        self.autoenc = autoenc

        # for compatibility with pytorch_pretrained_bert
        self.config = namedtuple('Config', ['vocab_size'])(out_tokens)

    def forward(self, input, hidden=None, decode=True):
        emb = self.encoder(input)
        output, hidden = self.recur(emb, hidden)

        if decode:
            decoded = self.decoder(output)
            return decoded, hidden
        else:
            return output, hidden

    def recur(self, emb, hidden=None):
        if self.bidirec:
            fw_emb = emb[:, :, :self.din]
            bw_emb = torch.flip(emb, [1])[:, :, -self.din:]

            if hidden is None:
                fw_hidden, bw_hidden = None, None
            else:
                fw_hidden, bw_hidden = hidden

            fw_output, fw_hidden = self.rnn(fw_emb, fw_hidden)
            fw_output = self.fc(fw_output)
            bw_output, bw_hidden = self.bw_rnn(bw_emb, bw_hidden)
            bw_output = self.bw_fc(bw_output)

            output = (torch.flip(bw_output, [1]) + fw_output if self.autoenc else
                      torch.flip(bw_output, [1])[:, 2:] + fw_output[:, :-2])
            hidden = (fw_hidden, bw_hidden)
        else:
            output, hidden = self.rnn(emb, hidden)
            output = self.fc(output)
        return output, hidden

    def reset_parameters(self, initrange=0.05):
        for p in self.parameters():
            p.data.uniform_(-initrange, initrange)


class LM1B(WordRNN):
    def __init__(self, in_tokens, out_to_in, d, nhid, nlayers=2, dropout=0.5):
        super(LM1B, self).__init__(
                in_tokens, len(out_to_in), 'TFLSTM', d, d, nhid,
                nlayers=nlayers, bidirec=False, decoder_bias=True,
                forget_bias=1.0, dropout=dropout, pad_idx=None)

        delattr(self, 'fc')
        self.register_parameter('fc', None)

        self.index_map = nn.Parameter(
                torch.tensor(out_to_in, dtype=torch.long), requires_grad=False)

    def forward(self, input, hidden=None, decode=True):
        mapped_input = self.index_map[input]
        emb = self.encoder(mapped_input)
        output, hidden = self.recur(emb, hidden)

        if decode:
            decoded = self.decoder(output)
            return decoded, hidden
        else:
            return output, hidden

    def recur(self, emb, hidden=None):
        dropped_emb = self.drop(emb)
        output, hidden = self.rnn(dropped_emb, hidden)
        output = self.drop(output)
        return output, hidden

    @classmethod
    def from_tf(cls, tf_path, embeddings_npy_path, out_to_in, m):
        import tensorflow as tf
        _load = lambda v: tf.train.load_variable(tf_path, v)

        var_list = tf.train.list_variables(tf_path)
        var_dict = {name: shape for name, shape in var_list}
        nhid = var_dict['lstm/lstm_0/W_I_diag'][0]

        logging.info("Loading input word embeddings")
        embeddings_npy = np.load(embeddings_npy_path)
        in_tokens, d = embeddings_npy.shape

        model = cls(in_tokens, out_to_in, d, nhid)
        model.encoder.weight.data = torch.as_tensor(
                embeddings_npy, dtype=torch.float)

        logging.info("Loading softmax weights")
        model.decoder.bias.data = torch.as_tensor(_load('softmax/b'))
        ws = [_load('softmax/W_%d' % i) for i in range(m)]
        w = np.concatenate(ws, axis=1).reshape([-1, d])
        model.decoder.weight.data = torch.as_tensor(w)

        logging.info("Loading lstm parameters")
        for j, layer in enumerate(model.rnn.layers):
            w_step = (d + d) // m
            p_step = nhid // m
            for i in range(m):
                w_i = _load('lstm/lstm_%d/W_%d' % (j, i))
                layer.weight.data[:, i * w_step:(i + 1) * w_step] = \
                    torch.as_tensor(w_i.transpose())

                p_i = _load('lstm/lstm_%d/W_P_%d' % (j, i))
                layer.projection.weight.data[:, i * p_step:(i + 1) * p_step] = \
                    torch.as_tensor(p_i.transpose())

            b = _load('lstm/lstm_%d/B' % j)
            layer.bias.data = torch.as_tensor(b)

            w_f_diag = _load('lstm/lstm_%d/W_F_diag' % j)
            layer.w_f_diag.data = torch.as_tensor(w_f_diag)
            w_i_diag = _load('lstm/lstm_%d/W_I_diag' % j)
            layer.w_i_diag.data = torch.as_tensor(w_i_diag)
            w_o_diag = _load('lstm/lstm_%d/W_O_diag' % j)
            layer.w_o_diag.data = torch.as_tensor(w_o_diag)

        return model
