import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import TensorDataset, Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pack_padded_sequence, pack_sequence, pad_packed_sequence, pad_sequence

import os
import sys
import time
import pickle
import logging
import random
import math
from math import log, ceil
from pathlib import Path
from typing import List, Tuple, Set, Dict

import numpy as np
import pandas as pd
from sklearn import metrics

import seaborn as sns
import matplotlib.pyplot as plt

from src.data import prepare_data, prepare_heatmap_data, SOURCE_ASSIST0910_SELF, SOURCE_ASSIST0910_ORIG
from src.utils import sAsMinutes, timeSince


class BaseDKT(nn.Module):
    ''' オリジナルのDKT '''

    def __init__(self, config, dev, model_name, n_input, n_hidden, n_output, n_layers, batch_size, dropout=0.6, bidirectional=False):
        super().__init__()
        self.config = config
        self.dev = dev
        self.model_name = model_name
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.batch_size = batch_size

        self.bidirectional = bidirectional
        self.directions = 2 if self.bidirectional else 1

        nonlinearity = 'tanh'
        # https://pytorch.org/docs/stable/nn.html#rnn
        if model_name == 'basernn':
            self.rnn = nn.RNN(n_input, n_hidden, n_layers,
                              nonlinearity=nonlinearity, dropout=dropout, bidirectional=self.bidirectional)
        elif model_name == 'baselstm':
            self.lstm = nn.LSTM(n_input, n_hidden, n_layers,
                                dropout=dropout, bidirectional=self.bidirectional)
        else:
            raise ValueError('Model name not supported')
        self.decoder = nn.Linear(n_hidden * self.directions, n_output)
        # self.sigmoid = nn.Sigmoid()

        self._loss = nn.BCELoss()

    def forward(self, inputs, yqs, target):
        if self.model_name == 'basernn':
            h0 = self.initHidden0()
            out, _hn = self.rnn(inputs, h0)
        elif self.model_name == 'baselstm':
            h0 = self.initHidden0()
            c0 = self.initC0()
            out, (_hn, _cn) = self.lstm(inputs, (h0, c0))
        # top_n, top_i = out.topk(1)
        # decoded = self.decoder(out.contiguous().view(out.size(0) * out.size(1), out.size(2)))
        out = self.decoder(out)
        # decoded = self.sigmoid(decoded)
        # print(out.shape) => [20, 100, 124] (sequence_len, batch_size, skill_size)

        pred = torch.sigmoid(out)  # [0, 1]区間にする
        # pred.shape: (20, 100, 124); (seqlen, batch_size, skill_size)
        # yqs.shape: (20, 100, 124); (seqlen, batch_size, skill_size)
        prob = torch.max(pred * yqs, 2)[0]
        # print(pred, pred.shape)  # (20, 100, 124)
        # print(prob, prob.shape)  # (20, 100)
        # print(target, target.shape)  # (20, 100)
        loss = self._loss(prob, target)  # TODO: 最後の1個だけじゃなくて、その他も損失関数に利用したら？

        out_dic = {
            'loss': loss
        }
        return out_dic

    def initHidden0(self):
        return torch.zeros(self.n_layers * self.directions, self.batch_size, self.n_hidden).to(self.dev)

    def initC0(self):
        return torch.zeros(self.n_layers * self.directions, self.batch_size, self.n_hidden).to(self.dev)

    def loss_batch(self, xseq, yseq, opt=None):
        '''
        DataLoaderの１イテレーションから，
        適宜back propagationし，
        lossを返す．

        xs: shapeは[100, 20, 654]
        yq: qのonehot配列からなる配列
        ya: aの0,1 intからなる配列
        '''
        # print(yq.shape) => [100, 124] = [batch_size, skill_size]
        # print(xseq.shape, yseq.shape) => [100, 20, 2], [100, 20, 2]
        # Convert to onehot; (12, 1) -> (0, 0, ..., 1, 0, ...)
        # https://pytorch.org/docs/master/nn.functional.html#one-hot
        skill_n = self.config.n_skills
        onehot_size = skill_n * 2 + 2
        device = self.dev
        # inputs = torch.dot(xseq, torch.as_tensor([[1], [skill_n]]))
        inputs = torch.LongTensor(np.dot(xseq.cpu().numpy(), np.array([[1], [skill_n]]))).to(device)  # -> (100, 20, 1)
        inputs = inputs.squeeze()
        inputs = F.one_hot(inputs, num_classes=onehot_size).float()
        yqs = torch.LongTensor(np.dot(yseq.cpu().numpy(), np.array([[1], [0]]))).to(device)  # -> (100, 20, 1)
        yqs = yqs.squeeze()
        yqs = F.one_hot(yqs, num_classes=skill_n).float()
        target = torch.Tensor(np.dot(yseq.cpu().numpy(), np.array([[0], [1]]))).to(device)  # -> (100, 20, 1)
        target = target.squeeze()
        # print(target, target.shape)
        compressed_sensing = True
        if compressed_sensing and onehot_size != self.n_input:
            SEED = 0
            torch.manual_seed(SEED)
            cs_basis = torch.randn(onehot_size, self.n_input).to(device)
            inputs = torch.mm(
                inputs.contiguous().view(-1, onehot_size), cs_basis)
            # https://pytorch.org/docs/stable/nn.html?highlight=rnn#rnn
            # inputの説明を見ると、input of shape (seq_len, batch, input_size)　とある
            inputs = inputs.view(self.batch_size, self.config.sequence_size, self.n_input)
        inputs = inputs.permute(1, 0, 2)

        yqs = yqs.permute(1, 0, 2)
        target = target.permute(1, 0)
        # actual_q = yq
        # actual_a = ya

        out = self.forward(inputs, yqs, target)
        loss = out['loss']

        if opt:
            # バックプロバゲーション
            opt.zero_grad()
            loss.backward()
            opt.step()

        # hm_pred_ks = pred[-1].squeeze()

        return loss
