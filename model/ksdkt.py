import logging
import math
import os
import pickle
import random
import sys
import time
import warnings
from math import ceil, log
from pathlib import Path
from typing import Dict, List, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn import metrics
from torch.autograd import Variable
from torch.nn.utils.rnn import (pack_padded_sequence, pack_sequence,
                                pad_packed_sequence, pad_sequence)
from torch.utils.data import DataLoader, Dataset, TensorDataset

from src.data import SOURCE_ASSIST0910_ORIG, SOURCE_ASSIST0910_SELF
from src.utils import sAsMinutes, timeSince
from model._base import BaseKTModel


class KSDKT(nn.Module, BaseKTModel):
    ''' Expansion of original DKT '''

    def __init__(self, config, device, bidirectional=False):
        super().__init__()
        self.config = config
        self.device = device

        self.model_name = config.model_name
        self.input_size = ceil(log(2 * config.n_skills))
        self.output_size = config.n_skills
        self.batch_size = config.batch_size
        self.hidden_size = config.dkt['hidden_size']
        self.n_layers = config.dkt['n_layers']
        self.bidirectional = config.dkt['bidirectional']
        self.directions = 2 if self.bidirectional else 1
        self.dropout = self.config.dkt['dropout_rate']
        self.cumsum_weights = torch.tensor([[[config.ksvector_weight_base ** i for i in range(config.sequence_size)]]],  dtype=torch.int64, device=device).permute(2, 1, 0)

        # self.cs_basis = torch.randn(config.n_skills * 2 + 2, self.input_size).to(device)
        self.embedding = nn.Embedding(config.n_skills * 2 + config.dkt['preserved_tokens'], self.input_size).to(device)

        # https://pytorch.org/docs/stable/nn.html#rnn
        if self.model_name == 'dkt:rnn':
            self.rnn = nn.RNN(self.input_size, self.hidden_size, self.n_layers,
                              nonlinearity='tanh', dropout=self.dropout, bidirectional=self.bidirectional)
        elif self.model_name == 'ksdkt':
            self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.n_layers,
                                dropout=self.dropout, bidirectional=self.bidirectional)
        else:
            raise ValueError('Model name not supported')
        self.fc = self.init_fc()
        # self.sigmoid = nn.Sigmoid()

        self._loss = nn.BCELoss()

    def forward(self, xseq, yseq, mask, opt=None):
        i_batch = self.config.batch_size
        if i_batch != xseq.shape[0]:
            # warnings.warn(f'batch size mismatch {i_batch} != {xseq.shape[0]}')
            i_batch = xseq.shape[0]
        i_skill = self.config.n_skills
        i_seqen = self.config.sequence_size
        assert xseq.shape == (i_batch, i_seqen, 2), '{} != {}'.format(xseq.shape, (i_batch, i_seqen, 2))
        assert yseq.shape == (i_batch, i_seqen, 2), '{} != {}'.format(yseq.shape, (i_batch, i_seqen, 2))
        # onehot_size = i_skill * 2 + 2
        device = self.device
        # Convert to onehot; (12, 1) -> (0, 0, ..., 1, 0, ...)
        # https://pytorch.org/docs/master/nn.functional.html#one-hot
        inputs = torch.matmul(xseq.float().to(device), torch.Tensor(
            [[1], [i_skill]]).to(device)).long().to(device)
        assert inputs.shape == (i_batch, i_seqen, 1)
        # inputs = inputs.squeeze()
        # inputs = F.one_hot(inputs, num_classes=onehot_size).float()
        yqs = torch.matmul(yseq.float().to(device), torch.Tensor(
            [[1], [0]]).to(device)).long().to(device)
        assert yqs.shape == (i_batch, i_seqen, 1)
        yqs = yqs.squeeze(2)
        assert torch.max(yqs).item() < i_skill, f'{torch.max(yqs)} < {i_skill} not fulfilled'
        yqs = F.one_hot(yqs, num_classes=i_skill).float()
        assert yqs.shape == (i_batch, i_seqen, i_skill)
        target = torch.matmul(yseq.float().to(
            device), torch.Tensor([[0], [1]]).to(device)).to(device)
        assert target.shape == (i_batch, i_seqen, 1)
        mask = mask.to(device)

        inputs = inputs.permute(1, 0, 2)
        yqs = yqs.permute(1, 0, 2)
        target = target.permute(1, 0, 2)

        inputs = self.embedding(inputs).squeeze(2)
        out, _Hn = self.lstm(inputs, self.init_Hidden0(i_batch))
        # top_n, top_i = out.topk(1)
        # decoded = self.decoder(out.contiguous().view(out.size(0) * out.size(1), out.size(2)))
        out = self.fc(out)
        # decoded = self.sigmoid(decoded)
        # print(out.shape) => [20, 100, 124] (sequence_len, batch_size, skill_size)

        pred_vect = torch.sigmoid(out)  # [0, 1]区間にする
        assert pred_vect.shape == (i_seqen, i_batch, i_skill), \
            "Unexpected shape {}".format(pred_vect.shape)
        pred_prob = torch.max(pred_vect * yqs, 2)[0]
        assert pred_prob.shape == (i_seqen, i_batch), \
            "Unexpected shape {}".format(pred_prob.shape)
        if self.config.pad == True:
            # _pred_prob = pack_padded_sequence(
            #     pred_prob.unsqueeze(2), mask, enforce_sorted=False).data
            # _target = pack_padded_sequence(
            #     target, mask, enforce_sorted=False).data
            _pred_prob = pred_prob.masked_select(mask.permute(1, 0))
            _target = target.squeeze(2).masked_select(mask.permute(1, 0))
        else:
            _pred_prob = pred_prob
            _target = target.squeeze(2)
        loss = self._loss(_pred_prob, _target)
        # print(loss, loss.shape) #=> scalar, []

        out_dic = {
            'loss': loss,
            'pred_vect': pred_vect,  # (20, 100, 124)
            'pred_prob': pred_prob,  # (20, 100)
            'filtered_pred': _pred_prob,
            'filtered_target': _target,
        }

        if True:
            assert yqs.shape == (i_seqen, i_batch, i_skill), \
                'Expected {}, got {}'.format((i_seqen, i_batch, i_skill), yqs.shape)
            assert target.shape == (i_seqen, i_batch, 1), \
                'Expected {}, got {}'.format((i_seqen, i_batch, 1), target.shape)
            dqa = yqs * target
            Sdqa = self.cumsum_weights * torch.cumsum(dqa, dim=0)
            Sdq = self.cumsum_weights * torch.cumsum(yqs, dim=0)
            ksvector_l1 = torch.sum(torch.abs((Sdq * pred_vect) - (Sdqa))) \
                / torch.sum(Sdq > 0)
                # / (Sdq.shape[0] * Sdq.shape[1] * Sdq.shape[2])
            out_dic['loss'] += self.config.ksvector_l1 * ksvector_l1
            out_dic['ksvector_l1'] = ksvector_l1.item()
            out_dic['Sdqa'] = Sdqa
            out_dic['Sdq'] = Sdq

        if self.config.waviness_l1 == True:
            waviness_norm_l1 = torch.abs(
                pred_vect[1:, :, :] - pred_vect[:-1, :, :])
            waviness_l1 = torch.sum(
                waviness_norm_l1) / ((pred_vect.shape[0] - 1) * pred_vect.shape[1] * pred_vect.shape[2])
            lambda_l1 = self.config.lambda_l1
            out_dic['loss'] += lambda_l1 * waviness_l1
            out_dic['waviness_l1'] = waviness_l1.item()

        if self.config.waviness_l2 == True:
            waviness_norm_l2 = torch.pow(
                pred_vect[1:, :, :] - pred_vect[:-1, :, :], 2)
            waviness_l2 = torch.sum(
                waviness_norm_l2) / ((pred_vect.shape[0] - 1) * pred_vect.shape[1] * pred_vect.shape[2])
            lambda_l2 = self.config.lambda_l2
            out_dic['loss'] += lambda_l2 * waviness_l2
            out_dic['waviness_l2'] = waviness_l2.item()


        if opt:
            # バックプロバゲーション
            opt.zero_grad()
            out_dic['loss'].backward()
            opt.step()

        return out_dic

    def init_h0(self, batch_size):
        return torch.zeros(self.n_layers * self.directions, batch_size, self.hidden_size).to(self.device)

    def init_c0(self, batch_size):
        return torch.zeros(self.n_layers * self.directions, batch_size, self.hidden_size).to(self.device)

    def init_Hidden0(self, i_batch: int):
        if self.model_name == 'dkt:rnn':
            h0 = self.init_h0(i_batch)
            return h0
        elif self.model_name == 'ksdkt':
            h0 = self.init_h0(i_batch)
            c0 = self.init_c0(i_batch)
            return (h0, c0)

    def init_fc(self):
        return nn.Linear(self.hidden_size * self.directions, self.output_size).to(self.device)

