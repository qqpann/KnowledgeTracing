'''
EDDKT - Encoder-Decoder Deep Knowledge Tracing
~~~~~
Author: Qiushi Pan (@qqhann)
'''
from src.utils import sAsMinutes, timeSince
from src.data import prepare_data, prepare_heatmap_data, SOURCE_ASSIST0910_SELF, SOURCE_ASSIST0910_ORIG

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
import pandas as pd
import numpy as np
from typing import List, Tuple, Set, Dict
from pathlib import Path
from math import log, ceil
import math
import random
import logging
import pickle
import time
import sys
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import TensorDataset, Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pack_padded_sequence, pack_sequence, pad_packed_sequence, pad_sequence
SEED = 0
torch.manual_seed(SEED)


class Encoder(nn.Module):
    def __init__(self, num_embeddings, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = dropout

        # Layers
        self.embedding = nn.Embedding(num_embeddings, emb_dim)

        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        outputs, (hidden, cell) = self.rnn(embedded)
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout = dropout

        # Layers
        self.embedding = nn.Embedding(output_dim, emb_dim)  # 250->6

        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers,
                           dropout=dropout)  # 6, 100, 1

        self.out = nn.Linear(hid_dim, output_dim)  # 100, 250

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.out(output)
        return prediction, hidden, cell


class EDDKT(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device

        NUM_EMBEDDIGNS = 2 * config.n_skills + 2
        input_size = ceil(log(2 * config.n_skills))
        INPUT_DIM, ENC_EMB_DIM, ENC_DROPOUT = NUM_EMBEDDIGNS, input_size, 0.6
        OUTPUT_DIM, DEC_EMB_DIM, DEC_DROPOUT = NUM_EMBEDDIGNS, input_size, 0.6
        HID_DIM, N_LAYERS = config.eddkt['hidden_size'], config.eddkt['n_layers']

        self.N_SKILLS = config.n_skills

        self.encoder = Encoder(
            NUM_EMBEDDIGNS, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
        self.decoder = Decoder(
            OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

        assert self.encoder.hid_dim == self.decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert self.encoder.n_layers == self.decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"

        self._loss = nn.BCELoss()

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        max_len = trg.shape[0]  # should be 1
        batch_size = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim

        hidden, cell = self.encoder(src)

        input_trg = trg

        output, hidden, cell = self.decoder(input_trg, hidden, cell)
        # Knowledge State
        o_wro = torch.sigmoid(output[:, :, 2:2+self.N_SKILLS])
        o_cor = torch.sigmoid(output[:, :, 2+self.N_SKILLS:])
        outputs_prob = (o_cor / (o_cor + o_wro))
        # print(output.shape) => [len(trg), batch_size, 2M+PAD]
        # print(outputs_prob.shape) => [len(trg), batch_size, M]

        return outputs_prob

    def forward_loss(self, input_src, input_trg, yqs, target):
        # print(input_src.shape, input_trg.shape)
        out = self.forward(input_src, input_trg)
        # print(out, out.shape)
        pred_vect = out #.permute(1, 0, 2)
        # assert tuple(pred_vect.shape) == (self.config.sequence_size, self.config.batch_size, self.config.n_skills), \
        #     "Unexpected shape {}".format(pred_vect.shape)

        pred_prob = torch.max(pred_vect * yqs, 2)[0]

        # Knowledge State
        # TODO: using predicted_ks for both ks-vector learning and heatmap is causing problem. Fix it.
        # predicted_ks = out_prob[:, -1, :].unsqueeze(1)
        # hm_pred_ks = out_prob[:, -1, :].squeeze()

        # if ks_loss:  # KS Loss learning
        #     loss = loss_func(predicted_ks, yp.float())

        # Olsfashion  # DeltaQ-A Loss learning (Piech et al. style)
        loss = self._loss(pred_prob, target)
        # print(loss, loss.shape) #=> scalar, []

        out_dic = {
            'loss': loss,
            'pred_vect': pred_vect,
            'pred_prob': pred_prob
        }

        if self.config.waviness_l1 == True:
            assert pred_vect.shape[0] > 1, pred_vect
            waviness_norm_l1 = torch.abs(
                pred_vect[1:, :, :] - pred_vect[:-1, :, :])
            waviness_l1 = torch.sum(
                waviness_norm_l1) / ((pred_vect.shape[0] - 1) * pred_vect.shape[1] * pred_vect.shape[2])
            lambda_l1 = self.config.lambda_l1
            out_dic['loss'] += lambda_l1 * waviness_l1
            out_dic['waviness_l1'] = waviness_l1.item()

        if self.config.waviness_l2 == True:
            assert pred_vect.shape[0] > 1, pred_vect
            waviness_norm_l2 = torch.pow(
                pred_vect[1:, :, :] - pred_vect[:-1, :, :], 2)
            waviness_l2 = torch.sum(
                waviness_norm_l2) / ((pred_vect.shape[0] - 1) * pred_vect.shape[1] * pred_vect.shape[2])
            lambda_l2 = self.config.lambda_l2
            out_dic['loss'] += lambda_l2 * waviness_l2
            out_dic['waviness_l2'] = waviness_l2.item()

        return out_dic

    def loss_batch(self, xseq, yseq, opt=None):
        # extend_forward=0; ks_loss=False
        # print(xseq.shape, yseq.shape)
        xseq = xseq.permute(1, 0, 2)
        yseq = yseq.permute(1, 0, 2)

        xseq_src = xseq[:-1-self.config.eddkt['extend_forward']]
        xseq_trg = xseq[-1-self.config.eddkt['extend_forward'] -
                         self.config.eddkt['extend_backward']:]
        yseq = yseq[-1-self.config.eddkt['extend_forward'] -
                    self.config.eddkt['extend_backward']:]
        # print(xseq_src.shape, xseq_trg.shape, yseq.shape)
        # TODO: use only torch to simplify
        skill_n = self.config.n_skills
        onehot_size = skill_n * 2 + 2
        device = self.device
        input_src = torch.LongTensor(
            np.dot(xseq_src.cpu().numpy(), np.array([[1], [skill_n]]))).to(device)  # -> (100, 20, 1)
        input_src = input_src.squeeze(2)
        # input_src = F.one_hot(input_src, num_classes=onehot_size).float()
        input_trg = torch.LongTensor(
            np.dot(xseq_trg.cpu().numpy(), np.array([[1], [skill_n]]))).to(device)  # -> (100, 20, 1)
        input_trg = input_trg.squeeze(2)
        # input_trg = F.one_hot(input_trg, num_classes=onehot_size).float()
        yqs = torch.LongTensor(
            np.dot(yseq.cpu().numpy(), np.array([[1], [0]]))).to(device)  # -> (100, 20, 1)
        yqs = yqs.squeeze(2)
        yqs = F.one_hot(yqs, num_classes=skill_n).float()
        target = torch.Tensor(
            np.dot(yseq.cpu().numpy(), np.array([[0], [1]]))).to(device)  # -> (100, 20, 1)
        target = target.squeeze(2)
        # print(input_src.shape, input_trg.shape, yqs.shape, target.shape)

        out = self.forward_loss(input_src, input_trg, yqs, target)
        loss = out['loss']

        if opt:
            # バックプロバゲーション
            opt.zero_grad()
            loss.backward()
            opt.step()

        # Returns loss number, batch size
        return out
