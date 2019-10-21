'''
EDDKT - Encoder-Decoder Deep Knowledge Tracing
~~~~~
Author: Qiushi Pan (@qqhann)
'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import TensorDataset, Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pack_padded_sequence, pack_sequence, pad_packed_sequence, pad_sequence
SEED = 0
torch.manual_seed(SEED)

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





# =========================
# Model
# =========================
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

        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)  # 6, 100, 1

        self.out = nn.Linear(hid_dim, output_dim)  # 100, 250

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.out(output)
        return prediction, hidden, cell



class EncDecDKT(nn.Module):
    def __init__(self,
                 NUM_EMBEDDIGNS, ENC_EMB_DIM, ENC_HID_DIM, ENC_N_LAYERS, ENC_DROPOUT,
                 OUTPUT_DIM, DEC_EMB_DIM, DEC_HID_DIM, DEC_N_LAYERS, DEC_DROPOUT,
                 N_SKILLS,
                 dev):
        super().__init__()
        self.N_SKILLS = N_SKILLS

        self.encoder = Encoder(
            NUM_EMBEDDIGNS, ENC_EMB_DIM, ENC_HID_DIM, ENC_N_LAYERS, ENC_DROPOUT)
        self.decoder = Decoder(
            OUTPUT_DIM, DEC_EMB_DIM, DEC_HID_DIM, DEC_N_LAYERS, DEC_DROPOUT)
        self.device = dev

        assert self.encoder.hid_dim == self.decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert self.encoder.n_layers == self.decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        max_len = trg.shape[0]  # should be 1
        batch_size = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim

        hidden, cell = self.encoder(src)

        input_trg = trg

        output, hidden, cell = self.decoder(input_trg, hidden, cell)
        # Knowledge State
        o_wro = torch.sigmoid(output[:,:, 2:2+self.N_SKILLS])
        o_cor = torch.sigmoid(output[:,:, 2+self.N_SKILLS:])
        outputs_prob = (o_cor / (o_cor + o_wro))
        # print(output.shape) => [len(trg), batch_size, 2M+PAD]
        # print(outputs_prob.shape) => [len(trg), batch_size, M]

        return outputs_prob


def get_loss_batch_encdec(extend_forward=0, ks_loss=False):
    def loss_batch_encdec(model, loss_func, *args, opt=None):
        torch.manual_seed(SEED)
        # Unpack data from DataLoader
        xs_src, xs_trg, ys, yq, ya, yp = args
        input_src = xs_src
        input_trg = xs_trg
        target = ys
        input_src = input_src.permute(1, 0)
        input_trg = input_trg.permute(1, 0)
        target = target.permute(1, 0)

        # print(input_src.shape, input_trg.shape)
        out_prob = model(input_src, input_trg)
        out_prob = out_prob.permute(1, 0, 2)

        # --- 指標評価用データ
        # print(out_prob.shape) => [1, 11, 124] = (batch_size, len(y), skill_size)
        prob = torch.max(out_prob * yq, 2)[0]
        # print(prob.shape) => [100, 11] = (batch_size, len(y)=ef+1)
        # print(ya.shape) => [100, 11] = (batch_size, len(y))
        predicted = prob[:,-1]
        actual_q = yq[:,-1]
        actual_a = ya[:,-1]
        # ---

        # Knowledge State
        # TODO: using predicted_ks for both ks-vector learning and heatmap is causing problem. Fix it.
        predicted_ks = out_prob[:,-1,:].unsqueeze(1)
        hm_pred_ks = out_prob[:, -1, :].squeeze()
        # print(out_prob.shape)
        # print(out_prob.squeeze().shape)
        # print(predicted_ks.shape)
        # print('-------------------')

        if ks_loss:  # KS Loss learning
            loss = loss_func(predicted_ks, yp.float())
        else:
            # Olsfashion  # DeltaQ-A Loss learning (Piech et al. style)
            loss = loss_func(prob, ya)

        if opt:
            # バックプロバゲーション
            opt.zero_grad()
            loss.backward()
            opt.step()

        # Returns loss number, batch size
        return loss.item(), len(ys), predicted, actual_q, actual_a, hm_pred_ks, None, None

    return loss_batch_encdec






# EOF













































# It is important to leave some room. That is how life is, isn't it?
