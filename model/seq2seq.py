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

from src.data import SOURCE_ASSIST0910_SELF, SOURCE_ASSIST0910_ORIG
from src.utils import sAsMinutes, timeSince


# =========================
# Model
# =========================
class _Encoder(nn.Module):
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
    
    
class _Decoder(nn.Module):
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
#         print(input.shape)  # 1, 100
#         input = input.unsqueeze(0)
#         もしx_trg1つだけ取り出して渡すと上のようになるので、unsqueezeする。
        embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
#         prediction = self.out(output.squeeze(0))
        prediction = self.out(output)
        return prediction, hidden, cell

    
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, dev):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = dev
        
        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"
    
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        max_len = trg.shape[0]  # should be 1
        batch_size = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim
        
        hidden, cell = self.encoder(src)
        
#         print(trg.shape) # 2, 100
#         input_trg = trg[-1,:]  # should be 1, 100, ?
        input_trg = trg
         
        output, hidden, cell = self.decoder(input_trg, hidden, cell)
#         print(output.shape) # 1, 100, 250
#         outputs = output.unsqueeze(0)
        outputs = output
        # Knowledge State
        o_wro = torch.sigmoid(output[:,:, 2:2+124])
        o_cor = torch.sigmoid(output[:,:, 2+124:])
        outputs_prob = (o_cor / (o_cor + o_wro))
        
        return outputs, outputs_prob
    
def get_loss_batch_seq2seq(extend_forward, ks_loss):
    def loss_batch_encdec(model, loss_func, *args, opt=None):
        # Unpack data from DataLoader
        xs_src, xs_trg, ys, yq, ya, yp = args
        input_src = xs_src
        input_trg = xs_trg
        target = ys
        input_src = input_src.permute(1, 0)
        input_trg = input_trg.permute(1, 0)
        target = target.permute(1, 0)

        out, out_prob = model(input_src, input_trg)
    #     print(out.shape, out_prob.shape) # 1, 100, 250; 1, 100, 124
        out = out.permute(1, 0, 2)
        out_prob = out_prob.permute(1, 0, 2)

        pred = torch.sigmoid(out)  # [0, 1]区間にする

        # --- 指標評価用データ
        prob = torch.max(out_prob * yq, 2)[0]
        predicted = prob[:,-1 - extend_forward]
        actual = ya[:,-1 - extend_forward]

        predicted_ks = out_prob[:,-1,:].unsqueeze(1)
        
        loss = loss_func(prob, ya) 

        if opt:
            # バックプロバゲーション
            opt.zero_grad()
            loss.backward()
            opt.step()
            
        # Returns loss number, batch size
        return loss.item(), len(ys), predicted, actual, predicted_ks
    return loss_batch_encdec


def get_Seq2Seq(NUM_EMBEDDIGNS, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT, OUTPUT_DIM, DEC_EMB_DIM, DEC_DROPOUT, dev):
    enc = _Encoder(NUM_EMBEDDIGNS, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
    dec = _Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

    model = Seq2Seq(enc, dec, dev).to(dev)
    return model


