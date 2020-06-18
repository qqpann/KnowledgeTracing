'''
GEDDKT - Generative Encoder-Decoder Deep Knowledge Tracing
~~~~~
Author: Qiushi Pan (@qqhann)
'''
import logging
import math
import os
import pickle
import random
import sys
import time
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
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split

from src.data import SOURCE_ASSIST0910_ORIG, SOURCE_ASSIST0910_SELF
from src.utils import sAsMinutes, timeSince
from model._base import BaseKTModel


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


class GEDDKT(nn.Module, BaseKTModel):
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device

        NUM_EMBEDDIGNS = 2 * config.n_skills + 2
        input_size = ceil(log(2 * config.n_skills))
        INPUT_DIM, ENC_EMB_DIM, ENC_DROPOUT = NUM_EMBEDDIGNS, input_size, 0.6
        OUTPUT_DIM, DEC_EMB_DIM, DEC_DROPOUT = NUM_EMBEDDIGNS, input_size, 0.6
        HID_DIM, N_LAYERS = config.eddkt['hidden_size'], config.eddkt['n_layers']
        self.generative = config.eddkt['generative']
        self.teacher_forcing_ratio = config.eddkt['teacher_forcing_ratio']

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

    def forward_loss(self, input_enc, input_dec, yqs):
        max_len = input_dec.shape[0]  # should be 1
        batch_size = input_dec.shape[1]
        trg_vocab_size = self.decoder.output_dim

        hidden, cell = self.encoder(input_enc)

        input_trg = input_dec

        if self.generative:
            input_trg_ = input_trg[0:1, :]
            outputs_prob = torch.zeros([input_trg.shape[0], self.config.batch_size, self.N_SKILLS],
                                       dtype=torch.float32, device=self.device)
            for di in range(input_trg.shape[0]):
                output, hidden, cell = self.decoder(input_trg_, hidden, cell)
                o_all = torch.sigmoid(output[:, :, 2:])
                o_wro = torch.sigmoid(output[:, :, 2:2+self.N_SKILLS])
                o_cor = torch.sigmoid(output[:, :, 2+self.N_SKILLS:])
                outputs_prob_ = (o_cor / (o_cor + o_wro))
                outputs_prob[di] = outputs_prob_

                # random.random() returns real number in the range[0.0, 1.0)
                use_teacher_forcing = random.random() > self.teacher_forcing_ratio
                if use_teacher_forcing:
                    input_trg_ = torch.max(o_all, 2)[1]
                else:
                    input_trg_ = input_trg[di+1:di+2, :]
                # print(input_trg_, input_trg_.shape) #=> [1,100] [1, batch_size]
        else:
            # print(input_trg.shape, hidden.shape, cell.shape) # => (16, 100), (2, 100, 200), (2, 100, 200)
            output, hidden, cell = self.decoder(input_trg, hidden, cell)
            # print(output.shape) # => (16, 100, 250)
            # Knowledge State
            o_wro = torch.sigmoid(output[:, :, 2:2+self.N_SKILLS])
            o_cor = torch.sigmoid(output[:, :, 2+self.N_SKILLS:])
            outputs_prob = (o_cor / (o_cor + o_wro))
            # print(output.shape) => [len(trg), batch_size, 2M+PAD]
            # print(outputs_prob.shape) => [len(trg), batch_size, M]

        return outputs_prob

    def forward(self, xseq, yseq, mask, opt=None):
        i_batch = self.config.batch_size
        if i_batch != xseq.shape[0]:
            # warnings.warn(f'batch size mismatch {i_batch} != {xseq.shape[0]}')
            i_batch = xseq.shape[0]
        i_skill = self.config.n_skills
        i_seqen = self.config.sequence_size
        i_extfw = self.config.eddkt['extend_forward']
        i_extbw = self.config.eddkt['extend_backward']
        assert xseq.shape == (i_batch, i_seqen, 2)
        assert yseq.shape == (i_batch, i_seqen, 2)
        # onehot_size = i_skill * 2 + 2
        device = self.device
        # extend_forward=0; ks_loss=False
        xseq = xseq.permute(1, 0, 2)
        yseq = yseq.permute(1, 0, 2)

        i_seqlen_enc = i_seqen - i_extfw - 1
        i_seqlen_dec = i_extbw + i_extfw + 1
        xseq_enc = xseq[:i_seqlen_enc]
        xseq_dec = xseq[i_seqlen_enc:]
        yseq = yseq[i_seqlen_enc:]
        # print(xseq_enc.shape, xseq_dec.shape, yseq.shape, i_seqlen_enc, i_seqlen_dec)
        # TODO: use only torch to simplify
        input_enc = torch.matmul(xseq_enc.float().to(device), torch.Tensor(
            [[1], [i_skill]]).to(device)).long().to(device)
        assert input_enc.shape == (
            i_seqen-1-i_extfw, i_batch, 1), input_enc.shape
        input_enc = input_enc.squeeze(2)
        # input_src = F.one_hot(input_src, num_classes=onehot_size).float()
        input_dec = torch.matmul(xseq_dec.float().to(device), torch.Tensor(
            [[1], [i_skill]]).to(device)).long().to(device)
        assert input_dec.shape == (1+i_extfw, i_batch, 1), input_dec.shape
        input_dec = input_dec.squeeze(2)
        # input_trg = F.one_hot(input_trg, num_classes=onehot_size).float()
        yqs = torch.matmul(yseq.float().to(device), torch.Tensor(
            [[1], [0]]).to(device)).long().to(device)
        assert yqs.shape == (1+i_extfw, i_batch, 1), yqs.shape
        yqs = yqs.squeeze(2)
        yqs = F.one_hot(yqs, num_classes=i_skill).float()
        target = torch.matmul(yseq.float().to(
            device), torch.Tensor([[0], [1]]).to(device)).to(device)
        assert target.shape == (1+i_extfw, i_batch, 1)
        # target = target.squeeze(2)
        mask = mask.to(device)

        # print(input_src.shape, input_trg.shape)
        out = self.forward_loss(input_enc, input_dec, yqs)
        # print(out, out.shape)
        pred_vect = out  # .permute(1, 0, 2)
        assert tuple(pred_vect.shape) == (1+i_extbw+i_extfw, i_batch, i_skill), \
            f"Unexpected shape {pred_vect.shape} != {(1+i_extbw+i_extfw, i_batch, i_skill)}"

        pred_prob = torch.max(pred_vect * yqs, 2)[0]

        # Knowledge State
        # TODO: using predicted_ks for both ks-vector learning and heatmap is causing problem. Fix it.
        # predicted_ks = out_prob[:, -1, :].unsqueeze(1)
        # hm_pred_ks = out_prob[:, -1, :].squeeze()

        # if ks_loss:  # KS Loss learning
        #     loss = loss_func(predicted_ks, yp.float())

        # Olsfashion  # DeltaQ-A Loss learning (Piech et al. style)
        # TODO: pad for GEDDKT
        if False and self.config.pad == True:
            # TODO: Special care needed to adapt mask to dec only
            _pred_prob = pred_prob.masked_select(mask.permute(1, 0))
            _target = target.squeeze(2).masked_select(mask.permute(1, 0))
        else:
            _pred_prob = pred_prob
            _target = target.squeeze(2)
        assert _pred_prob.shape == _target.shape, f'{_pred_prob.shape}!={_target.shape}'
        loss = self._loss(_pred_prob, _target)
        # print(loss, loss.shape) #=> scalar, []

        out_dic = {
            'loss': loss,
            'pred_vect': pred_vect,
            'pred_prob': pred_prob,
            'filtered_pred': _pred_prob,
            'filtered_target': _target,
        }

        if True:
            assert yqs.shape == (i_extfw+1, i_batch, i_skill), \
                'Expected {}, got {}'.format(
                    (i_extfw+1, i_batch, i_skill), yqs.shape)
            assert target.shape == (i_extfw+1, i_batch, 1), \
                'Expected {}, got {}'.format(
                    (i_extfw+1, i_batch, 1), target.shape)
            dqa = yqs * target
            Sdqa = torch.cumsum(dqa, dim=0)
            Sdq = torch.cumsum(yqs, dim=0)
            ksvector_l1 = torch.sum(torch.abs((Sdq * pred_vect) - (Sdqa))) \
                / (Sdq.shape[0] * Sdq.shape[1] * Sdq.shape[2])
            out_dic['loss'] += self.config.ksvector_l1 * ksvector_l1
            out_dic['ksvector_l1'] = ksvector_l1.item()
            out_dic['Sdqa'] = Sdqa
            out_dic['Sdq'] = Sdq

        if self.config.reconstruction or self.config.reconstruction_and_waviness:
            reconstruction_target = torch.matmul(xseq_dec.float().to(
                device), torch.Tensor([[0], [1]]).to(device)).to(device)
            assert reconstruction_target.shape == (1+i_extfw, i_batch, 1)
            reconstruction_target = reconstruction_target.squeeze(2)
            reconstruction_loss = self._loss(_pred_prob, reconstruction_target)
            out_dic['loss'] += self.config.lambda_rec * reconstruction_loss
            out_dic['reconstruction_loss'] = reconstruction_loss.item()
            out_dic['filtered_target_c'] = reconstruction_target.masked_select(mask.permute(1, 0)) if self.config.pad == True else reconstruction_target

        if self.config.waviness == True:
            # assert pred_vect.shape[0] > 1, pred_vect
            waviness_norm_l1 = torch.abs(
                pred_vect[1:, :, :] - pred_vect[:-1, :, :])
            waviness_l1 = torch.sum(
                waviness_norm_l1) / ((pred_vect.shape[0] - 1) * pred_vect.shape[1] * pred_vect.shape[2])
            lambda_l1 = self.config.lambda_l1
            out_dic['loss'] += lambda_l1 * waviness_l1
            out_dic['waviness_l1'] = waviness_l1.item()

        if self.config.waviness == True:
            # assert pred_vect.shape[0] > 1, pred_vect
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

