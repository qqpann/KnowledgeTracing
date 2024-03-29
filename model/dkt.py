"""
Reproduction implementation of the original DKT model by Piech et al.
"""
import logging
import math
import os
import pickle
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
from torch.utils.data import DataLoader, Dataset, TensorDataset

from src.data import SOURCE_ASSIST0910_ORIG, SOURCE_ASSIST0910_SELF
from src.utils import sAsMinutes, timeSince


class DKT(nn.Module):
    """ オリジナルのDKT """

    def __init__(self, config, device, bidirectional=False):
        super().__init__()
        self.config = config
        self.device = device

        self.model_name = config.model_name
        self.input_size = ceil(log(2 * config.n_skills))
        self.hidden_size = config.dkt["hidden_size"]
        self.output_size = config.n_skills
        self.n_layers = config.dkt["n_layers"]
        self.batch_size = config.batch_size
        self.bidirectional = config.dkt["bidirectional"]
        self.directions = 2 if self.bidirectional else 1

        # self.cs_basis = torch.randn(config.n_skills * 2 + 2, self.input_size).to(device)
        self.embedding = nn.Embedding(config.n_skills * 2 + 2, self.input_size).to(
            device
        )

        nonlinearity = "tanh"
        # https://pytorch.org/docs/stable/nn.html#rnn
        if self.model_name == "dkt:rnn":
            self.rnn = nn.RNN(
                self.input_size,
                self.hidden_size,
                self.n_layers,
                nonlinearity=nonlinearity,
                dropout=self.dkt["dropout_rate"],
                bidirectional=self.bidirectional,
            )
        elif self.model_name == "dkt":
            self.lstm = nn.LSTM(
                self.input_size,
                self.hidden_size,
                self.n_layers,
                dropout=config.dkt["dropout_rate"],
                bidirectional=self.bidirectional,
            )
        else:
            raise ValueError("Model name not supported")
        self.decoder = nn.Linear(self.hidden_size * self.directions, self.output_size)
        # self.sigmoid = nn.Sigmoid()

        self._loss = nn.BCELoss()

    def forward(self, inputs):
        inputs = self.embedding(inputs).squeeze(2)
        if self.model_name == "dkt:rnn":
            h0 = self.initHidden0()
            out, _hn = self.rnn(inputs, h0)
        elif self.model_name == "dkt":
            h0 = self.initHidden0()
            c0 = self.initC0()
            out, (_hn, _cn) = self.lstm(inputs, (h0, c0))
        # top_n, top_i = out.topk(1)
        # decoded = self.decoder(out.contiguous().view(out.size(0) * out.size(1), out.size(2)))
        out = self.decoder(out)
        # decoded = self.sigmoid(decoded)
        # print(out.shape) => [20, 100, 124] (sequence_len, batch_size, skill_size)
        return out

    def forward_loss(self, inputs, yqs, target):
        out = self.forward(inputs)
        pred_vect = torch.sigmoid(out)  # [0, 1]区間にする
        assert tuple(pred_vect.shape) == (
            self.config.sequence_size,
            self.config.batch_size,
            self.config.n_skills,
        ), "Expected shape {}, but got {}".format(
            (self.config.sequence_size, self.config.batch_size, self.config.n_skills),
            pred_vect.shape,
        )
        # pred.shape: (20, 100, 124); (seqlen, batch_size, skill_size)
        # yqs.shape: (20, 100, 124); (seqlen, batch_size, skill_size)
        pred_prob = torch.max(pred_vect * yqs, 2)[0]
        # print(target, target.shape)  # (20, 100)
        # TODO: 最後の1個だけじゃなくて、その他も損失関数に利用したら？
        loss = self._loss(pred_prob, target)
        # print(loss, loss.shape) #=> scalar, []

        out_dic = {
            "loss": loss,
            "pred_vect": pred_vect,  # (20, 100, 124)
            "pred_prob": pred_prob,  # (20, 100)
        }

        if self.config.waviness_l1 == True:
            waviness_norm_l1 = torch.abs(pred_vect[1:, :, :] - pred_vect[:-1, :, :])
            waviness_l1 = torch.sum(waviness_norm_l1) / (
                (pred_vect.shape[0] - 1) * pred_vect.shape[1] * pred_vect.shape[2]
            )
            lambda_l1 = self.config.lambda_l1
            out_dic["loss"] += lambda_l1 * waviness_l1
            out_dic["waviness_l1"] = waviness_l1.item()

        if self.config.waviness_l2 == True:
            waviness_norm_l2 = torch.pow(pred_vect[1:, :, :] - pred_vect[:-1, :, :], 2)
            waviness_l2 = torch.sum(waviness_norm_l2) / (
                (pred_vect.shape[0] - 1) * pred_vect.shape[1] * pred_vect.shape[2]
            )
            lambda_l2 = self.config.lambda_l2
            out_dic["loss"] += lambda_l2 * waviness_l2
            out_dic["waviness_l2"] = waviness_l2.item()

        return out_dic

    def initHidden0(self):
        return torch.zeros(
            self.n_layers * self.directions, self.batch_size, self.hidden_size
        ).to(self.device)

    def initC0(self):
        return torch.zeros(
            self.n_layers * self.directions, self.batch_size, self.hidden_size
        ).to(self.device)

    def loss_batch(self, xseq, yseq, opt=None):
        """
        DataLoaderの１イテレーションから，
        適宜back propagationし，
        lossを返す．

        xseq: [(q, a), (q, a), ... ]
        yseq: [(q, a), (q, a), ... ]
        """
        # print(yq.shape) => [100, 124] = [batch_size, skill_size]
        # print(xseq.shape, yseq.shape) => [100, 20, 2], [100, 20, 2]
        # Convert to onehot; (12, 1) -> (0, 0, ..., 1, 0, ...)
        # https://pytorch.org/docs/master/nn.functional.html#one-hot
        skill_n = self.config.n_skills
        onehot_size = skill_n * 2 + 2
        device = self.device
        # inputs = torch.dot(xseq, torch.as_tensor([[1], [skill_n]]))
        inputs = torch.LongTensor(
            np.dot(xseq.cpu().numpy(), np.array([[1], [skill_n]]))
        ).to(
            device
        )  # -> (100, 20, 1)
        # inputs = inputs.squeeze()
        # inputs = F.one_hot(inputs, num_classes=onehot_size).float()
        yqs = torch.LongTensor(np.dot(yseq.cpu().numpy(), np.array([[1], [0]]))).to(
            device
        )  # -> (100, 20, 1)
        yqs = yqs.squeeze(2)
        yqs = F.one_hot(yqs, num_classes=skill_n).float()
        target = torch.Tensor(np.dot(yseq.cpu().numpy(), np.array([[0], [1]]))).to(
            device
        )  # -> (100, 20, 1)
        target = target.squeeze(2)
        # print(target, target.shape)
        # compressed_sensing = True
        # if compressed_sensing and onehot_size != self.input_size:
        #     inputs = torch.mm(
        #         inputs.contiguous().view(-1, onehot_size), self.cs_basis)
        #     # https://pytorch.org/docs/stable/nn.html?highlight=rnn#rnn
        #     # inputの説明を見ると、input of shape (seq_len, batch, input_size)　とある
        #     inputs = inputs.view(
        #         self.batch_size, self.config.sequence_size, self.input_size)
        inputs = inputs.permute(1, 0, 2)

        yqs = yqs.permute(1, 0, 2)
        target = target.permute(1, 0)
        # actual_q = yq
        # actual_a = ya

        out = self.forward_loss(inputs, yqs, target)
        loss = out["loss"]

        if opt:
            # バックプロバゲーション
            opt.zero_grad()
            loss.backward()
            opt.step()

        # hm_pred_ks = pred[-1].squeeze()

        return out
