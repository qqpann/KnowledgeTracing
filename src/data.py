import os
import pickle
import logging
from math import log, ceil
from pathlib import Path
from typing import List, Tuple, Set, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold

import torch
from torch.utils.data import TensorDataset, Dataset, DataLoader, random_split


dirname = os.path.join(os.path.dirname(__file__), '../data')
# Self made from ASSISTments
# TODO: use collections.namedtuple
SOURCE_ASSIST0910_SELF = 'selfmade_ASSISTmentsSkillBuilder0910'
SOURCE_ASSIST0910_ORIG = 'original_ASSISTmentsSkillBuilder0910'  # Piech et al.

PAD = 0
SOS = 1

ASSIST2009, ASSIST2015, STATICS2011, SYNTHETIC = 'assist2009', 'assist2015', 'statics2011', 'synthetic'
PREPARED_SOURCES = {ASSIST2009, ASSIST2015, STATICS2011, SYNTHETIC}


def load_source(projectdir, name) -> List[List[Tuple[int]]]:
    if name in {ASSIST2009, ASSIST2015, STATICS2011, SYNTHETIC}:
        if name == ASSIST2009:
            sourcedir = projectdir / 'data/input/assist2009_updated'
            train = 'assist2009_updated_train.csv'
            test = 'assist2009_updated_test.csv'
        elif name == ASSIST2015:
            sourcedir = projectdir / 'data/input/assist2015'
            train = 'assist2015_train.csv'
            test = 'assist2015_test.csv'
        elif name == STATICS2011:
            sourcedir = projectdir / 'data/input/STATICS'
            train = 'STATICS_train.csv'
            test = 'STATICS_test.csv'
        elif name == SYNTHETIC:
            sourcedir = projectdir / 'data/input/synthetic'
            train = 'naive_c5_q50_s4000_v1_train.csv'
            test = 'naive_c5_q50_s4000_v1_test.csv'
        else:
            raise ValueError('name is wrong')
        train_data = load_qa_format_source(sourcedir / train)
        test_data = load_qa_format_source(sourcedir / test)
        return train_data + test_data

    if name == SOURCE_ASSIST0910_SELF:
        filename = os.path.join(
            dirname, 'input/skill_builder_data_corrected.pickle')
        with open(filename, 'rb') as f:
            data_dict = pickle.load(f)
            data = list(data_dict.values())
    elif name == SOURCE_ASSIST0910_ORIG:
        trainfname = os.path.join(dirname, 'input/builder_train.csv')
        testfname = os.path.join(dirname, 'input/builder_test.csv')
        data = load_qa_format_source(trainfname)
    else:
        filename = os.path.join(dirname, f'input/{name}.pickle')
        with open(filename, 'rb') as f:
            data_dict = pickle.load(f)
            data = list(data_dict.values())
    return data


def load_qa_format_source(filename: str) -> List[List[Tuple[int]]]:
    with open(filename, 'r') as f:
        lines = f.readlines()
    data = []
    for idx in range(0, len(lines), 3):
        qlist = list(map(int, lines[idx + 1].strip().rstrip(',').split(',')))
        alist = list(map(int, lines[idx + 2].strip().rstrip(',').split(',')))
        data.append([(q, a) for q, a in zip(qlist, alist)])
    return data


def slice_data_list(d: List, seq_size: int, pad=False):
    '''
    >>> d = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    >>> list(slice_data_list(d, seq_size=3))
    [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    >>> d = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    >>> list(slice_data_list(d, seq_size=3))
    [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    >>> list(slice_data_list(d, seq_size=3, pad=True))
    [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    >>> d = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    >>> list(slice_data_list(d, seq_size=3, pad=True))
    [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10]]
    '''
    max_iter = len(d) // seq_size
    if pad:
        max_iter += 1
    for i in range(0, max_iter):
        res = d[i * seq_size: i * seq_size + seq_size]
        if len(res) <= 1:
            return
        yield res


class DataHandler:
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.projectdir = self.config.projectdir
        self.folds = self.config.kfold

        # if config.source_data in {ASSIST2009, ASSIST2015, STATICS2011, SYNTHETIC}:
        #     pass
        # else:
        #     all_data = prepare_dataloader(self.config, self.device, self.config.pad)
        #     self.trainval_data, self.test_data = train_test_split(all_data, test_size=self.config.test_size)

    # def get_data(self):
    #     return load_source(self.config.source_data)

    def get_kfold(self, folds, random_state=None):
        return KFold(n_splits=folds, shuffle=True, random_state=random_state)

    @staticmethod
    def get_ds(config, device, data, data_idx):
        x_values = []
        y_values = []
        y_mask = []
        for idx in data_idx:
            d = data[idx]
            if len(d) < config.sequence_size + 1:
                continue
            # x and y seqsize is sequence_size + 1
            for xy_seq in slice_data_list(d, seq_size=config.sequence_size + 1, pad=config.pad):
                seq_actual_size = len(xy_seq)
                if config.pad == True and seq_actual_size < config.sequence_size+1:
                    xy_seq = xy_seq + [(0, 2)] * (config.sequence_size+1-seq_actual_size)
                x_values.append(xy_seq[:-1])
                y_values.append(xy_seq[1:])
                y_mask.append([True]*(seq_actual_size - 1) +
                              [False]*(config.sequence_size + 1 - seq_actual_size))

        all_ds = TensorDataset(
            torch.LongTensor(x_values).to(device),
            torch.LongTensor(y_values).to(device),
            torch.BoolTensor(y_mask).to(device),
        )
        return all_ds

    def get_test_dl(self):
        test_ds = self.get_ds(self.config, self.device, self.test_data, range(len(self.test_data)))
        test_dl = DataLoader(test_ds, batch_size=self.config.batch_size, drop_last=True)
        return test_dl

    def gen_trainval_dl(self):
        kf = self.get_kfold(self.folds)

        for train_idx, valid_idx in kf.split(self.trainval_data):
            train_ds = self.get_ds(self.config, self.device, self.trainval_data, train_idx)
            valid_ds = self.get_ds(self.config, self.device, self.trainval_data, valid_idx)

            train_dl = DataLoader(
                train_ds, batch_size=self.config.batch_size, drop_last=True)
            valid_dl = DataLoader(
                valid_ds, batch_size=self.config.batch_size, drop_last=True)
            yield train_dl, valid_dl

    def get_traintest_dl(self, projectdir: Path, name: str):
        if name == ASSIST2009:
            sourcedir = projectdir / 'data/input/assist2009_updated'
            train = 'assist2009_updated_train.csv'
            test = 'assist2009_updated_test.csv'
        elif name == ASSIST2015:
            sourcedir = projectdir / 'data/input/assist2015'
            train = 'assist2015_train.csv'
            test = 'assist2015_test.csv'
        elif name == STATICS2011:
            sourcedir = projectdir / 'data/input/STATICS'
            train = 'STATICS_train.csv'
            test = 'STATICS_test.csv'
        elif name == SYNTHETIC:
            sourcedir = projectdir / 'data/input/synthetic'
            train = 'naive_c5_q50_s4000_v1_train.csv'
            test = 'naive_c5_q50_s4000_v1_test.csv'
        else:
            raise ValueError('name is wrong')
        train_data = load_qa_format_source(sourcedir / train)
        test_data = load_qa_format_source(sourcedir / test)
        train_ds = self.get_ds(self.config, self.device, train_data, range(len(train_data)))
        test_ds = self.get_ds(self.config, self.device, test_data, range(len(test_data)))
        train_dl = DataLoader(
            train_ds, batch_size=self.config.batch_size, drop_last=True)
        test_dl = DataLoader(
            test_ds, batch_size=self.config.batch_size, drop_last=True)
        return train_dl, test_dl

    def generate_trainval_dl(self, projectdir: Path, name: str):
        if name == ASSIST2009:
            sourcedir = projectdir / 'data/input/assist2009_updated'
            train = 'assist2009_updated_train{}.csv'
            valid = 'assist2009_updated_valid{}.csv'
            kfold = 5
        elif name == ASSIST2015:
            sourcedir = projectdir / 'data/input/assist2015'
            train = 'assist2015_train{}.csv'
            valid = 'assist2015_valid{}.csv'
            kfold = 5
        elif name == STATICS2011:
            sourcedir = projectdir / 'data/input/STATICS'
            train = 'STATICS_train{}.csv'
            valid = 'STATICS_valid{}.csv'
            kfold = 5
        elif name == SYNTHETIC:
            sourcedir = projectdir / 'data/input/synthetic'
            train = 'naive_c5_q50_s4000_v1_train{}.csv'
            valid = 'naive_c5_q50_s4000_v1_valid{}.csv'
            kfold = 1
        else:
            raise ValueError('name is wrong')

        for i in range(1, kfold + 1):
            train_data = load_qa_format_source(sourcedir / train.format(i))
            valid_data = load_qa_format_source(sourcedir / valid.format(i))
            train_ds = self.get_ds(self.config, self.device, train_data, range(len(train_data)))
            valid_ds = self.get_ds(self.config, self.device, valid_data, range(len(valid_data)))
            train_dl = DataLoader(
                train_ds, batch_size=self.config.batch_size, drop_last=True)
            valid_dl = DataLoader(
                valid_ds, batch_size=self.config.batch_size, drop_last=True)
            yield train_dl, valid_dl


# load_source->list
# train, valid, test split based on index
#


def prepare_dataloader(config, device, pad=False):
    '''
    '''
    data = load_source(
        config.source_data)  # -> List[List[Tuple[int]]]; [[(12,1), (13,0), ...], ...]

    train_num = int(len(data) * .8)
    train_data, eval_data = random_split(
        data, [train_num, len(data) - train_num])

    sequence_size = config.sequence_size

    def get_ds(data):
        x_values = []
        y_values = []
        y_mask = []
        for d in data:
            if len(d) < sequence_size + 1:
                continue
            # x and y seqsize is sequence_size + 1
            for xy_seq in slice_data_list(d, seq_size=sequence_size + 1, pad=pad):
                seq_actual_size = len(xy_seq)
                if pad == True and seq_actual_size < sequence_size+1:
                    xy_seq = xy_seq + [(0, 2)] * \
                        (sequence_size+1-seq_actual_size)
                x_values.append(xy_seq[:-1])
                y_values.append(xy_seq[1:])
                y_mask.append([True]*(seq_actual_size - 1) +
                              [False]*(sequence_size + 1 - seq_actual_size))

        all_ds = TensorDataset(
            torch.LongTensor(x_values).to(device),
            torch.LongTensor(y_values).to(device),
            torch.BoolTensor(y_mask).to(device),
        )
        return all_ds

    train_ds = get_ds(train_data)
    eval_ds = get_ds(eval_data)

    # all_dl = DataLoader(all_ds, batch_size=batch_size, drop_last=True)
    train_dl = DataLoader(
        train_ds, batch_size=config.batch_size, drop_last=True)
    eval_dl = DataLoader(eval_ds, batch_size=config.batch_size, drop_last=True)
    return train_dl, eval_dl


def prepare_dummy_dataloader(config, seq_size, batch_size, device):
    # TODO: do not load_source twice just for dummy
    # -> List[List[Tuple[int]]]; [[(12,1), (13,0), ...], ...]
    data = load_source(config.projectdir, config.source_data)
    knowledge_concepts_set = set()
    for seq in data:
        for q, a in seq:
            knowledge_concepts_set.add(q)
    # assert config.n_skills == len(knowledge_concepts_set), 'KC size asserted to be {}, got {}'.format(
    #     config.n_skills, len(knowledge_concepts_set))
    # TODO: change to warn?

    x_values = []
    y_values = []
    y_mask = []
    for v in knowledge_concepts_set:
        # wrong
        x_values.append([(v, 0) for _ in range(seq_size)])
        y_values.append([(v, 0) for _ in range(seq_size)])
        y_mask.append([True] * seq_size)
        # correct
        x_values.append([(v, 1) for _ in range(seq_size)])
        y_values.append([(v, 1) for _ in range(seq_size)])
        y_mask.append([True] * seq_size)
    dummy_ds = TensorDataset(
        torch.LongTensor(x_values).to(device),
        torch.LongTensor(y_values).to(device),
        torch.BoolTensor(y_mask).to(device),
    )
    dummy_dl = DataLoader(dummy_ds, batch_size=batch_size, drop_last=True)
    return dummy_dl


def slide_d(d: List, seq_size: int) -> List[List]:
    '''
    >>> d = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    >>> list(slide_d(d, seq_size=4))
    [[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7], [5, 6, 7, 8]]
    '''
    max_iter = len(d) - seq_size + 1
    for i in range(0, max_iter):
        yield d[i: i + seq_size]


def prepare_heatmap_dataloader(config, seq_size, batch_size, device):
    SEED = 42
    # random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # -> List[List[Tuple[int]]]; [[(12,1), (13,0), ...], ...]
    data = load_source(config.source_data)

    sequence_size = config.sequence_size

    train_num = int(len(data) * .8)
    train_data, eval_data = random_split(
        data, [train_num, len(data) - train_num])

    x_values = []
    y_values = []
    for uid, d in enumerate(eval_data):
        if len(d) < sequence_size + 1 or len(d) < 80 or uid < 100:
            continue
        # x and y seqsize is sequence_size + 1
        # NOTE: for heatmap, use SLIDE_d to get continuous result.
        for xy_seq in slide_d(d, seq_size=sequence_size + 1):
            x_values.append(xy_seq[:-1])
            y_values.append(xy_seq[1:])
        break

    eval_ds = TensorDataset(
        torch.LongTensor(x_values).to(device),
        torch.LongTensor(y_values).to(device),
    )

    eval_dl = DataLoader(eval_ds, batch_size=config.batch_size, drop_last=True)
    return uid, eval_dl


if __name__ == '__main__':
    import doctest
    doctest.testmod()
