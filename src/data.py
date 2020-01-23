import os
import pickle
from math import log, ceil
from typing import List, Tuple, Set, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import TensorDataset, Dataset, DataLoader, random_split


dirname = os.path.join(os.path.dirname(__file__), '../data')
SOURCE_ASSIST0910_SELF = 'selfmade_ASSISTmentsSkillBuilder0910'  # Self made from ASSISTments
SOURCE_ASSIST0910_ORIG = 'original_ASSISTmentsSkillBuilder0910'  # Piech et al.

PAD = 0
SOS = 1


def load_source(source) -> List[List[Tuple[int]]]:
    if source == SOURCE_ASSIST0910_SELF:
        filename = os.path.join(dirname, 'input/skill_builder_data_corrected.pickle')
        with open(filename, 'rb') as f:
            data_dict = pickle.load(f)
            data = list(data_dict.values())
    elif source == SOURCE_ASSIST0910_ORIG:
        trainfname = os.path.join(dirname, 'input/builder_train.csv')
        testfname = os.path.join(dirname, 'input/builder_test.csv')
        with open(trainfname, 'r') as f:
            data = []
            lines = f.readlines()
            for idx in range(0, len(lines), 3):
                qlist = list(map(int, lines[idx + 1].strip().rstrip(',').split(',')))
                alist = list(map(int, lines[idx + 2].strip().rstrip(',').split(',')))
                data.append([(q, a) for q, a in zip(qlist, alist)])
    else:
        filename = os.path.join(dirname, f'input/{source}.pickle')
        with open(filename, 'rb') as f:
            data_dict = pickle.load(f)
            data = list(data_dict.values())
    return data


class QandAEmbedder:
    def __init__(self, M: int, sequence_size: int, preserved_tokens:int=2):
        self.M = M
        self.sequence_size = sequence_size
        self.PT = preserved_tokens
        self.PAD_IDX = 0
        self.SOS_IDX = 1
        
    def qaToIdxNum(self, q_and_a: Tuple) -> int:
        '''
        qaToIdxNum(q_ant_a)
        
        Docs
        ----
        (q, a)のタプルを2M+PT長のインデックスに変換する（PTは0:pad, 1:sos）

        Examples
        --------
        >>> emb = QandAEmbedder(M=3, sequence_size=4)
        >>> emb.qaToIdxNum((0, 0))
        2
        >>> emb.qaToIdxNum((2, 1))
        7
        >>> emb.qaToIdxNum((3, 0))
        Traceback (most recent call last):
            ...
        ValueError: q out of range. Got (3, 0) but expected 0<=q<M=3
        >>> emb.qaToIdxNum((0, -1))
        Traceback (most recent call last):
            ...
        ValueError: a out of range
        '''
        if not (0 <= q_and_a[0] < self.M):
            raise ValueError('q out of range. Got {0} but expected 0<=q<M={1}'.format(q_and_a, self.M))
        elif (q_and_a[1] not in {0, 1}):
            raise ValueError('a out of range')
            
        # 0: PAD, 1: SOS
        return q_and_a[0] + q_and_a[1] * self.M + self.PT  # consider 0:pad and 1:sos

    def idxToOneHot(self, idx: int) -> np.array:  # idxnum should already considered 0
        '''
        >>> emb = QandAEmbedder(M=3, sequence_size=4)
        >>> emb.idxToOneHot(0)
        array([1., 0., 0., 0., 0., 0., 0., 0.])
        >>> emb.idxToOneHot(6)
        array([0., 0., 0., 0., 0., 0., 1., 0.])
        >>> emb.idxToOneHot(8)
        Traceback (most recent call last):
            ...
        ValueError: idx out of range
        '''
        if not (0 <= idx < 2 * self.M + self.PT):
            raise ValueError('idx out of range')
        onehot = np.zeros(2 * self.M + self.PT)  # consider 0:pad and 1:sos
        onehot[idx] = 1
        return onehot

    def qaToOnehot(self, q_and_a: Tuple) -> np.array:
        ''' To 2M + PreservedNum Size Onehot '''
        idx = self.qaToIdxNum(q_and_a)
        onehot = self.idxToOneHot(idx)
        return onehot

    def sequenceToOnehot(self, sequence_qa: List) -> List[np.array]:
        length = len(sequence_qa)
        sequence = [self.qaToIdxNum(qa) for qa in sequence_qa] + \
            [self.PAD_IDX] * (self.sequence_size - length)
        onehotSeq = [self.idxToOneHot(idx) for idx in sequence]
        return onehotSeq
    
    def qaToDeltaQ(self, qa:Tuple):
        '''
        >>> emb = QandAEmbedder(M=3, sequence_size=4)
        >>> emb.qaToDeltaQ((2, 1))
        array([0., 0., 1.])
        '''
        delta_q = np.zeros(self.M)
        delta_q[qa[0]] = 1
        return delta_q
        
    def qaToDeltaQandA(self, qa: Tuple) -> np.array:
        ''' To skill size onehot '''
        delta_q = self.qaToDeltaQ(qa)
        a = qa[1]
        return delta_q, a

    def sequenceToDeltaQandA(self, sequence_qa:List) -> Tuple:
        delta_qs, a_s = [], []
        for qa in sequence_qa:
            delta_q, a = self.qaToDeltaQandA(qa)
            delta_qs.append(delta_q)
            a_s.append(a)
        return delta_qs, a_s
    
    def sequenceToProbSeq(self, sequence_qa:List) -> np.array:
        '''
        >>> emb = QandAEmbedder(M=3, sequence_size=4)
        >>> emb.sequenceToProbSeq([(1, 1), (2, 0)])
        array([[0.5, 1. , 0.5],
               [0.5, 1. , 0. ]])
        '''
        base = np.ones(shape=(len(sequence_qa), self.M)) / 2
        for i, qa in enumerate(sequence_qa):
            if i != 0:
                base[i] = base[i-1]
            base[i, qa[0]] = qa[1]
        return base

    
def slice_d(d: List, x_seq_size:int, type:str='base', sliding_window:int=0, reverse:bool=False, extend_backward:int=0, extend_forward:int=0) -> Tuple[List]:
    '''
    Params
    ------
    d:
        list. data sequence.
        
    x_seq_size:
        int.
        
    sliding_window:
        int. if 1, only get 1 result for 1 user.
        if 0, get all for a user.
    
    generative:
        bool. if True, x and y have same length and do not overlap.
        
    reversed:
        bool. if True, get data from latest ones.
    
    Example
    -------
    >>> d = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    >>> slice_d(d, x_seq_size=2, type='base', sliding_window=1, reverse=False)
    [[0, 1, 2]]
    >>> slice_d(d, x_seq_size=2, type='base', sliding_window=2, reverse=False)
    [[0, 1, 2], [3, 4, 5]]
    >>> slice_d(d, x_seq_size=2, type='base', sliding_window=0, reverse=False)
    [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    >>> slice_d(d, x_seq_size=2, type='base', sliding_window=9, reverse=False)
    [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    
    >>> slice_d(d, x_seq_size=2, type='encdec', sliding_window=1, reverse=False)
    [[0, 1, 2, 3]]
    >>> slice_d(d, x_seq_size=2, type='encdec', sliding_window=2, reverse=False)
    [[0, 1, 2, 3], [4, 5, 6, 7]]
    >>> slice_d(d, x_seq_size=2, type='encdec', sliding_window=0, reverse=False)
    [[0, 1, 2, 3], [4, 5, 6, 7]]
    >>> slice_d(d, x_seq_size=2, type='encdec', sliding_window=9, reverse=False)
    [[0, 1, 2, 3], [4, 5, 6, 7]]
    
    >>> slice_d(d, x_seq_size=2, type='base', sliding_window=1, reverse=True)
    [[6, 7, 8]]
    >>> slice_d(d, x_seq_size=2, type='base', sliding_window=2, reverse=True)
    [[6, 7, 8], [3, 4, 5]]
    >>> slice_d(d, x_seq_size=2, type='base', sliding_window=0, reverse=True)
    [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    >>> slice_d(d, x_seq_size=2, type='base', sliding_window=9, reverse=True)
    [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    
    >>> slice_d(d, x_seq_size=2, type='generative', sliding_window=1, reverse=False)
    [[0, 1, 2, 3]]
    >>> slice_d(d, x_seq_size=2, type='generative', sliding_window=0, reverse=False)
    [[0, 1, 2, 3], [4, 5, 6, 7]]
    >>> slice_d(d, x_seq_size=2, type='generative', sliding_window=1, reverse=True)
    [[5, 6, 7, 8]]
    >>> slice_d(d, x_seq_size=2, type='generative', sliding_window=0, reverse=True)
    [[5, 6, 7, 8], [1, 2, 3, 4]]
    
    # >>> d = [0, 1, 2]
    # >>> slice_d(d, x_seq_size=5, type='base', sliding_window=1, reverse=False)
    # [[0, 1, 2, ?, ?]]
    
    >>> d = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    >>> slice_d(d, x_seq_size=2, type='encdec', sliding_window=1, reverse=False, extend_backward=1, extend_forward=1)
    [[0, 1, 2, 3, 4]]
    '''
    result = []
    if type[:4] == 'base':
        seq_size = x_seq_size + 1
    elif type == 'generative':
        seq_size = x_seq_size + x_seq_size
    elif type in {'encdec', 'seq2seq'}:
        seq_size = x_seq_size + 2 + extend_forward
    else:
        raise ValueError
    max_iter = len(d) // seq_size
    assert sliding_window >= 0 and sliding_window % 1 == 0
    max_iter = min(max_iter, sliding_window) if sliding_window != 0 else max_iter
    prefix = 0 if not reverse else len(d) - seq_size
    direction = 1 if not reverse else -1
    for i in range(0, direction * max_iter, direction * 1):
        result.append(d[prefix + i * seq_size : prefix + i * seq_size + seq_size])
    return result 
        

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
        res = d[i * seq_size : i * seq_size + seq_size]
        if len(res) <= 1:
            return
        yield res


def split_encdec(xsty_seq:List, extend_backward=0, extend_forward=0) -> (List, List, List):
    '''
    Split x_src, x_trg, y

    eb co-exist with x_src, but ef does not.
    
    Examples
    --------
    >>> xsty_seq = [2, 3, 4, 5, 6, 7]
    >>> split_encdec(xsty_seq)
    ([2, 3, 4, 5], [6], [7])
    >>> split_encdec(xsty_seq, 1, 0)
    ([2, 3, 4, 5], [5, 6], [6, 7])
    >>> split_encdec(xsty_seq, 0, 1)
    ([2, 3, 4], [5, 6], [6, 7])
    '''
    x_src = xsty_seq[:-2 - extend_forward]
    x_trg = xsty_seq[-2 - extend_backward - extend_forward:-1]
    y = xsty_seq[-1 - extend_backward - extend_forward:]
    return x_src, x_trg, y
    

def prepare_dataloader(config, device, pad=False):
    '''
    '''
    data = load_source(config.source_data)  # -> List[List[Tuple[int]]]; [[(12,1), (13,0), ...], ...]

    M = config.n_skills
    sequence_size = config.sequence_size
    N = ceil(log(2 * M))
    
    qa_emb = QandAEmbedder(M, sequence_size)

    train_num = int(len(data) * .8)
    train_data, eval_data = random_split(data, [train_num, len(data) - train_num])

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
                    xy_seq = xy_seq + [(0, 2)]*(sequence_size+1-seq_actual_size)
                x_values.append(xy_seq[:-1])
                y_values.append(xy_seq[1:])
                y_mask.append([True]*(seq_actual_size - 1) + [False]*(sequence_size + 1 - seq_actual_size))

        all_ds = TensorDataset(
            torch.LongTensor(x_values).to(device), 
            torch.LongTensor(y_values).to(device), 
            torch.BoolTensor(y_mask).to(device),
        )
        return all_ds
    
    train_ds = get_ds(train_data)
    eval_ds = get_ds(eval_data)

    # all_dl = DataLoader(all_ds, batch_size=batch_size, drop_last=True)
    train_dl = DataLoader(train_ds, batch_size=config.batch_size, drop_last=True)
    eval_dl = DataLoader(eval_ds, batch_size=config.batch_size, drop_last=True)
    return train_dl, eval_dl


def prepare_dummy_dataloader(config, seq_size, batch_size, device):
    # TODO: do not load_source twice just for dummy
    data = load_source(config.source_data)  # -> List[List[Tuple[int]]]; [[(12,1), (13,0), ...], ...]
    knowledge_concepts_set = set()
    for seq in data:
        for q, a in seq:
            knowledge_concepts_set.add(q)
    # assert config.n_skills == len(knowledge_concepts_set), 'KC size asserted to be {}, got {}'.format(
    #     config.n_skills, len(knowledge_concepts_set))
    # TODO: change to warn?

    x_values = []
    y_values = []
    for v in knowledge_concepts_set:
        # wrong
        x_values.append([(v, 0) for _ in range(seq_size)])
        y_values.append([(v, 0) for _ in range(seq_size)])
        # correct
        x_values.append([(v, 1) for _ in range(seq_size)])
        y_values.append([(v, 1) for _ in range(seq_size)])
    dummy_ds = TensorDataset(
        torch.LongTensor(x_values).to(device), 
        torch.LongTensor(y_values).to(device), 
    )
    dummy_dl = DataLoader(dummy_ds, batch_size=batch_size, drop_last=True)
    return dummy_dl


def slide_d(d: List, seq_size:int) -> List[List]:
    '''
    >>> d = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    >>> list(slide_d(d, seq_size=4))
    [[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7], [5, 6, 7, 8]]
    '''
    max_iter = len(d) - seq_size + 1
    for i in range(0, max_iter):
        yield d[i : i + seq_size]


def prepare_heatmap_dataloader(config, seq_size, batch_size, device):
    SEED = 42
    # random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    data = load_source(config.source_data)  # -> List[List[Tuple[int]]]; [[(12,1), (13,0), ...], ...]

    M = config.n_skills
    sequence_size = config.sequence_size
    N = ceil(log(2 * M))
    
    qa_emb = QandAEmbedder(M, sequence_size)

    train_num = int(len(data) * .8)
    train_data, eval_data = random_split(data, [train_num, len(data) - train_num])

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
