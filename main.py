import torch
import torch.nn as nn
import torch.optim as optim

import time
import datetime
import logging
import random
import configparser
from pprint import pprint
import pickle
from pathlib import Path
from math import log, ceil
from typing import List, Tuple, Set, Dict  # noqa

import click
import numpy as np
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt

from src.data import prepare_data, prepare_heatmap_data, SOURCE_ASSIST0910_SELF, SOURCE_ASSIST0910_ORIG
from src.utils import sAsMinutes, timeSince
from src.config import get_option_fallback, Config
from model.eddkt import EncDecDKT, get_loss_batch_encdec
from model.basedkt import BaseDKT, get_loss_batch_basedkt
from model.seq2seq import get_Seq2Seq, get_loss_batch_seq2seq


def get_name_prefix(debug):
    debug = 'debug_' if debug else ''
    now = datetime.datetime.now().strftime('%Y%m%d%H%M')
    return debug + now


def save_model(model, name, auc, epoch, debug):
    prefix = get_name_prefix(debug)
    torch.save(model.state_dict(), f'models/{prefix}_{name}_{auc}.{epoch}')


def save_log(data, name, auc, epoch, debug):
    prefix = get_name_prefix(debug)
    with open(f'data/output/{prefix}_{name}_{auc}.{epoch}.pickle', 'wb') as f:
        pickle.dump(data, f)


def save_sns_fig(sns_fig, fname):
    prefix = get_name_prefix(debug=False)
    sns_fig.savefig(f'data/output/heatmap/{prefix}_{fname}.png')


def save_learning_curve(x, train_loss_list, train_auc_list, eval_loss_list, eval_auc_list, fname, debug):
    prefix = get_name_prefix(debug)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    if train_loss_list:
        ax.plot(x, train_loss_list, label='train loss')
    if train_auc_list:
        ax.plot(x, train_auc_list, label='train auc')
    if eval_loss_list:
        ax.plot(x, eval_loss_list, label='eval loss')
    ax.plot(x, eval_auc_list, label='eval auc')
    ax.legend()
    print(len(train_loss_list), len(eval_loss_list), len(eval_auc_list))
    plt.savefig(f'data/output/learning_curve/{prefix}_{fname}.png')


@click.command()
@click.option('--config', default='')
def main(config):
    if not config:
        print('Other options are depricated. Please use --config.')
        return
    cp = configparser.ConfigParser()
    cp.read(config)
    section_list = cp.sections()
    pprint(section_list)
    common_opt = dict(cp['common']) if 'common' in section_list else dict()
    for section in section_list:
        if section == 'common':
            continue
        section_opt = dict(cp[section])
        default_dict = {
            'debug': False,
            'model_name': str,
            'load_model': '',
            'plot_heatmap': False,
            'plot_lc': False,
            'source_data': SOURCE_ASSIST0910_ORIG,
            'ks_loss': False,
            'extend_backward': 0,
            'extend_forward': 0,
            'epoch_size': 200,
            'sequence_size': 20,
            'lr': 0.05,
            'n_skills': 124,
            'cuda': True,
        }
        config_dict = get_option_fallback({**common_opt, **section_opt}, fallback=default_dict)
        config = Config(config_dict)
        pprint(config.as_dict())

        train(config)


def train(config):
    assert config.model_name in {'encdec', 'basernn', 'baselstm', 'seq2seq'}
    # =========================
    # Outfile name
    # =========================
    model_fname = config.model_name
    model_fname += f'eb{config.extend_backward}' if config.extend_backward else ''
    model_fname += f'ef{config.extend_forward}' if config.extend_forward else ''
    model_fname += f'ks' if config.ks_loss else ''

    # =========================
    # Seed
    # =========================
    SEED = 0
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    # =========================
    # Version, Device
    # =========================
    print('PyTorch:', torch.__version__)
    dev = torch.device('cuda' if config.cuda and torch.cuda.is_available() else 'cpu')
    print('Using Device:', dev)

    # =========================
    # Logging
    # =========================
    logging.basicConfig()
    logger = logging.getLogger(config.model_name)
    logger.setLevel(logging.INFO)

    # =========================
    # Parameters
    # =========================
    batch_size, n_hidden, n_skills, n_layers = 100, 200, config.n_skills, 2
    n_output = n_skills
    PRESERVED_TOKENS = 2  # PAD, SOS
    onehot_size = 2 * n_skills + PRESERVED_TOKENS
    n_input = ceil(log(2 * n_skills))

    INPUT_DIM, ENC_EMB_DIM, ENC_DROPOUT = onehot_size, n_input, 0.6
    OUTPUT_DIM, DEC_EMB_DIM, DEC_DROPOUT = onehot_size, n_input, 0.6
    HID_DIM, N_LAYERS = n_hidden, n_layers
    N_SKILLS = n_skills
    # OUTPUT_DIM = n_output = 124  # TODO: ほんとはこれやりたい

    # =========================
    # Prepare models, LossBatch, and Data
    # =========================
    if config.model_name == 'encdec':
        model = EncDecDKT(
            INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT,
            OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT,
            N_SKILLS,
            dev).to(dev)
        loss_batch = get_loss_batch_encdec(config.extend_forward, ks_loss=config.ks_loss)
        train_dl, eval_dl = prepare_data(
            config.source_data, 'encdec', n_skills, PRESERVED_TOKENS,
            min_n=3, max_n=config.sequence_size, batch_size=batch_size, device=dev, sliding_window=0,
            params={'extend_backward': config.extend_backward, 'extend_forward': config.extend_forward})
    elif config.model_name == 'seq2seq':
        model = get_Seq2Seq(
            onehot_size, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT,
            OUTPUT_DIM, DEC_EMB_DIM, DEC_DROPOUT, dev)
        loss_batch = get_loss_batch_seq2seq(config.extend_forward, ks_loss=config.ks_loss)
        train_dl, eval_dl = prepare_data(
            config.source_data, 'encdec', n_skills, PRESERVED_TOKENS,
            min_n=3, max_n=config.sequence_size, batch_size=batch_size, device=dev, sliding_window=0,
            params={'extend_backward': config.extend_backward, 'extend_forward': config.extend_forward})

    elif config.model_name == 'basernn':
        model = BaseDKT(
            dev, config.model_name, n_input, n_hidden, n_output, n_layers, batch_size
        ).to(dev)
        loss_batch = get_loss_batch_basedkt(
            onehot_size, n_input, batch_size, config.sequence_size, dev)
        train_dl, eval_dl = prepare_data(
            config.source_data, 'base', n_skills, preserved_tokens='?',
            min_n=3, max_n=config.sequence_size, batch_size=batch_size, device=dev, sliding_window=0)
    elif config.model_name == 'baselstm':
        model = BaseDKT(
            dev, config.model_name, n_input, n_hidden, n_output, n_layers, batch_size
        ).to(dev)
        loss_batch = get_loss_batch_basedkt(
            onehot_size, n_input, batch_size, config.sequence_size, dev)
        train_dl, eval_dl = prepare_data(
            config.source_data, 'base', n_skills, preserved_tokens='?',
            min_n=3, max_n=config.sequence_size, batch_size=batch_size, device=dev, sliding_window=0)
    else:
        raise ValueError(f'model_name {config.model_name} is wrong')
    logger.log('train_dl.dataset size: {}'.format(len(train_dl.dataset)))
    logger.log('eval_dl.dataset size: {}'.format(len(eval_dl.dataset)))

    print(model)

    # ========================
    # Load trained model
    # ========================
    if config.load_model:
        model.load_state_dict(torch.load(config.load_model))
        model = model.to(dev)
    else:
        # -------------------------
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.log(
            f'The model has {count_parameters(model):,} trainable parameters')

        loss_func = nn.BCELoss()
        opt = optim.SGD(model.parameters(), lr=config.lr)

        # ==========================
        # Run!
        # ==========================
        train_loss_list = []
        train_auc_list = []
        eval_loss_list = []
        eval_auc_list = []
        eval_recall_list = []
        eval_f1_list = []
        x = []

        start_time = time.time()
        for epoch in range(1, config.epoch_size + 1):
            print_train = epoch % 10 == 0
            print_eval = epoch % 10 == 0
            print_auc = epoch % 10 == 0

            # =====
            # TRAIN
            # =====
            model.train()

            # ------------------ train -----------------
            val_pred = []
            val_actual = []
            current_epoch_train_loss = []
            for args in train_dl:
                loss_item, batch_n, pred, actu_q, actu, pred_ks, _, _ = loss_batch(
                    model, loss_func, *args, opt=opt)
                current_epoch_train_loss.append(loss_item)
                val_pred.append(pred)
                val_actual.append(actu)

                # stop at first batch if debug
                if config.debug:
                    break

            if print_train:
                loss = np.array(current_epoch_train_loss)
                if epoch % 100 == 0:
                    logger.log(logging.INFO,
                               'TRAIN Epoch: {} Loss: {}'.format(epoch, loss.mean()))
                train_loss_list.append(loss.mean())

                # # AUC, Recall, F1
                # # TRAINの場合、勾配があるから処理が必要
                # y = torch.cat(val_targ).cpu().detach().numpy()
                # pred = torch.cat(val_prob).cpu().detach().numpy()
                # # AUC
                # fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)
                # if epoch % 100 == 0:
                # logger.log(logging.INFO,
                #            'TRAIN Epoch: {} AUC: {}'.format(epoch, metrics.auc(fpr, tpr)))
                # train_auc_list.append(metrics.auc(fpr, tpr))
            # -----------------------------------

            # =====
            # EVAL
            # =====
            if print_eval:
                with torch.no_grad():
                    model.eval()

                    # ------------------ eval -----------------
                    val_pred = []
                    val_actual = []
                    current_eval_loss = []
                    for args in eval_dl:
                        loss_item, batch_n, pred, actu_q, actu, pred_ks, _, _ = loss_batch(
                            model, loss_func, *args, opt=None)
                        current_eval_loss.append(loss_item)
                        val_pred.append(pred)
                        val_actual.append(actu)

                        # stop at first batch if debug
                        if config.debug:
                            break

                    loss = np.array(current_eval_loss)
                    if epoch % 100 == 0:
                        logger.log(logging.INFO,
                                   'EVAL  Epoch: {} Loss: {}'.format(epoch,  loss.mean()))
                    eval_loss_list.append(loss.mean())

                    # AUC, Recall, F1
                    if print_auc:
                        # TODO: viewしない？　最後の1個で？
                        y = torch.cat(val_actual).view(-1).cpu()
                        pred = torch.cat(val_pred).view(-1).cpu()
                        # AUC
                        fpr, tpr, thresholds = metrics.roc_curve(
                            y, pred, pos_label=1)
                        if epoch % 100 == 0:
                            logger.log(logging.INFO,
                                       'EVAL  Epoch: {} AUC: {}'.format(epoch, metrics.auc(fpr, tpr)))
                        auc = metrics.auc(fpr, tpr)
                        eval_auc_list.append(auc)
                        if epoch % 100 == 0:
                            save_model(model, model_fname, auc, epoch, config.debug)
                            save_log(
                                (x, train_loss_list, train_auc_list,
                                 eval_loss_list, eval_auc_list),
                                model_fname, auc, epoch, config.debug
                            )

                    #     # Recall
                    #     logger.debug('EVAL  Epoch: {} Recall: {}'.format(epoch, metrics.recall_score(y, pred.round())))
                    #     # F1 score
                    #     logger.debug('EVAL  Epoch: {} F1 score: {}'.format(epoch, metrics.f1_score(y, pred.round())))
                    # -----------------------------------

            if epoch % 10 == 0:
                x.append(epoch)
                if epoch % 100 == 0:
                    logger.log(logging.INFO,
                               f'{timeSince(start_time, epoch / config.epoch_size)} ({epoch} {epoch / config.epoch_size * 100})')

        if config.plot_lc:
            fname = model_fname
            save_learning_curve(x, train_loss_list, train_auc_list,
                                eval_loss_list, eval_auc_list, fname, config.debug)

    # model is trained or loaded now.

    if config.plot_heatmap:
        batch_size = 1
        # TODO: don't repeat yourself
        if config.model_name == 'encdec':
            model = EncDecDKT(
                INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT,
                OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT,
                N_SKILLS,
                dev).to(dev)
            loss_batch = get_loss_batch_encdec(config.extend_forward, ks_loss=config.ks_loss)
        elif config.model_name == 'seq2seq':
            model = get_Seq2Seq(
                onehot_size, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT,
                OUTPUT_DIM, DEC_EMB_DIM, DEC_DROPOUT, dev)
            loss_batch = get_loss_batch_seq2seq(
                config.extend_forward, ks_loss=config.ks_loss)
        elif config.model_name == 'basernn':
            model = BaseDKT(
                dev, config.model_name, n_input, n_hidden, n_output, n_layers, batch_size
            ).to(dev)
            loss_batch = get_loss_batch_basedkt(
                onehot_size, n_input, batch_size, config.sequence_size, dev)
        elif config.model_name == 'baselstm':
            model = BaseDKT(
                dev, config.model_name, n_input, n_hidden, n_output, n_layers, batch_size
            ).to(dev)
            loss_batch = get_loss_batch_basedkt(
                onehot_size, n_input, batch_size, config.sequence_size, dev)
        else:
            raise ValueError(f'model_name {config.model_name} is wrong')
        if config.load_model:
            model.load_state_dict(torch.load(config.load_model))
            model = model.to(dev)
        heat_dl = prepare_heatmap_data(
            config.source_data, config.model_name, n_skills, PRESERVED_TOKENS,
            min_n=3, max_n=config.sequence_size, batch_size=batch_size, device=dev, sliding_window=0,
            params={'extend_backward': config.extend_backward, 'extend_forward': config.extend_forward})
        loss_func = nn.BCELoss()
        opt = optim.SGD(model.parameters(), lr=config.lr)

        debug = False
        logging.basicConfig()
        logger = logging.getLogger('dkt log')
        logger.setLevel(logging.INFO)
        train_loss_list = []
        train_auc_list = []
        eval_loss_list = []
        eval_auc_list = []
        eval_recall_list = []
        eval_f1_list = []
        x = []

        with torch.no_grad():
            model.eval()
            # =====
            # HEATMAP
            # =====
            all_out_prob = []
            # ------------------ heatmap (eval) -----------------
            val_pred = []
            val_actual = []
            current_eval_loss = []
            yticklabels = set()
            xticklabels = []
            for args in heat_dl:
                loss_item, batch_n, pred, actu_q, actu, pred_ks, _, _ = loss_batch(
                    model, loss_func, *args, opt=None)
                # current_eval_loss.append(loss_item[-1])
                # print(pred.shape, actu.shape)
                # val_pred.append(pred[-1])
                # val_actual.append(actu[-1])
                yq = torch.max(actu_q.squeeze(), 0)[1].item()
                ya = int(actu.item())
                yticklabels.add(yq)
                xticklabels.append((yq, ya))

                # print(pred_ks.shape)
                assert len(pred_ks.shape) == 1, 'pred_ks dimention {}, expected 1'.format(
                    pred_ks.shape)
                assert pred_ks.shape[0] == n_skills
                all_out_prob.append(pred_ks.unsqueeze(0))

            # loss = np.array(current_eval_loss)
            # logger.log(logging.INFO ,
            #            'EVAL Loss: {}'.format( loss.mean()))
            # eval_loss_list.append(loss.mean())

            # -----------------------------------

            # AUC, Recall, F1
            # y = torch.cat(val_actual).view(-1).cpu()  # TODO: viewしない？　最後の1個で？
            # pred = torch.cat(val_pred).view(-1).cpu()
            # AUC
            # fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)
            # eval_auc_list.append(metrics.auc(fpr, tpr))
        #     # Recall
        #     logger.debug('EVAL  Epoch: {} Recall: {}'.format(epoch, metrics.recall_score(y, pred.round())))
        #     # F1 score
        #     logger.debug('EVAL  Epoch: {} F1 score: {}'.format(epoch, metrics.f1_score(y, pred.round())))

        prefix = get_name_prefix(debug)
        _d = torch.cat(all_out_prob).transpose(0, 1)
        _d = _d.cpu().numpy()
        print(_d.shape)
        print(len(yticklabels), len(xticklabels))
        yticklabels = sorted(list(yticklabels))
        related_d = np.matrix([_d[x, :] for x in yticklabels])

        # Regular Heatmap
        # fig, ax = plt.subplots(figsize=(20, 10))
        # sns.heatmap(_d, ax=ax)

        fig, ax = plt.subplots(figsize=(20, 7))
        sns.heatmap(
            related_d, vmin=0, vmax=1, ax=ax,
            # cmap="Reds_r",
            xticklabels=['{}'.format(y) for y in xticklabels],
            yticklabels=['s{}'.format(x) for x in yticklabels],
        )
        xtick_dic = {s: i for i, s in enumerate(yticklabels)}
        # 正解
        sca_x = [t + 0.5 for t, qa in enumerate(xticklabels) if qa[1] == 1]
        sca_y = [xtick_dic[qa[0]] + 0.5 for t,
                 qa in enumerate(xticklabels) if qa[1] == 1]
        ax.scatter(sca_x, sca_y, marker='o', s=100, color='white')
        # 不正解
        sca_x = [t + 0.5 for t, qa in enumerate(xticklabels) if qa[1] == 0]
        sca_y = [xtick_dic[qa[0]] + 0.5 for t,
                 qa in enumerate(xticklabels) if qa[1] == 0]
        ax.scatter(sca_x, sca_y, marker='X', s=100, color='black')

        save_sns_fig(fig, model_fname)


if __name__ == '__main__':
    main()
