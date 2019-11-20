import torch
import torch.nn as nn
import torch.optim as optim

import os
import sys
import random
import time
import datetime
import logging
from pprint import pprint
import configparser
import json
import pickle
from pathlib import Path
from typing import List, Tuple, Set, Dict  # noqa

import numpy as np
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt

from src.utils import sAsMinutes, timeSince
from src.config import get_option_fallback, Config
from src.save import save_model, save_log, save_hm_fig, save_learning_curve
from knowledge_tracing.trainer import Trainer

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


def main(configpath: Path):
    cp = configparser.ConfigParser()
    cp.read(str(configpath))
    section_list = cp.sections()
    common_opt = dict(cp['common']) if 'common' in section_list else dict()
    report_list = list()
    for section in section_list:
        if section == 'common':
            continue
        section_opt = dict(cp[section])
        default_dict = {
            'config_name': configpath.stem,
            'common_name': '',
            'section_name': common_opt.get('common_name', '') + section,

            'debug': False,
            'model_name': str,
            'load_model': '',
            'plot_heatmap': False,
            'plot_lc': False,
            'source_data': 'original_ASSISTmentsSkillBuilder0910',  # SOURCE_ASSIST0910_ORIG,
            'ks_loss': False,
            'extend_backward': 0,
            'extend_forward': 0,
            'epoch_size': 200,
            'sequence_size': 20,
            'lr': 0.05,
            'n_skills': 124,
            'cuda': True,

            'batch_size': 100,
        }
        config_dict = get_option_fallback(
            {**common_opt, **section_opt}, fallback=default_dict)
        projectdir = Path(os.path.dirname(os.path.realpath(__file__)))
        config = Config(config_dict, projectdir=projectdir)
        pprint(config.as_dict())

        report = run(config)
        report_list.append(report)
    print(report)
    if report is not None:
        with open(projectdir / 'output' / 'reports' / '{}result.json'.format(config._get_stem_name()), 'w') as f:
            json.dump(report_list, f)


def run(config):
    assert config.model_name in {'encdec', 'basernn', 'baselstm', 'seq2seq'}
    report = dict()
    report['model_fname'] = config.outfname

    trainer = Trainer(config)
    try:
        trainer.train_model()
    except KeyboardInterrupt as e:
        print(e)
    return None

    # # ========================
    # # Load trained model
    # # ========================
    # if config.load_model:
    #     model.load_state_dict(torch.load(config.load_model))
    #     model = model.to(dev)

    # # model is trained or loaded now.

    # if config.plot_heatmap:
    #     batch_size = 1
    #     # TODO: don't repeat yourself
    #     if config.model_name == 'encdec':
    #         model = EncDecDKT(
    #             INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT,
    #             OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT,
    #             N_SKILLS,
    #             dev).to(dev)
    #         loss_batch = get_loss_batch_encdec(
    #             config.extend_forward, ks_loss=config.ks_loss)
    #     elif config.model_name == 'seq2seq':
    #         model = get_Seq2Seq(
    #             onehot_size, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT,
    #             OUTPUT_DIM, DEC_EMB_DIM, DEC_DROPOUT, dev)
    #         loss_batch = get_loss_batch_seq2seq(
    #             config.extend_forward, ks_loss=config.ks_loss)
    #     elif config.model_name == 'basernn':
    #         model = BaseDKT(
    #             dev, config.model_name, n_input, n_hidden, n_output, n_layers, batch_size
    #         ).to(dev)
    #         loss_batch = get_loss_batch_basedkt(
    #             onehot_size, n_input, batch_size, config.sequence_size, dev)
    #     elif config.model_name == 'baselstm':
    #         model = BaseDKT(
    #             dev, config.model_name, n_input, n_hidden, n_output, n_layers, batch_size
    #         ).to(dev)
    #         loss_batch = get_loss_batch_basedkt(
    #             onehot_size, n_input, batch_size, config.sequence_size, dev)
    #     else:
    #         raise ValueError(f'model_name {config.model_name} is wrong')
    #     if config.load_model:
    #         model.load_state_dict(torch.load(config.load_model))
    #         model = model.to(dev)
    #     heat_dl = prepare_heatmap_data(
    #         config.source_data, config.model_name, n_skills, PRESERVED_TOKENS,
    #         min_n=3, max_n=config.sequence_size, batch_size=batch_size, device=dev, sliding_window=0,
    #         params={'extend_backward': config.extend_backward, 'extend_forward': config.extend_forward})
    #     loss_func = nn.BCELoss()
    #     opt = optim.SGD(model.parameters(), lr=config.lr)

    #     debug = False
    #     logging.basicConfig()
    #     logger = logging.getLogger('dkt log')
    #     logger.setLevel(logging.INFO)
    #     train_loss_list = []
    #     train_auc_list = []
    #     eval_loss_list = []
    #     eval_auc_list = []
    #     eval_recall_list = []
    #     eval_f1_list = []
    #     x = []

    #     with torch.no_grad():
    #         model.eval()
    #         # =====
    #         # HEATMAP
    #         # =====
    #         all_out_prob = []
    #         # ------------------ heatmap (eval) -----------------
    #         val_pred = []
    #         val_actual = []
    #         current_eval_loss = []
    #         yticklabels = set()
    #         xticklabels = []
    #         for args in heat_dl:
    #             loss_item, batch_n, pred, actu_q, actu, pred_ks, _, _ = loss_batch(
    #                 model, loss_func, *args, opt=None)
    #             # current_eval_loss.append(loss_item[-1])
    #             # print(pred.shape, actu.shape)
    #             # val_pred.append(pred[-1])
    #             # val_actual.append(actu[-1])
    #             yq = torch.max(actu_q.squeeze(), 0)[1].item()
    #             ya = int(actu.item())
    #             yticklabels.add(yq)
    #             xticklabels.append((yq, ya))

    #             # print(pred_ks.shape)
    #             assert len(pred_ks.shape) == 1, 'pred_ks dimention {}, expected 1'.format(
    #                 pred_ks.shape)
    #             assert pred_ks.shape[0] == n_skills
    #             all_out_prob.append(pred_ks.unsqueeze(0))

    #     _d = torch.cat(all_out_prob).transpose(0, 1)
    #     _d = _d.cpu().numpy()
    #     print(_d.shape)
    #     print(len(yticklabels), len(xticklabels))
    #     yticklabels = sorted(list(yticklabels))
    #     related_d = np.matrix([_d[x, :] for x in yticklabels])

    #     # Regular Heatmap
    #     # fig, ax = plt.subplots(figsize=(20, 10))
    #     # sns.heatmap(_d, ax=ax)

    #     fig, ax = plt.subplots(figsize=(20, 7))
    #     sns.heatmap(
    #         related_d, vmin=0, vmax=1, ax=ax,
    #         # cmap="Reds_r",
    #         xticklabels=['{}'.format(y) for y in xticklabels],
    #         yticklabels=['s{}'.format(x) for x in yticklabels],
    #     )
    #     xtick_dic = {s: i for i, s in enumerate(yticklabels)}
    #     # 正解
    #     sca_x = [t + 0.5 for t, qa in enumerate(xticklabels) if qa[1] == 1]
    #     sca_y = [xtick_dic[qa[0]] + 0.5 for t,
    #              qa in enumerate(xticklabels) if qa[1] == 1]
    #     ax.scatter(sca_x, sca_y, marker='o', s=100, color='white')
    #     # 不正解
    #     sca_x = [t + 0.5 for t, qa in enumerate(xticklabels) if qa[1] == 0]
    #     sca_y = [xtick_dic[qa[0]] + 0.5 for t,
    #              qa in enumerate(xticklabels) if qa[1] == 0]
    #     ax.scatter(sca_x, sca_y, marker='X', s=100, color='black')

    #     save_hm_fig(config, fig)

    # return report


if __name__ == '__main__':
    SEED = 0
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    config = sys.argv[1]
    config = Path(config)
    assert config.exists(), config
    main(config)
