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
