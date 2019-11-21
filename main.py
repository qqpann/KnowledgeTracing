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
    with open(configpath, 'r') as f:
        cfg = json.load(f)
    with open(configpath.parent / 'fallback.json', 'r') as f:
        default_cfg = json.load(f)
    default_cfg['config_name'] = configpath.stem
    projectdir = Path(os.path.dirname(os.path.realpath(__file__)))
    experiments = cfg['experiments']
    cmn_dict = cfg.get('common', dict())
    cmn_dict = get_option_fallback(cmn_dict, fallback=default_cfg)
    for exp_dict in experiments:
        config_dict = get_option_fallback(exp_dict, fallback=cmn_dict)
        config = Config(config_dict, projectdir=projectdir)
        pprint(config.as_dict())

        report = run(config)
    # print(report)
    # if report is not None:
    #     with open(projectdir / 'output' / 'reports' / '{}result.json'.format(config._get_stem_name()), 'w') as f:
    #         json.dump(report_list, f)


def run(config):
    assert config.model_name in {'encdec', 'basernn', 'baselstm', 'seq2seq'}
    # report = dict()
    # report['model_fname'] = config.outfname

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
