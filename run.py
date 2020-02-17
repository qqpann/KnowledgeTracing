import os
import sys
import json
import torch
import random
import numpy as np
from pathlib import Path

from src.config import get_option_fallback, Config
from src.slack import slack_message
from src.logging import get_logger
from knowledge_tracing.trainer import Trainer


logger = get_logger(__name__, 'tmp.log')


def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def run(configpath: Path):
    with open(configpath, 'r') as f:
        cfg = json.load(f)
    with open(configpath.parent.parent / 'fallback.json', 'r') as f:
        default_cfg = json.load(f)
    default_cfg['config_name'] = configpath.parent.name
    projectdir = Path(os.path.dirname(os.path.realpath(__file__)))

    config_dict = get_option_fallback(cfg, fallback=default_cfg)
    config_dict['exp_name'] = configpath.stem
    config = Config(config_dict, projectdir=projectdir)
    logger.info('Starting Experiment: {}'.format(config.exp_name))

    seed_everything()
    trainer = Trainer(config)
    if config.load_model:
        # trainer.evaluate_model()
        # trainer.evaluate_model_heatmap()
        logger.info('All evaluations done!?')
        return
    try:
        trainer.kfold()
    except KeyboardInterrupt as e:
        print(e)
    finally:
        trainer.dump_report()
    logger.info('All experiments done!')
    slack_message('All experiments done for {}'.format(configpath.stem))


if __name__ == '__main__':
    config = sys.argv[1]
    config = Path(config)
    assert config.exists(), config
    run(config)
