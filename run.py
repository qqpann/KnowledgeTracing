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

def check_prev_report(config_dict, projectdir):
    config_name, exp_name = config_dict['config_name'], config_dict['exp_name']
    reportdir = projectdir / 'output' / config_name / exp_name / 'report'
    checkpointdir = projectdir / 'output' / config_name / exp_name / 'checkpoints'
    if not reportdir.exists() or not checkpointdir.exists():
        return None
    report_path = sorted(reportdir.glob('*/*.json'))[-1]
    checkpoint_path = sorted(checkpointdir.glob('*/*.model'))[-1]
    return report_path, checkpoint_path


def run(configpath: Path):
    with open(configpath, 'r') as f:
        cfg = json.load(f)
    with open(configpath.parent.parent / 'fallback.json', 'r') as f:
        default_cfg = json.load(f)
    default_cfg['config_name'] = configpath.parent.name
    projectdir = Path(os.path.dirname(os.path.realpath(__file__)))

    config_dict = get_option_fallback(cfg, fallback=default_cfg)
    config_dict['exp_name'] = configpath.stem
    if not config_dict['overwrite'] and check_prev_report(config_dict, projectdir):
        report_path, checkpoint_path = check_prev_report(config_dict, projectdir)
        with open(report_path, 'r') as f:
            report_dict = json.load(f)
        config_dict = report_dict['config']
        config_dict['load_model'] = str(checkpoint_path)
    config = Config(config_dict, projectdir=projectdir)
    logger.info('Starting Experiment: {}'.format(config.exp_name))

    seed_everything()
    trainer = Trainer(config)
    # if config.load_model:
    #     trainer.evaluate_model()
    #     # trainer.evaluate_model_heatmap()
    #     logger.info('All evaluations done!')
    #     return
    try:
        # trainer.kfold()
        trainer.cv()
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
