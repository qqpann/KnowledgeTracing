import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

from knowledge_tracing.trainer import Trainer
from src.config import Config, get_option_fallback
from src.log import get_logger
from src.slack import slack_is_available, slack_message

logger = get_logger(__name__, "tmp.log")


def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def check_prev_report(config_dict, projectdir):
    config_name, exp_name = config_dict["config_name"], config_dict["exp_name"]
    reportdir = projectdir / "output" / config_name / exp_name / "report"
    checkpointdir = projectdir / "output" / config_name / exp_name / "checkpoints"
    if not reportdir.exists() or not checkpointdir.exists():
        return None
    report_path = sorted(reportdir.glob("*/*.json"))[-1]
    checkpoint_path = sorted(checkpointdir.glob("*/fall_final.model"))[-1]
    return report_path, checkpoint_path


def run(configpath: Path):
    projectdir = Path(os.path.dirname(os.path.realpath(__file__)))
    with open(configpath, "r") as f:
        cfg = json.load(f)
    with open(projectdir / "config/fallback.json", "r") as f:
        default_cfg = json.load(f)
    default_cfg["config_name"] = configpath.parent.name

    config_dict = get_option_fallback(cfg, fallback=default_cfg)
    config_dict["exp_name"] = configpath.stem
    if not config_dict["overwrite"] and check_prev_report(config_dict, projectdir):
        report_path, checkpoint_path = check_prev_report(config_dict, projectdir)
        with open(report_path, "r") as f:
            report_dict = json.load(f)
        config_dict = report_dict["config"]
        config_dict["load_model"] = str(checkpoint_path)
    config = Config(config_dict, projectdir=projectdir)
    if not slack_is_available():
        logger.warning("Slack message is not available.")
    logger.info("Starting Experiment: {}".format(config.exp_name))

    seed_everything()
    trainer = Trainer(config)
    if config.load_model:
        trainer.evaluate_model()
        logger.info('All evaluations done!')
        return
    try:
        trainer.cv()
    except KeyboardInterrupt as e:
        print(e)
    except Exception as e:
        slack_message("Exception: {}".format(e))
        raise e
    finally:
        trainer.dump_report()
    logger.info("All experiments done!")
    slack_message(
        "All experiments done for {}.\nBest: ```{}```".format(
            configpath.stem, str(trainer.report.as_dict()["best"])
        )
    )


if __name__ == "__main__":
    config = sys.argv[1]
    config = Path(config)
    assert config.exists(), config
    run(config)
