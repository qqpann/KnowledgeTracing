import json
import os
import shutil
import sys
import random
from pathlib import Path
from typing import Dict

import optuna
import torch
import numpy as np

from src.trainer import Trainer
from src.config import Config, get_option_fallback
from src.slack import slack_message, slack_is_available
from src.log import get_logger

logger = get_logger(__name__, "tmp.log")


def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def run(config_dict: Dict, trial: optuna.Trial):
    projectdir = Path(os.path.dirname(os.path.realpath(__file__)))
    config = Config(config_dict, projectdir=projectdir)
    if not slack_is_available():
        logger.warning("Slack message is not available.")
    logger.info("Starting Experiment: {}".format(config.exp_name))

    seed_everything()
    trainer = Trainer(config, trial)
    try:
        trainer.optimize()
    except KeyboardInterrupt as e:
        print(e)
    except Exception as e:
        slack_message("Exception: {}".format(e))
        raise e
    finally:
        trainer.dump_report()
    logger.info("All experiments done!")
    return trainer.best_score


def objective(trial: optuna.Trial):
    config = Path(sys.argv[1])
    assert config.exists()
    # Grid search
    if not config.name.endswith(".optuna.json"):
        return
    with open(config, "r") as f:
        grid: dict = json.load(f)
    exp = grid.copy()
    projectdir = Path(os.path.dirname(os.path.realpath(__file__)))
    with open(projectdir / "config/fallback.json", "r") as f:
        default_cfg = json.load(f)
    default_cfg["config_name"] = config.parent.name
    exp = get_option_fallback(exp, fallback=default_cfg)
    exp["exp_name"] = config.stem
    for key, val in grid.items():
        print(key, val)
        if key in {'dkt', 'eddkt'}:
            print(val)
            for k, v in val.items():
                if type(v) is not list:
                    continue
                if type(v[0]) == int:
                    suggestion = trial.suggest_int(k, v[0], v[1])
                    logger.info(f"suggest_int {suggestion} from low:{v[0]} high:{v[1]}")
                    exp[key][k] = suggestion
                if type(v[0]) == float:
                    suggestion = trial.suggest_loguniform(k, v[0], v[1])
                    logger.info(f"suggest_logiform {suggestion} from low:{v[0]} high:{v[1]}")
                    exp[key][k] = suggestion
            continue
        if type(val) != list:
            continue
        if type(val[0]) == int:
            suggestion = trial.suggest_int(key, val[0], val[1])
            logger.info(f"suggest_int {suggestion} from low:{val[0]} high:{val[1]}")
            exp[key] = suggestion
        if type(val[0]) == float:
            suggestion = trial.suggest_loguniform(key, val[0], val[1])
            logger.info(f"suggest_logiform {suggestion} from low:{val[0]} high:{val[1]}")
            exp[key] = suggestion
        # if type(val[0]) == float:
        #     exp[key] = trial.suggest_float(key, val[0], val[1])
    score = run(exp, trial)
    return score


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100, timeout=60*30)
    pruned_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
    ]
    complete_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # slack_message(
    #     "Start optimize\n{}".format("\n".join([str(c) for c in config.iterdir()]))
    # )
