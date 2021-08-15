import os
import pickle
import sys
from pathlib import Path
from typing import Dict

import click
import matplotlib.pyplot as plt
import seaborn as sns

from src.path import get_exp_paths, get_report_path, load_json

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))



projectdir = Path(".")


def get_report(config_name: str, exp_name: str) -> Dict:
    report: Dict = load_json(get_report_path(projectdir, config_name, exp_name))
    return report


def get_ndcg_dict(config_name: str, exp_name: str) -> Dict:
    report = get_report(config_name, exp_name)
    ndcg_list = report["indicator"]["RPhard"]["all"]
    kc_dict = _get_kc_dict(config_name, exp_name)
    assert len(ndcg_list) == len(kc_dict)
    return {k: ndcg_list[v] for k, v in kc_dict.items()}


def _get_kc_dict(config_name: str, exp_name: str) -> Dict:
    report = get_report(config_name, exp_name)
    config = report["config"]
    dict_path = (
        projectdir
        / "data/input"
        / config["source_data"]
        / f"{config['source_data']}_dic.pickle"
    )
    assert dict_path.exists(), dict_path
    with open(dict_path, "rb") as f:
        kc_dict: Dict = pickle.load(f)
    return kc_dict


def ndcg_distplot(config_name: str, exp_name: str, bins=20):
    report = get_report(config_name, exp_name)
    ndcg_list = report["indicator"]["RPhard"]["all"]
    name = report["config"]["exp_name"]
    sns.distplot(ndcg_list, bins=bins, label=name, kde_kws={"clip": (0.0, 1.0)})
    plt.legend()
    plt.xlabel("NDCG distribution")
    plt.ylabel("frequency")
    plt.title(config_name)
    plt.show()


@click.command()
@click.option("--config-name", "config_name", default="")
@click.option("--exp-name", "exp_name", default="")
def main(config_name: str, exp_name: str):
    # ndcg_distplot(config_name, exp_name)
    kc_dict = _get_kc_dict(config_name, exp_name)
    ndcg_dict = get_ndcg_dict(config_name, exp_name)
    while True:
        cmd = input('Enter command. (q): quit; (h): display all KC; (<KC>): KC name to show ndcg \n > ')
        if cmd == 'q':
            break
        elif cmd == 'h':
            print(list(kc_dict.keys()))
            continue
        elif cmd in ndcg_dict.keys():
            print(f'{cmd} NDCG: {ndcg_dict[cmd]}')
            continue
        else:
            print('Input error. Available: q, h, <KC>')
            continue


if __name__ == "__main__":
    main()
