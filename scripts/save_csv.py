import os
import sys
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict

import click
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.path import load_json, get_report_path, get_exp_paths


projectdir = Path(".")


def get_report(config_name: str, exp_name: str) -> Dict:
    report: Dict = load_json(get_report_path(projectdir, config_name, exp_name))
    return report


def get_report_dir(config_name: str, exp_name: str) -> Path:
    outputdir = projectdir / "output" / config_name / exp_name
    assert outputdir.exists()
    reportdir: Path = sorted(outputdir.glob("report/*"))[-1]
    return reportdir


def get_ndcg_dict(config_name: str, exp_name: str, fold: str = "all") -> Dict:
    report = get_report(config_name, exp_name)
    ndcg_list = report["indicator"]["RPhard"][fold]
    kc_dict = _get_kc_dict(config_name, exp_name)
    assert len(ndcg_list) == len(kc_dict)
    return {k: ndcg_list[v] for k, v in kc_dict.items()}


def get_simu_dict(config_name: str, exp_name: str, fold: str = "all") -> Dict:
    report = get_report(config_name, exp_name)
    simu_dict = report["indicator"]["simu_pred"][fold]
    kc_dict = _get_kc_dict(config_name, exp_name)
    assert len(simu_dict) == len(kc_dict)
    return {k: simu_dict[str(v)] for k, v in kc_dict.items()}


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


@click.command()
@click.option("--config-name", "config_name", default="")
@click.option("--exp-name", "exp_name", default="")
def main(config_name: str, exp_name: str):
    # kc_dict = _get_kc_dict(config_name, exp_name)
    simu_dict = get_simu_dict(config_name, exp_name)
    ndcg_dict = get_ndcg_dict(config_name, exp_name)
    reportdir = get_report_dir(config_name, exp_name)
    assert reportdir.exists()
    data = defaultdict(list)
    for lo, (simu, pred) in simu_dict.items():
        data['LO'].append(lo)
        data['ndcg'].append(ndcg_dict[lo])
        for s, p in zip(simu, pred):
            data[f'pred_{s}'].append(p)
    df = pd.DataFrame(dict(data))
    df.to_csv(reportdir / "simu.csv")
    print("saved to", reportdir / "simu.csv")


if __name__ == "__main__":
    main()
