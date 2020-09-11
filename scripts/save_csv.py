import os
import sys
import pickle
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Optional, Union

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


def get_goodbad_dict(config_name: str, exp_name: str, fold: str = "all") -> Dict:
    report = get_report(config_name, exp_name)
    goodbad_list = report["indicator"]["RPsoft"][fold]["goodbad"]
    kc_dict = _get_kc_dict(config_name, exp_name)
    assert len(goodbad_list) == len(kc_dict)
    return {k: goodbad_list[v] for k, v in kc_dict.items()}


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


def cast_kc(kc_dict: Dict, target) -> Dict:
    assert len(target) == len(kc_dict)
    if type(target) is list:
        return {k: target[v] for k, v in kc_dict.items()}
    else:
        return {k: target[str(v)] for k, v in kc_dict.items()}


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
    report = load_json(get_report_path(projectdir, config_name, exp_name))
    kc_dict = _get_kc_dict(config_name, exp_name)
    reportdir = get_report_dir(config_name, exp_name)
    assert reportdir.exists(), f"reportdir {reportdir} not found."

    # NDCG, goodbad -> simu.csv
    simu_dict = get_simu_dict(config_name, exp_name)
    ndcg_dict = get_ndcg_dict(config_name, exp_name)
    goodbad_dict = get_goodbad_dict(config_name, exp_name)
    data = defaultdict(list)
    for lo, (simu, pred) in simu_dict.items():
        data["LO"].append(lo)
        data["ndcg"].append(ndcg_dict[lo])
        data["goodbad"].append(goodbad_dict[lo])
        for s, p in zip(simu, pred):
            data[f"pred_{s}"].append(p)
    df = pd.DataFrame(dict(data))
    df.to_csv(reportdir / "simu.csv")
    print("saved to", reportdir / "simu.csv")

    # NDCG oracle->fail, goodbad -> simu.csv
    simu_dict = cast_kc(kc_dict, report["indicator"]["simu_pred_oracle2fail"]["all"])
    ndcg_dict = cast_kc(kc_dict, report["indicator"]["RPhard_oracle2fail"]["all"])
    goodbad_dict = cast_kc(
        kc_dict, report["indicator"]["RPsoft_oracle2fail"]["all"]["goodbad"]
    )
    data = defaultdict(list)
    for lo, (simu, pred) in simu_dict.items():
        data["LO"].append(lo)
        data["ndcg"].append(ndcg_dict[lo])
        data["goodbad"].append(goodbad_dict[lo])
        for s, p in zip(simu, pred):
            data[f"pred_{s}"].append(p)
    df = pd.DataFrame(dict(data))
    df.to_csv(reportdir / "simu_oracle_to_fail.csv")
    print("saved to", reportdir / "simu_oracle_to_fail.csv")

    fold = "all"
    ip = report["indicator"]["inverted_performance"][fold]
    # ip: dic fold[0,all]
    # /lst KC[0, N]
    # /dic s(cor num included)[0,T]
    # /lst t(sequence time step)
    seq_len = len(ip[0]["0"])
    assert len(ip) == len(kc_dict)
    data = defaultdict(list)
    data_fail = defaultdict(list)
    for lo, kcip in cast_kc(kc_dict, ip).items():
        oracle = kcip[str(seq_len)]
        failing = kcip["0"]
        data["LO"].append(lo)
        data_fail["LO"].append(lo)
        for i, s in enumerate(oracle, start=1):
            data[f"seq{i}pad{seq_len-i}"].append(s)
        for i, s in enumerate(failing, start=1):
            data_fail[f"seq{i}pad{seq_len-i}"].append(s)
    df = pd.DataFrame(dict(data))
    df.to_csv(reportdir / "pad_pred_of_each_timestep.csv")
    print("saved to", reportdir / "pad_pred_of_each_timestep.csv")
    df = pd.DataFrame(dict(data_fail))
    df.to_csv(reportdir / "pad_pred_of_each_timestep_fail.csv")
    print("saved to", reportdir / "pad_pred_of_each_timestep_fail.csv")


if __name__ == "__main__":
    main()
