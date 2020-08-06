import os
import json
from pathlib import Path
from typing import List

# /opt/ml
# |-- input
# | |-- config
# | | |-- hyperparameters.json
# | | |-- init-config.json
# | | |-- inputdataconfig.json
# | | |-- metric-definition-regex.json
# | | |-- resourceconfig.json
# | | |-- trainingjobconfig.json
# | | `-- upstreamoutputdataconfig.json
# | `-- data
# |   |-- {channel_name}
# |-- model
# `-- output |-- data |-- metrics | `-- sagemaker `-- profiler


class PathHandler:
    def __init__(self, project_path: str):
        is_sm = os.environ.get("ENV") == "sagemaker"
        self.is_sagemaker = is_sm
        self.codedir = Path(project_path)
        self.projectdir = Path(project_path)
        self.outdir = self.projectdir / "output"
        self.inputdir = self.projectdir / "data/input"
        if self.is_sagemaker:
            self.projectdir = Path("/opt/ml")
            assert self.projectdir.exists(), FileNotFoundError(
                f"{self.projectdir} not found."
            )
            self.outdir = self.projectdir / "model"
            channel_name = "train"
            self.inputdir = self.projectdir / "input/data" / channel_name
        self.outdir.mkdir(exist_ok=True)


def load_json(cfgjsonpath: Path) -> dict:
    with open(cfgjsonpath, "r") as f:
        return json.load(f)


def load_rep_cfg(reportpath: Path) -> dict:
    report_dict = load_json(reportpath)
    return report_dict["config"]


def get_exp_paths(projectdir: Path, config_name: str,) -> List[Path]:
    assert projectdir.exists()
    configdir = projectdir / "config" / config_name
    exp_paths = [expcfgjson for expcfgjson in configdir.glob("*.json")]
    return exp_paths


def get_exp_names(projectdir: Path, config_name: str) -> List[str]:
    outputdir = projectdir / "output" / config_name
    if outputdir.exists():
        res = [exp_path.name for exp_path in outputdir.iterdir()]
        print(res)
        return res
    else:
        return None


def get_report_path(projectdir: Path, config_name: str, exp_name: str) -> Path:
    outputdir = projectdir / "output" / config_name / exp_name
    assert outputdir.exists(), FileNotFoundError(f"{outputdir} not found")
    # report.json from latest starttime
    return sorted(outputdir.glob("report/*/report.json"))[-1]


def get_report_paths(projectdir: Path, config_name: str) -> List[Path]:
    assert projectdir.exists()
    outputdir = projectdir / "output" / config_name
    assert outputdir.exists(), FileNotFoundError(f"{outputdir} not found")
    exp_paths = [expoutdir for expoutdir in outputdir.iterdir() if expoutdir.is_dir()]
    # report.json from latest starttime
    return [sorted(d.glob("report/*/*.json"))[-1] for d in exp_paths]


def get_best_model_paths(projectdir: Path, config_dict: dict) -> List[Path]:
    starttime = config_dict["starttime"]
    config_name = config_dict["config_name"]
    exp_name = config_dict["exp_name"]
    cpdir = projectdir / "output" / config_name / exp_name / "checkpoints" / starttime
    return sorted(cpdir.glob("*best.model"))


if __name__ == "__main__":
    from pprint import pprint

    p = Path().cwd()
    pprint(
        [
            get_best_model_path(p, load_rep_cfg(get_report_path(p, e)))
            for e in get_exp_paths(p, "20_0220_edm2020_asmt15")
        ]
    )
