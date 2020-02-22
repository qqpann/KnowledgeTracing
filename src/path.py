import json
from pathlib import Path
from typing import List


def load_json(cfgjsonpath: Path) -> dict:
    with open(cfgjsonpath, 'r') as f:
        return json.load(f)


def load_rep_cfg(reportpath: Path) -> dict:
    report_dict = load_json(reportpath)
    return report_dict['config']


def get_exp_paths(projectdir: Path, config_name: str, ) -> List[Path]:
    assert projectdir.exists()
    configdir = projectdir / 'config' / config_name
    exp_paths = [expcfgjson for expcfgjson in configdir.glob('*.json')]
    return exp_paths


def get_report_path(projectdir: Path, expcfgpath: Path) -> Path:
    assert projectdir.exists()
    config_name = expcfgpath.parent.name
    exp_name = expcfgpath.stem
    outputdir = projectdir / 'output' / config_name / exp_name
    assert outputdir.exists()
    return sorted(outputdir.glob('report/*/*.json'))[-1]  # report.json from latest starttime


def get_best_model_paths(projectdir: Path, config_dict: dict) -> List[Path]:
    starttime = config_dict['starttime']
    config_name = config_dict['config_name']
    exp_name = config_dict['exp_name']
    cpdir = projectdir / 'output' / config_name / exp_name / 'checkpoints' / starttime
    return sorted(cpdir.glob('*best.model'))


if __name__ == '__main__':
    from pprint import pprint
    p = Path().cwd()
    pprint([get_best_model_path(p, load_rep_cfg(get_report_path(p, e))) for e in get_exp_paths(p, '20_0220_edm2020_asmt15')])
