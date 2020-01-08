import configparser
from typing import Dict
from pathlib import Path
import datetime


def get_option_fallback(options: Dict, fallback: Dict, hard=False, depth=0):
    '''
    Returns a merged dict with fallback as default values.

    >>> flb = {'a': 1, 'b': 2, 'c': {'e': 4, 'f': 5}}
    >>> opt = {'a': 2, 'd': 3, 'c': {'e': 6, 'g': 7}}
    >>> get_option_fallback(opt, flb)
    {'a': 2, 'b': 2, 'c': {'e': 6, 'f': 5, 'g': 7}, 'd': 3}
    '''
    if depth > 2:
        raise 'Nest is too deep.'
    # Thx: https://thispointer.com/how-to-merge-two-or-more-dictionaries-in-python/
    updated = {**fallback, **options}
    for k, v in fallback.items():
        if isinstance(v, dict):
            _v = get_option_fallback(
                options.get(k, {}), fallback[k], depth=depth+1)
            updated[k] = _v
        if isinstance(v, list):
            raise 'Found {}: {} list, which is not supported'.format(k, v)
    return updated


class BaseConfig(object):
    '''
    BaseConfig is designed not to be affected by specific model or experiment.
    It provides base functionalities, but not concrete ones.

    最小構成として、config.modelのようなアクセスをtrainのなかで利用できるようにする。
    そのために、まずはdictからclass objectのattributeに変換することで利便性を高める。
    '''

    def __init__(self, options: Dict):
        self.options = options
        self._attr_list = list()
        for attr, value in options.items():
            setattr(self, attr, value)
            self._attr_list.append(attr)

    def as_dict(self):
        return {k: getattr(self, k) for k in self._attr_list}

    def get(self, *attr, **pattr):
        return self.options.get(*attr, **pattr)


class Config(BaseConfig):
    '''
    Specific functionalities.
    '''

    def __init__(self, options: Dict, projectdir: Path):
        super().__init__(options)
        self.projectpdir = projectdir
        self.outdir = projectdir / 'output'
        self.outdir.mkdir(exist_ok=True)
        self.starttime = datetime.datetime.now().strftime('%Y%m%d-%H%M')

    @property
    def resultsdir(self):
        resultdir = self.outdir / self.config_name / self.exp_name
        resultdir.mkdir(parents=True, exist_ok=True)
        return resultdir

    @property
    def load_model_path(self):
        if not self.load_model:
            return None
        load_model_path = self.projectpdir / self.load_model
        assert load_model_path.exists(), '{} not found'.format(load_model_path)
        return load_model_path

    @property
    def outfname(self):
        outfname = self.model_name
        outfname += f'eb{self.extend_backward}' if self.extend_backward else ''
        outfname += f'ef{self.extend_forward}' if self.extend_forward else ''
        outfname += f'ks' if self.ks_loss else ''
        return outfname


if __name__ == "__main__":
    import doctest
    failure_count, test_count = doctest.testmod()
    print('{} tests run / {} failures'.format(test_count, failure_count))
