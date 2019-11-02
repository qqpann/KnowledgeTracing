import configparser
from typing import Dict
from pathlib import Path
import datetime


def get_option_fallback(options: Dict, fallback: Dict):
    '''
    Returns a merged dict with fallback as default values.
    '''
    # Thx: https://thispointer.com/how-to-merge-two-or-more-dictionaries-in-python/
    updated = {**fallback, **options}
    for key in updated.keys():
        try:
            fallback[key]
        except KeyError:
            raise KeyError('key `{}` found, but is not in fallback.'.format(key))
        if type(fallback[key]) == type:
            # a required option
            fallback_type = fallback[key]
            try:
                # it must be specified in options, not in fallback
                value = fallback_type(options[key])
            except KeyError:
                raise KeyError('key `{}` is required'.format(key))
        else:
            fallback_type = type(fallback[key])
            # an option with default
            value = fallback_type(updated[key])
        updated[key] = value
    return updated


class BaseConfig(object):
    '''
    BaseConfig is designed not to be affected by specific model or experiment.
    It provides base functionalities, but not concrete ones.

    最小構成として、config.modelのようなアクセスをtrainのなかで利用できるようにする。
    そのために、まずはdictからclass objectのattributeに変換することで利便性を高める。
    '''

    def __init__(self, options: Dict):
        self._attr_list = list()
        for attr, value in options.items():
            setattr(self, attr, value)
            self._attr_list.append(attr)

    def as_dict(self):
        return {k: getattr(self, k) for k in self._attr_list}


class Config(BaseConfig):
    '''
    Specific functionalities.
    '''
    def __init__(self, options: Dict, projectdir: Path):
        super().__init__(options)
        self.projectpdir = projectdir
        self.outdir = projectdir / 'output' / 'results'
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.starttime = datetime.datetime.now().strftime('%Y%m%d-%H%M')

    def _get_stem_name(self):
        debug_prefix = 'debug' if self.debug else ''
        return '_'.join([debug_prefix, self.starttime, self.section_name, self.model_name])

    @property
    def resultsdir(self):
        resultdir = self.outdir / self._get_stem_name()
        resultdir.mkdir(parents=True, exist_ok=True)
        return resultdir

    @property
    def outfname(self):
        outfname = self.model_name
        outfname += f'eb{self.extend_backward}' if self.extend_backward else ''
        outfname += f'ef{self.extend_forward}' if self.extend_forward else ''
        outfname += f'ks' if self.ks_loss else ''
        return outfname
