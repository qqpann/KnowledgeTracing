from collections import defaultdict
from src.save import save_report


class Report:

    def __init__(self, config):
        self.config = config
        self._indicator = defaultdict(lambda: defaultdict(list))
        self._best = defaultdict(lambda: defaultdict(float))
        self.subname = 0

    def set_value(self, key, value):
        self._indicator[key][self.subname] = value

    def get_value(self, key):
        return self._indicator[key][self.subname]

    def append_value(self, key, value):
        self._indicator[key][self.subname].append(value)

    def __call__(self, key, value):
        self.append_value(key, value)

    def set_best(self, key, value):
        self._best[key][self.subname] = value

    def get_best(self, key):
        return self._best[key][self.subname]

    def as_dict(self):
        _report = {
            'config': self.config.as_dict(),
            'indicator': {k: dict(v) for k, v in self._indicator.items()},
            'best': {k: dict(v) for k, v in self._best.items()},
        }
        return _report

    def dump(self):
        save_report(self.config, self.as_dict())
