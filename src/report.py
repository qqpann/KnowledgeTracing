from collections import defaultdict
from src.save import save_report


class Report:

    def __init__(self, config):
        self.config = config
        self._indicator = defaultdict(lambda: defaultdict(list))
        self._best = defaultdict(lambda: defaultdict(float))
        self.fold = 0

    def __call__(self, key, value):
        self._indicator[key][self.fold].append(value)

    def set_best(self, key, value):
        self._best[key][self.fold] = value

    def get_best(self, key):
        return self._best[key][self.fold]

    def as_dict(self):
        _report = {
            'config': self.config.as_dict(),
            'indicator': {k: dict(v) for k, v in self._indicator.items()},
            'best': {k: dict(v) for k, v in self._best.items()},
        }
        return _report

    def dump(self):
        save_report(self.config, self.as_dict())
