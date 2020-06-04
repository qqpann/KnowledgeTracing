from abc import ABC, abstractmethod


class BaseKTModel(ABC):
    """ Base class for Knowledge Tracing Classes """

    @abstractmethod
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device

    @abstractmethod
    def forward(self, xseq, yseq, mask, opt=None):
        out_dic = {}
        return out_dic
