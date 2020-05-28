from abc import ABC, abstractmethod


class BaseKTModel(ABC):
    """ Base class for Knowledge Tracing Classes """

    @abstractmethod
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device

    @abstractmethod
    def forward(self, xseq, yseq, mask):
        out_dic = {}
        return out_dic

    @abstractmethod
    def loss_batch(self, xseq, yseq, mask, opt=None):
        out = self.forward(xseq, yseq, mask)
        loss = out["loss"]

        if opt:
            # バックプロバゲーション
            opt.zero_grad()
            loss.backward()
            opt.step()
        return out
