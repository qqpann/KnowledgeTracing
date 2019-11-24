import torch

import time
import logging
import numpy as np
from math import log, ceil
from sklearn import metrics
from collections import defaultdict

from src.data import prepare_data, prepare_dataloader
from src.save import save_model, save_log, save_hm_fig, save_learning_curve, save_pred_accu_relation
from src.utils import sAsMinutes, timeSince
from model.eddkt import EDDKT
from model.dkt import DKT
from model.seq2seq import get_Seq2Seq, get_loss_batch_seq2seq


class Trainer(object):

    def __init__(self, config):
        self.config = config
        self.logger = self.get_logger()
        self.device = self.get_device()
        self.model, self.train_dl, self.eval_dl = self.get_model()
        self.opt = self.get_opt(self.model)

    def get_logger(self):
        logging.basicConfig()
        logger = logging.getLogger(self.config.model_name)
        logger.setLevel(logging.INFO)
        return logger

    def get_device(self):
        self.logger.info('PyTorch: {}'.format(torch.__version__))
        device = torch.device(
            'cuda' if self.config.cuda and torch.cuda.is_available() else 'cpu')
        self.logger.info('Using Device: {}'.format(device))
        return device

    def get_model(self):
        if self.config.model_name == 'eddkt':
            model = EDDKT(self.config, self.device).to(self.device)
            train_dl, eval_dl = prepare_dataloader(
                self.config, device=self.device)
        elif self.config.model_name == 'dkt':
            model = DKT(self.config, self.device).to(self.device)
            train_dl, eval_dl = prepare_dataloader(
                self.config, device=self.device)
        else:
            raise ValueError(f'model_name {self.config.model_name} is wrong')
        self.logger.info(
            'train_dl.dataset size: {}'.format(len(train_dl.dataset)))
        self.logger.info(
            'eval_dl.dataset size: {}'.format(len(eval_dl.dataset)))

        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(
            f'The model has {count_parameters(model):,} trainable parameters')

        return model, train_dl, eval_dl

    def get_opt(self, model):
        opt = torch.optim.SGD(model.parameters(), lr=self.config.lr)
        return opt

    def train_model(self, validate=True):
        self.logger.info('Starting train')
        best = {
            'auc': 0.,
            'auc_epoch': 0,
        }
        x_list = []
        train_loss_list, train_auc_list = [], []
        eval_loss_list, eval_auc_list = [], []
        start_time = time.time()
        for epoch in range(1, self.config.epoch_size + 1):
            self.model.train()
            indicators = self.exec_core(self.train_dl, self.opt)
            t_loss, t_auc = indicators['loss'], indicators['auc']

            if epoch % 10 == 0:
                x_list.append(epoch)
                train_loss_list.append(t_loss)
                train_auc_list.append(t_auc)
            if epoch % 100 == 0:
                self.logger.info('\tEpoch {}\tTrain Loss: {:.6}\tAUC: {:.6}'.format(
                    epoch, t_loss, t_auc))

            if epoch % 10 == 0 and validate:
                with torch.no_grad():
                    self.model.eval()
                    indicators = self.exec_core(dl=self.eval_dl, opt=None)
                    v_loss, v_auc = indicators['loss'], indicators['auc']
                eval_loss_list.append(v_loss)
                eval_auc_list.append(v_auc)
            if epoch % 100 == 0 and validate:
                self.logger.info('\tEpoch {}\tValid Loss: {:.6}\tAUC: {:.6}'.format(
                    epoch, v_loss, v_auc))
                if self.config.waviness_l1 or self.config.waviness_l2:
                    self.logger.info('\tEpoch {}\tW1: {:.6}\tW2: {:.6}'.format(
                        epoch, indicators['waviness_l1'], indicators['waviness_l2']))
                if v_auc > best['auc']:
                    best['auc'] = v_auc
                    best['auc_epoch'] = epoch
                    # report['best_eval_auc'] = bset_eval_auc
                    # report['best_eval_auc_epoch'] = epoch
                    save_model(self.config, self.model, v_auc, epoch)
                    self.logger.info(
                        f'Best AUC {v_auc:.6} refreshed and saved!')

                    # Pred & Accu Relation
                    q_all_count, q_cor_count, q_pred_list = indicators['qa_relation']
                    pa_scat_x = list()
                    pa_scat_y = list()
                    for q, l in q_pred_list.items():
                        all_acc = q_cor_count[q] / q_all_count[q]
                        for p in l:
                            pa_scat_x.append(p)
                            pa_scat_y.append(all_acc)
                    save_pred_accu_relation(self.config, pa_scat_x, pa_scat_y)

                else:
                    self.logger.info(
                        f'Best AUC {best["auc"]:.6} at epoch {best["auc_epoch"]}')

            if epoch % 100 == 0:
                self.logger.info(
                    f'{timeSince(start_time, epoch / self.config.epoch_size)} ({epoch} {epoch / self.config.epoch_size * 100})')

        # save_log(self.config, (x_list, train_loss_list, train_auc_list,
        #                   eval_loss_list, eval_auc_list), v_auc, epoch)
        save_learning_curve(x_list, train_loss_list, train_auc_list,
                            eval_loss_list, eval_auc_list, self.config)

    def exec_core(self, dl, opt, last_epoch=False):
        arr_len = len(dl) if not self.config.debug else 1
        pred_mx = np.zeros([arr_len, self.config.batch_size])
        actu_mx = np.zeros([arr_len, self.config.batch_size])
        loss_ar = np.zeros(arr_len)
        wvn1_ar = np.zeros(arr_len)
        wvn2_ar = np.zeros(arr_len)
        if last_epoch:
            q_all_count = defaultdict(int)
            q_cor_count = defaultdict(int)
            q_pred_list = defaultdict(list)
        for i, (xseq, yseq) in enumerate(dl):
            # yseq.shape : (100, 20, 2) (batch_size, seq_size, len([q, a]))
            out = self.model.loss_batch(xseq, yseq, opt=opt)
            loss_ar[i] = out['loss'].item()
            wvn1_ar[i] = out.get('waviness_l1')
            wvn2_ar[i] = out.get('waviness_l2')
            # out['pred_prob'].shape : (20, 100) (seq_len, batch_size)
            pred_mx[i] = out['pred_prob'][-1, :].detach().view(-1).cpu()
            actu_mx[i] = yseq[:, -1, 1].view(-1).cpu()
            if last_epoch:
                for p, a, q in zip(pred_mx[i], actu_mx[i], yseq[:, -1, 0].view(-1).cpu()):
                    q_all_count[q.item()] += 1
                    q_cor_count[q.item()] += int(a)
                    q_pred_list[q.item()].append(p)

            if self.config.debug:
                break

        # AUC
        fpr, tpr, _thresholds = metrics.roc_curve(
            actu_mx.reshape(-1), pred_mx.reshape(-1), pos_label=1)
        auc = metrics.auc(fpr, tpr)

        indicators = {
            'loss': loss_ar.mean(),
            'auc': auc,
            'waviness_l1': wvn1_ar.mean(),
            'waviness_l2': wvn2_ar.mean(),
        }
        if last_epoch:
            indicators['qa_relation'] = (q_all_count, q_cor_count, q_pred_list)
        return indicators

    def _train_model_simple(self):
        '''最小構成を見て基本を思い出す'''
        for epoch in range(1, self.config.epoch_size + 1):
            self.model.train()
            for i, (xseq, yseq) in enumerate(self.train_dl):
                out = self.model.loss_batch(xseq, yseq, opt=self.opt)

    def evaluate_model(self):
        self.logger.info('Starting evaluation')
        start_time = time.time()
        with torch.no_grad():
            self.model.eval()
            indicators = self.exec_core(dl=self.eval_dl, opt=None, last_epoch=True)
            v_loss, v_auc = indicators['loss'], indicators['auc']

            self.logger.info('\tValid Loss: {:.6}\tAUC: {:.6}'.format(
                v_loss, v_auc))
            if self.config.waviness_l1 or self.config.waviness_l2:
                self.logger.info('\tW1: {:.6}\tW2: {:.6}'.format(
                    indicators['waviness_l1'], indicators['waviness_l2']))

            # Pred & Accu Relation
            q_all_count, q_cor_count, q_pred_list = indicators['qa_relation']
            pa_scat_x = list()
            pa_scat_y = list()
            for q, l in q_pred_list.items():
                all_acc = q_cor_count[q] / q_all_count[q]
                for p in l:
                    pa_scat_x.append(p)
                    pa_scat_y.append(all_acc)
            save_pred_accu_relation(self.config, pa_scat_x, pa_scat_y)

        self.logger.info(f'{timeSince(start_time, 1)}')

    # # ========================
    # # Load trained model
    # # ========================
    # if config.load_model:
    #     model.load_state_dict(torch.load(config.load_model))
    #     model = model.to(dev)

    # # model is trained or loaded now.

    # if config.plot_heatmap:
    #     batch_size = 1
    #     # TODO: don't repeat yourself
    #     if config.load_model:
    #         model.load_state_dict(torch.load(config.load_model))
    #         model = model.to(dev)
    #     heat_dl = prepare_heatmap_data(
    #         config.source_data, config.model_name, n_skills, PRESERVED_TOKENS,
    #         min_n=3, max_n=config.sequence_size, batch_size=batch_size, device=dev, sliding_window=0,
    #         params={'extend_backward': config.extend_backward, 'extend_forward': config.extend_forward})
    #     loss_func = nn.BCELoss()
    #     opt = optim.SGD(model.parameters(), lr=config.lr)

    #     with torch.no_grad():
    #         model.eval()
    #         all_out_prob = []
    #         val_pred = []
    #         val_actual = []
    #         current_eval_loss = []
    #         yticklabels = set()
    #         xticklabels = []
    #         for args in heat_dl:
    #             loss_item, batch_n, pred, actu_q, actu, pred_ks, _, _ = loss_batch(
    #                 model, loss_func, *args, opt=None)
    #             # current_eval_loss.append(loss_item[-1])
    #             # print(pred.shape, actu.shape)
    #             # val_pred.append(pred[-1])
    #             # val_actual.append(actu[-1])
    #             yq = torch.max(actu_q.squeeze(), 0)[1].item()
    #             ya = int(actu.item())
    #             yticklabels.add(yq)
    #             xticklabels.append((yq, ya))

    #             # print(pred_ks.shape)
    #             assert len(pred_ks.shape) == 1, 'pred_ks dimention {}, expected 1'.format(
    #                 pred_ks.shape)
    #             assert pred_ks.shape[0] == n_skills
    #             all_out_prob.append(pred_ks.unsqueeze(0))

    #     _d = torch.cat(all_out_prob).transpose(0, 1)
    #     _d = _d.cpu().numpy()
    #     print(_d.shape)
    #     print(len(yticklabels), len(xticklabels))
    #     yticklabels = sorted(list(yticklabels))
    #     related_d = np.matrix([_d[x, :] for x in yticklabels])

    #     # Regular Heatmap
    #     # fig, ax = plt.subplots(figsize=(20, 10))
    #     # sns.heatmap(_d, ax=ax)

    #     fig, ax = plt.subplots(figsize=(20, 7))
    #     sns.heatmap(
    #         related_d, vmin=0, vmax=1, ax=ax,
    #         # cmap="Reds_r",
    #         xticklabels=['{}'.format(y) for y in xticklabels],
    #         yticklabels=['s{}'.format(x) for x in yticklabels],
    #     )
    #     xtick_dic = {s: i for i, s in enumerate(yticklabels)}
    #     # 正解
    #     sca_x = [t + 0.5 for t, qa in enumerate(xticklabels) if qa[1] == 1]
    #     sca_y = [xtick_dic[qa[0]] + 0.5 for t,
    #              qa in enumerate(xticklabels) if qa[1] == 1]
    #     ax.scatter(sca_x, sca_y, marker='o', s=100, color='white')
    #     # 不正解
    #     sca_x = [t + 0.5 for t, qa in enumerate(xticklabels) if qa[1] == 0]
    #     sca_y = [xtick_dic[qa[0]] + 0.5 for t,
    #              qa in enumerate(xticklabels) if qa[1] == 0]
    #     ax.scatter(sca_x, sca_y, marker='X', s=100, color='black')

    #     save_hm_fig(config, fig)

    # return report
