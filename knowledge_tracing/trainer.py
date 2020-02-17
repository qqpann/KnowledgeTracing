import torch

import time
import logging
import numpy as np
from pathlib import Path
from math import log, ceil
from sklearn import metrics
from collections import defaultdict

from src.data import prepare_dataloader, prepare_dummy_dataloader, prepare_heatmap_dataloader, DataHandler
from src.save import save_model, save_log, save_report, save_hm_fig, save_learning_curve, save_pred_accu_relation
from src.utils import sAsMinutes, timeSince
from src.logging import get_logger
from model.dkt import DKT
from model.ksdkt import KSDKT
from model.eddkt import EDDKT
from model.geddkt import GEDDKT
from model.dkvmn import MODEL as DKVMN


class Trainer(object):

    def __init__(self, config):
        self.config = config
        self.logger = self.get_logger(self.config)
        self.device = self.get_device(self.config)
        self.dh = DataHandler(self.config, self.device, folds=5)
        self.dummy_dl = self.get_dummy_dataloader(self.config, self.device)

    def init_model(self):
        self.model = self.get_model(self.config, self.device)
        self.opt = self.get_opt(self.model)

    def load_model(self):
        if self.config.load_model:
            model.load_state_dict(torch.load(str(self.config.load_model_path)))
            model = model.to(self.device)
        self.model = model

    def init_report(self):
        self._report = {
            'config': self.config.as_dict(),
            'indicator': defaultdict(lambda: defaultdict(list))
        }

    def dump_report(self):
        # self._report['indicator'] = dict(self._report['indicator'])
        save_report(self.config, self._report)

    def report(self, k, key, val):
        self._report['indicator'][key][k].append(val)

    def get_logger(self, config):
        outdir = config.resultsdir / 'report' / config.starttime
        outdir.mkdir(parents=True, exist_ok=True)
        logger = get_logger(
            '{}/{}'.format(config.model_name, config.exp_name),
            outdir / '{}_{}.log'.format(config.config_name, config.exp_name)
        )
        return logger

    def get_device(self, config):
        self.logger.info('PyTorch: {}'.format(torch.__version__))
        device = torch.device(
            'cuda' if config.cuda and torch.cuda.is_available() else 'cpu')
        self.logger.info('Using Device: {}'.format(device))
        return device

    def get_model(self, config, device):
        if config.model_name == 'eddkt':
            model = EDDKT(config, device).to(device)
        elif config.model_name == 'geddkt':
            model = GEDDKT(config, device).to(device)
        elif config.model_name == 'dkt':
            model = DKT(config, device).to(device)
        elif config.model_name == 'ksdkt':
            model = KSDKT(config, device).to(device)
        elif config.model_name == 'dkvmn':
            model = DKVMN(config, device).to(device)
        else:
            raise ValueError(f'model_name {config.model_name} is wrong')

        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(
            f'The model has {count_parameters(model):,} trainable parameters')
        return model

    def get_dataloader(self, config, device):
        train_dl, eval_dl = prepare_dataloader(
            config, device=device, pad=config.pad)
        self.logger.info(
            'train_dl.dataset size: {}'.format(len(train_dl.dataset)))
        self.logger.info(
            'eval_dl.dataset size: {}'.format(len(eval_dl.dataset)))
        return train_dl, eval_dl

    def get_dummy_dataloader(self, config, device):
        return prepare_dummy_dataloader(config, config.sequence_size, 1, device)

    def get_opt(self, model):
        opt = torch.optim.SGD(model.parameters(), lr=self.config.lr)
        if self.config.model_name == 'dkvmn':
            opt = torch.optim.Adam(params=model.parameters(
            ), lr=self.config.lr, betas=(0.9, 0.9))  # from DKVMN
        return opt

    def kfold(self):
        self.init_report()
        test_dl = self.dh.get_test_dl()
        for k, (train_dl, valid_dl) in enumerate(self.dh.gen_trainval_dl()):
            self.init_model()
            self.logger.info('train_dl.dataset size: {}'.format(len(train_dl.dataset)))
            self.logger.info('valid_dl.dataset size: {}'.format(len(valid_dl.dataset)))
            self.train_model(k, train_dl, valid_dl)

            self.logger.info('test_dl.dataset size: {}'.format(len(test_dl.dataset)))
            self.test_model(k, test_dl)

    def pre_train_model(self):
        epoch_size = self.config.pre_dummy_epoch_size
        if epoch_size == 0:
            return
        self.logger.info('Start pre train')
        real_batch_size = self.model.config.batch_size
        try:
            self.model.batch_size = 1
        except AttributeError as e:
            self.logger.warning('{}'.format(e))
        except Exception as e:
            self.logger.error('{}'.format(e))
        self.model.config.batch_size = 1
        for epoch in range(1, epoch_size + 1):
            self.model.train()
            for i, (xseq, yseq, mask) in enumerate(self.dummy_dl):
                out = self.model.loss_batch(xseq, yseq, mask, opt=self.opt)
        self.model.batch_size = real_batch_size
        self.model.config.batch_size = real_batch_size

    def train_model(self, k, train_dl, valid_dl, validate=True):
        self.pre_train_model()
        self.logger.info('Starting train')
        best = {
            'auc': 0.,
            'auc_epoch': 0,
        }
        start_time = time.time()
        for epoch in range(1, self.config.epoch_size + 1):
            self.model.train()
            t_idc = self.exec_core(train_dl, self.opt)
            t_loss, t_auc = t_idc['loss'], t_idc['auc']

            if epoch % 10 == 0:
                self.report(k, 'epoch', epoch)
                self.report(k, 'train_loss', t_loss)
                self.report(k, 'train_auc', t_auc)
            if epoch % 100 == 0:
                self.logger.info('\tEpoch {}\tTrain Loss: {:.6}\tAUC: {:.6}'.format(
                    epoch, t_loss, t_auc))

            if epoch % 10 == 0 and validate:
                with torch.no_grad():
                    self.model.eval()
                    v_idc = self.exec_core(dl=valid_dl, opt=None)
                    v_loss, v_auc = v_idc['loss'], v_idc['auc']
                self.report(k, 'eval_loss', v_loss)
                self.report(k, 'eval_auc', v_auc)
                self.report(k, 'ksvector_l1', v_idc['ksvector_l1'])
                if self.config.waviness_l1 or self.config.waviness_l2:
                    self.report(k, 'waviness_l1', v_idc['waviness_l1'])
                    self.report(k, 'waviness_l2', v_idc['waviness_l2'])
            if epoch % 100 == 0 and validate:
                self.logger.info('\tEpoch {}\tValid Loss: {:.6}\tAUC: {:.6}'.format(
                    epoch, v_loss, v_auc))
                self.logger.info('\tEpoch {}\tKSVectorLoss: {:.6}'.format(
                    epoch, v_idc['ksvector_l1']))
                if self.config.waviness_l1 or self.config.waviness_l2:
                    self.logger.info('\tEpoch {}\tW1: {:.6}\tW2: {:.6}'.format(
                        epoch, v_idc['waviness_l1'], v_idc['waviness_l2']))
                if v_auc > best['auc']:
                    best['auc'] = v_auc
                    best['auc_epoch'] = epoch
                    # report['best_eval_auc'] = bset_eval_auc
                    # report['best_eval_auc_epoch'] = epoch
                    save_model(self.config, self.model, v_auc, epoch)
                    self.logger.info(
                        f'Best AUC {v_auc:.6} refreshed and saved!')
                else:
                    self.logger.info(
                        f'Best AUC {best["auc"]:.6} at epoch {best["auc_epoch"]}')

            if epoch % 100 == 0:
                self.logger.info(
                    f'{timeSince(start_time, epoch / self.config.epoch_size)} ({epoch}epoch {epoch / self.config.epoch_size * 100:.1f}%)')

        # save_log(self.config, (x_list, train_loss_list, train_auc_list,
        #                   eval_loss_list, eval_auc_list), v_auc, epoch)
        save_learning_curve(
            {key: self._report['indicator'][key][k] for key in
             ['epoch', 'train_loss', 'train_auc', 'eval_loss', 'eval_auc',
              'ksvector_l1', 'waviness_l1', 'waviness_l2']},
            self.config)

    def exec_core(self, dl, opt, only_eval=False):
        arr_len = len(dl) if not self.config.debug else 1
        pred_mx = np.zeros([arr_len, self.config.batch_size])
        actu_mx = np.zeros([arr_len, self.config.batch_size])
        pred_ls = []
        actu_ls = []
        pred_v_mx = np.zeros(
            [arr_len, self.config.batch_size * self.config.n_skills])
        actu_v_mx = np.zeros(
            [arr_len, self.config.batch_size * self.config.n_skills])
        loss_ar = np.zeros(arr_len)
        wvn1_ar = np.zeros(arr_len)
        wvn2_ar = np.zeros(arr_len)
        ksv1_ar = np.zeros(arr_len)
        # ##
        # if self.config.model_name == 'dkvmn':
        #     pred_list = []
        #     target_list = []
        # ##
        if only_eval:
            q_all_count = defaultdict(int)
            q_cor_count = defaultdict(int)
            q_pred_list = defaultdict(list)
        for i, (xseq, yseq, mask) in enumerate(dl):
            # yseq.shape : (100, 20, 2) (batch_size, seq_size, len([q, a]))
            out = self.model.loss_batch(xseq, yseq, mask, opt=opt)
            loss_ar[i] = out['loss'].item()
            wvn1_ar[i] = out.get('waviness_l1')
            wvn2_ar[i] = out.get('waviness_l2')
            ksv1_ar[i] = out.get('ksvector_l1')
            # ##
            # if self.config.model_name == 'dkvmn':
            #     right_target = np.asarray(out.get('filtered_target').data.tolist())
            #     right_pred = np.asarray(out.get('filtered_pred').data.tolist())
            #     pred_list.append(right_pred)
            #     target_list.append(right_target)
            # ##
            # out['pred_prob'].shape : (20, 100) (seq_len, batch_size)
            if out.get('pred_prob', False) is not False:
                # print(out['pred_prob'], out['pred_prob'].shape)
                pred_mx[i] = out['pred_prob'][-1, :].detach().view(-1).cpu()
                pred_ls.append(out['filtered_pred'])
                actu_ls.append(out['filtered_target'])
            actu_mx[i] = yseq[:, -1, 1].view(-1).cpu()
            # ksvector_l1 = torch.sum(torch.abs((Sdq * pred_vect) - (Sdqa))) \
            #     / (Sdq.shape[0] * Sdq.shape[1] * Sdq.shape[2])
            if out.get('Sdq', False) is not False:
                pred_v_mx[i] = (out['Sdq'] * out['pred_vect'])[-1, :, :]\
                    .detach().view(-1).cpu()
                actu_v_mx[i] = out['Sdqa'][-1, :, :].view(-1).cpu()
            if only_eval:
                for p, a, q in zip(pred_mx[i], actu_mx[i], yseq[:, -1, 0].view(-1).cpu()):
                    q_all_count[q.item()] += 1
                    q_cor_count[q.item()] += int(a)
                    q_pred_list[q.item()].append(p)

            if self.config.debug:
                break
        # #
        # if self.config.model_name == 'dkvmn':
        #     all_pred = np.concatenate(pred_list, axis=0)
        #     all_target = np.concatenate(target_list, axis=0)
        # #
        # AUC
        # fpr, tpr, _thresholds = metrics.roc_curve(
        #     actu_mx.reshape(-1), pred_mx.reshape(-1), pos_label=1)

        fpr, tpr, _thresholds = metrics.roc_curve(
            torch.cat(actu_ls).detach().cpu().numpy().reshape(-1),
            torch.cat(pred_ls).detach().cpu().numpy().reshape(-1), pos_label=1)
        auc = metrics.auc(fpr, tpr)
        # if self.config.model_name == 'dkvmn':
        #     auc = metrics.roc_auc_score(all_target, all_pred)  # for DKVMN
        # KSVector AUC
        fpr_v, tpr_v, _thresholds_v = metrics.roc_curve(
            actu_v_mx.reshape(-1), pred_v_mx.reshape(-1), pos_label=1)
        auc_ksv = metrics.auc(fpr_v, tpr_v)

        indicators = {
            'loss': loss_ar.mean(),
            'auc': auc,
            'ksv_auc': auc_ksv,
            'waviness_l1': wvn1_ar.mean(),
            'waviness_l2': wvn2_ar.mean(),
            'ksvector_l1': ksv1_ar.mean(),
        }
        if only_eval:
            indicators['qa_relation'] = (q_all_count, q_cor_count, q_pred_list)
        return indicators

    def _train_model_simple(self, train_dl):
        '''最小構成を見て基本を思い出す'''
        for epoch in range(1, self.config.epoch_size + 1):
            self.model.train()
            for i, (xseq, yseq) in enumerate(train_dl):
                out = self.model.loss_batch(xseq, yseq, opt=self.opt)

    def test_model(self, k, test_dl):
        self.logger.info('Starting evaluation')
        start_time = time.time()
        with torch.no_grad():
            self.model.eval()
            indicators = self.exec_core(dl=test_dl, opt=None, only_eval=True)
            v_loss, v_auc = indicators['loss'], indicators['auc']

            self.logger.info('\tValid Loss: {:.6}\tAUC: {:.6}'.format(v_loss, v_auc))
            self.logger.info('\tValid KSV AUC: {:.6}'.format(indicators['ksv_auc']))
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

            # BAD: avoid error for only DKVMN
            if self.config.model_name == 'dkvmn':
                return
            bad = 0
            good = 0
            bak_batch_size = self.model.batch_size
            self.model.batch_size = 1
            self.model.config.batch_size = 1
            dummy_len = self.config.sequence_size
            for v in range(self.config.n_skills):
                # wrong
                wro = self.model.loss_batch(
                    torch.Tensor([(v, 0) for _ in range(dummy_len)]).unsqueeze(0),
                    torch.Tensor([(v, 0) for _ in range(dummy_len)]).unsqueeze(0),
                    torch.BoolTensor([True]*self.config.sequence_size).unsqueeze(0),
                    opt=None)
                wro = wro['pred_prob']
                # correct
                cor = self.model.loss_batch(
                    torch.Tensor([(v, 1) for _ in range(dummy_len)]).unsqueeze(0),
                    torch.Tensor([(v, 1) for _ in range(dummy_len)]).unsqueeze(0),
                    torch.BoolTensor([True]*self.config.sequence_size).unsqueeze(0),
                    opt=None)
                cor = cor['pred_prob']
                if (cor - wro)[-1].item() < 0:
                    bad += 1
                else:
                    good += 1
            self.logger.info('Good: {} \t Bad: {}'.format(good, bad))
            self.model.batch_size = bak_batch_size
            self.model.config.batch_size = bak_batch_size

    def evaluate_model_heatmap(self):
        uid, heat_dl = prepare_heatmap_dataloader(
            self.config, self.config.sequence_size, 1, self.device)
        self.logger.info("Heatmap data's user id is {}".format(uid))
        self.plot_heatmap(self.config, heat_dl)

    def plot_heatmap(self, config, heat_dl=None):
        real_batch_size = self.model.config.batch_size
        try:
            self.model.batch_size = 1
        except AttributeError as e:
            self.logger.warning('{}'.format(e))
        except Exception as e:
            self.logger.error('{}'.format(e))
        self.model.config.batch_size = 1

        # TODO: don't repeat yourself
        with torch.no_grad():
            self.model.eval()
            all_out_prob = []
            yticklabels = set()
            xticklabels = []
            for i, (xseq, yseq) in enumerate(heat_dl):
                # yseq.shape : (100, 20, 2) (batch_size, seq_size, len([q, a]))
                out = self.model.loss_batch(xseq, yseq, opt=None)

                assert out['pred_vect'].shape == (
                    self.config.sequence_size if self.config.model_name in {'ksdkt', 'dkt'} else self.config.eddkt['extend_forward']+1, self.config.batch_size, self.config.n_skills)
                pred_ks = out['pred_vect'][-1, :, :].squeeze()
                # print(pred_ks.shape)

                yq = int(yseq[-1, -1, 0].item())
                ya = int(yseq[-1, -1, 1].item())
                yticklabels.add(yq)
                xticklabels.append((yq, ya))
                all_out_prob.append(pred_ks)

        _d = torch.stack(all_out_prob).transpose(0, 1)
        _d = _d.cpu().numpy()
        # print(_d.shape)
        # print(len(yticklabels), len(xticklabels))
        yticklabels = sorted(list(yticklabels))
        related_d = np.matrix([_d[x, :] for x in yticklabels])

        # Regular Heatmap
        # fig, ax = plt.subplots(figsize=(20, 10))
        # sns.heatmap(_d, ax=ax)

        import matplotlib.pyplot as plt
        import seaborn as sns
        fig, ax = plt.subplots(figsize=(20, 7))
        sns.heatmap(
            related_d, vmin=0, vmax=1, ax=ax,
            # cmap="Reds_r",
            xticklabels=['{}'.format(y) for y in xticklabels],
            yticklabels=['s{}'.format(x) for x in yticklabels],
        )
        xtick_dic = {s: i for i, s in enumerate(yticklabels)}
        # 正解
        sca_x = [t + 0.5 for t, qa in enumerate(xticklabels) if qa[1] == 1]
        sca_y = [xtick_dic[qa[0]] + 0.5 for t,
                 qa in enumerate(xticklabels) if qa[1] == 1]
        ax.scatter(sca_x, sca_y, marker='o', s=100, color='white')
        # 不正解
        sca_x = [t + 0.5 for t, qa in enumerate(xticklabels) if qa[1] == 0]
        sca_y = [xtick_dic[qa[0]] + 0.5 for t,
                 qa in enumerate(xticklabels) if qa[1] == 0]
        ax.scatter(sca_x, sca_y, marker='X', s=100, color='black')

        save_hm_fig(config, fig)

        self.model.batch_size = real_batch_size
        self.model.config.batch_size = real_batch_size
