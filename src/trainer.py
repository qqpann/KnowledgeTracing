import logging
import time
from collections import defaultdict
from math import ceil, log
from pathlib import Path
from statistics import mean, stdev
from typing import Union, DefaultDict, Dict, List

import numpy as np
import torch
from sklearn import metrics
from sklearn.metrics import ndcg_score as ndcg
import optuna

from model.dkt import DKT
from model.dkvmn import MODEL as DKVMN
from model.eddkt import EDDKT
from model.geddkt import GEDDKT
from model.ksdkt import KSDKT
from src.data import DataHandler
from src.log import get_logger
from src.report import Report
from src.save import (
    save_hm_fig,
    save_learning_curve,
    save_log,
    save_model,
    save_pred_accu_relation,
    save_report,
)
from src.utils import sAsMinutes, timeSince


class Trainer(object):
    def __init__(self, config, trial: Union[None, optuna.Trial] = None):
        self.config = config
        self.logger = self.get_logger(self.config)
        self.device = self.get_device(self.config)
        self.dh = DataHandler(self.config, self.device)
        self.dummy_dl = self.dh.get_straighten_dl()
        self.best_score = 0.0
        self.trial = trial

    def init_model(self):
        self.model = self.get_model(self.config, self.device)
        self.opt = self.get_opt(self.model)

    def load_model(self, load_model=None):
        if load_model:
            load_model_path = load_model
        elif self.config.load_model:
            load_model_path = str(self.config.load_model_path)
        else:
            return None
        self.logger.info("Loading model {}".format(load_model_path))
        self.model.load_state_dict(torch.load(load_model_path))
        self.model.to(self.device)

    def init_report(self):
        self.report = Report(self.config)

    def dump_report(self):
        self.report.dump()

    def get_logger(self, config):
        outdir = config.resultsdir / "log" / config.starttime
        outdir.mkdir(parents=True, exist_ok=True)
        logger = get_logger(
            "{}/{}".format(config.model_name, config.exp_name),
            outdir / "{}_{}.log".format(config.config_name, config.exp_name),
        )
        return logger

    def get_device(self, config):
        self.logger.info("PyTorch: {}".format(torch.__version__))
        device = torch.device(
            "cuda" if config.cuda and torch.cuda.is_available() else "cpu"
        )
        self.logger.info("Using Device: {}".format(device))
        return device

    def get_model(self, config, device):
        if config.model_name == "eddkt":
            model = EDDKT(config, device).to(device)
        elif config.model_name == "geddkt":
            model = GEDDKT(config, device).to(device)
        elif config.model_name == "dkt":
            model = DKT(config, device).to(device)
        elif config.model_name == "ksdkt":
            model = KSDKT(config, device).to(device)
        elif config.model_name == "dkvmn":
            model = DKVMN(config, device).to(device)
        else:
            raise ValueError(f"model_name {config.model_name} is wrong")

        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        self.logger.info(
            f"The model has {count_parameters(model):,} trainable parameters"
        )
        return model

    def get_opt(self, model):
        if self.config.model_name == "dkvmn":
            opt = torch.optim.Adam(
                params=model.parameters(), lr=self.config.lr, betas=(0.9, 0.9)
            )  # from DKVMN
            return opt
        opt = torch.optim.SGD(model.parameters(), lr=self.config.lr)
        return opt

    def optimize(self):
        projectdir = self.config.projectdir
        name = self.config.source_data
        self.init_report()
        fintrain_dl, fintest_dl = self.dh.get_traintest_dl()
        self.logger.info(
            "fintrain_dl.dataset size: {}".format(len(fintrain_dl.dataset))
        )
        self.logger.info("fintest_dl.dataset size: {}".format(len(fintest_dl.dataset)))
        self.report.subname = "all"
        self.init_model()
        self.train_model(
            fintrain_dl,
            None,
            self.config.epoch_size,
            subname="all",
            validate=False,
            optimize=True,
        )
        # self.test_model(fintest_dl, subname='all', do_report=True)

    def cv(self):
        projectdir = self.config.projectdir
        name = self.config.source_data
        self.init_report()
        fintrain_dl, fintest_dl = self.dh.get_traintest_dl()
        for k, (train_dl, valid_dl) in enumerate(self.dh.generate_trainval_dl()):
            self.report.subname = k
            self.init_model()
            self.logger.info("train_dl.dataset size: {}".format(len(train_dl.dataset)))
            self.logger.info("valid_dl.dataset size: {}".format(len(valid_dl.dataset)))
            # assert len(train_dl) > 0 and len(valid_dl) > 0, 'k:{},train:{},valid:{}'.format(k, len(train_dl), len(valid_dl))
            self.train_model(train_dl, valid_dl, self.config.epoch_size, subname=k)

            self.load_model(
                self.config.resultsdir
                / "checkpoints"
                / self.config.starttime
                / "f{}_best.model".format(k)
            )
            self.logger.info("test_dl.dataset size: {}".format(len(fintest_dl.dataset)))
            self.test_model(fintest_dl, subname=k, do_report=True)
            if self.config.debug:
                break
        self.logger.info(
            "fintrain_dl.dataset size: {}".format(len(fintrain_dl.dataset))
        )
        self.logger.info("fintest_dl.dataset size: {}".format(len(fintest_dl.dataset)))
        self.report.subname = "all"
        self.init_model()
        # TODO: Fix bad usage of k
        test_epoch_size = round(
            mean([self.report._best["auc_epoch"][i] for i in range(k + 1)]), -1
        )
        self.train_model(
            fintrain_dl, None, test_epoch_size, subname="all", validate=False
        )
        self.test_model(fintest_dl, subname="all", do_report=True)

    def evaluate_model(self, load_model: str = None):
        self.init_report()
        self.init_model()
        self.load_model(load_model)
        fintrain_dl, fintest_dl = self.dh.get_traintest_dl()
        # fintest_dl = self.dh.get_enwrap_test_dl()
        self.test_model(fintest_dl, subname="all", do_report=True)

    def evaluate_model_enwrapped(self, load_model: str = None):
        self.init_report()
        self.init_model()
        self.load_model(load_model)
        defaultcount: DefaultDict[str, int] = defaultdict(int)
        for student_data in self.dh.fintrain_data:
            q, _ = student_data[0]
            for qnext, _ in student_data[1:]:
                defaultcount[f"{q}->{qnext}"] -= 1
                q = qnext
        count: Dict[str, int] = dict(defaultcount)
        fintrain_dl, _ = self.dh.get_traintest_dl()
        for i, (xseq, yseq, mask) in enumerate(fintrain_dl):
            masks = [int(torch.sum(m).item()) for m in mask]
            for xq, yq, m in zip(xseq[:, :, 0], yseq[:, :, 0], masks):  # batch loop
                for x, y in zip(xq[:m], yq[:m]):
                    count[f"{x.item()}->{y.item()}"] += 1
        print("enwrap:", self.config.split_data_enwrap)
        fintest_dl = self.dh.get_enwrap_test_dl()
        duo_context: DefaultDict[str, DefaultDict[str, List]] = defaultdict(
            lambda: defaultdict(list)
        )
        for i, (xseq, yseq, mask) in enumerate(fintest_dl):
            out = self.model.forward(xseq, yseq, mask, opt=None)
            for ys, prob in zip(yseq, out["pred_prob"].permute(1, 0)):
                duo_context["->".join(map(str, ys[-2:, 0].cpu().numpy()))][
                    "pred"
                ].append(prob[-1].item())
                duo_context["->".join(map(str, ys[-2:, 0].cpu().numpy()))][
                    "actu"
                ].append(ys[-1:, 1].item())
        fintrain_dl, _ = self.dh.get_traintest_dl()
        print(count)
        for key, value in duo_context.items():
            fpr, tpr, _ = metrics.roc_curve(value["actu"], value["pred"], pos_label=1)
            auc = metrics.auc(fpr, tpr)
            print(key, auc)
            duo_context[key]["auc"] = [auc]
            duo_context[key]["count"] = [count.get(key, 0)]
        self.report.subname = "all"
        self.report("duo_context", duo_context)
        self.report.dump(fname="duo_context.json")

    def straighten_train_model(self, epoch_size: int):
        if epoch_size == 0:
            return
        self.logger.info("Start straightening pre-train")
        for _ in range(1, epoch_size + 1):
            self.model.train()
            for xseq, yseq, mask in self.dummy_dl:
                self.model.forward(xseq, yseq, mask, opt=self.opt)

    def train_model(
        self,
        train_dl,
        valid_dl,
        epoch_size: int,
        subname: str,
        validate=True,
        optimize=False,
    ):
        self.straighten_train_model(epoch_size=self.config.pre_dummy_epoch_size)
        # if self.config.transfer_learning:
        #     self.logger.info('Transfer learning')
        #     self.model.embedding.weight.requires_grad = False
        self.logger.info("Starting train")
        start_time = time.time()
        for epoch in range(1, epoch_size + 1):
            self.model.train()
            if (
                self.config.straighten_during_train_every
                and epoch % self.config.straighten_during_train_every == 0
            ):
                self.logger.info("Start straightening during train")
                self.straighten_train_model(
                    epoch_size=self.config.straighten_during_train_for
                )
            t_idc = self.exec_core(train_dl, self.opt)
            t_loss, t_auc = t_idc["loss"], t_idc["auc"]

            if epoch % 10 == 0:
                self.report("epoch", epoch)
                self.report("train_loss", t_loss)
                self.report("train_auc", t_auc)
            if epoch % 100 == 0:
                self.logger.info(
                    "\tEpoch {}\tTrain Loss: {:.6}\tAUC: {:.6}".format(
                        epoch, t_loss, t_auc
                    )
                )

            if epoch % 10 == 0 and validate:
                with torch.no_grad():
                    self.model.eval()
                    v_idc = self.exec_core(dl=valid_dl, opt=None)
                    v_loss, v_auc = v_idc["loss"], v_idc["auc"]
                self.report("eval_loss", v_loss)
                self.report("eval_auc", v_auc)
                self.report("ksvector_l1", v_idc["ksvector_l1"])
                if self.config.waviness:
                    self.report("waviness_l1", v_idc["waviness_l1"])
                    self.report("waviness_l2", v_idc["waviness_l2"])
                if (
                    self.config.reconstruction
                    or self.config.reconstruction_and_waviness
                ):
                    self.report("eval_auc_c", v_idc["reconstruction_loss"])
                if v_auc > self.report.get_best("auc"):  # best auc
                    self.report.set_best("auc", v_auc)
                    self.report.set_best("auc_epoch", epoch)
                    save_model(
                        self.config, self.model, "f{}_best.model".format(subname)
                    )
            if epoch % 100 == 0 and validate:
                self.logger.info(
                    "\tEpoch {}\tValid Loss: {:.6}\tAUC: {:.6}".format(
                        epoch, v_loss, v_auc
                    )
                )
                self.logger.info(
                    "\tEpoch {}\tKSVectorLoss: {:.6}".format(
                        epoch, v_idc["ksvector_l1"]
                    )
                )
                if self.config.waviness:
                    self.logger.info(
                        "\tEpoch {}\tW1: {:.6}\tW2: {:.6}".format(
                            epoch, v_idc["waviness_l1"], v_idc["waviness_l2"]
                        )
                    )
                if v_auc >= self.report.get_best("auc"):
                    # refresh.
                    save_model(
                        self.config,
                        self.model,
                        f"{self.config.model_name}_auc{v_auc:.4f}_e{epoch}.model",
                    )
                    self.logger.info(f"Best AUC {v_auc:.6} refreshed and saved!")
                elif (
                    epoch - self.report.get_best("auc_epoch")
                ) > self.config.early_stop:
                    # early stop.
                    self.logger.info(
                        f'Best AUC {self.report.get_best("auc"):.6} at epoch {self.report.get_best("auc_epoch")}'
                    )
                    self.logger.info(f"Early stopping.")
                    break
                else:
                    # no refresh, but no early stop yet.
                    self.logger.info(
                        f'Best AUC {self.report.get_best("auc"):.6} at epoch {self.report.get_best("auc_epoch")}'
                    )

            if epoch % 100 == 0:
                self.logger.info(
                    f"{timeSince(start_time, epoch / epoch_size)} ({epoch}epoch {epoch / epoch_size * 100:.1f}%)"
                )
                if optimize and self.trial is not None:
                    self.best_score = self.report.get_best("auc")
                    self.trial.report(t_auc, epoch)
                    if self.trial.should_prune():
                        raise optuna.exceptions.TrialPruned()

        # This is the model checkpoint at the end of epoch, or early stopping
        save_model(self.config, self.model, "f{}_final.model".format(subname))

        # save_log(self.config, (x_list, train_loss_list, train_auc_list,
        #                   eval_loss_list, eval_auc_list), v_auc, epoch)
        # save_learning_curve(
        #     {key: self.report._indicator[key][k] for key in
        #      ['epoch', 'train_loss', 'train_auc', 'eval_loss', 'eval_auc',
        #       'ksvector_l1', 'waviness_l1', 'waviness_l2']},
        #     self.config)

    def exec_core(self, dl, opt, only_eval=False):
        # assert len(dl) > 0, f'{len(dl)}, {len(dl.dataset)}'
        # arr_len = len(dl) if not self.config.debug else 1
        # assert arr_len > 0, f'{dl}, {len(dl)}, {dl.dataset}'
        # pred_mx = np.zeros([arr_len, self.config.batch_size])
        # actu_mx = np.zeros([arr_len, self.config.batch_size])
        pred_ls = []
        actu_ls = []
        actu_c_ls = []
        # pred_v_mx = np.zeros(
        #     [arr_len, self.config.batch_size * self.config.n_skills])
        # actu_v_mx = np.zeros(
        #     [arr_len, self.config.batch_size * self.config.n_skills])
        pred_v_ls = []
        actu_v_ls = []
        # loss_ar = np.zeros(arr_len)
        # wvn1_ar = np.zeros(arr_len)
        # wvn2_ar = np.zeros(arr_len)
        # ksv1_ar = np.zeros(arr_len)
        loss_ls = list()
        wvn1_ls = list()
        wvn2_ls = list()
        ksv1_ls = list()
        rcns_ls = list()
        # ##
        # if self.config.model_name == 'dkvmn':
        #     pred_list = []
        #     target_list = []
        # ##
        # if only_eval:
        #     q_all_count = defaultdict(int)
        #     q_cor_count = defaultdict(int)
        #     q_pred_list = defaultdict(list)
        for i, (xseq, yseq, mask) in enumerate(dl):
            # yseq.shape : (100, 20, 2) (batch_size, seq_size, len([q, a]))
            out = self.model.forward(xseq, yseq, mask, opt=opt)
            # loss_ar[i] = out['loss'].item()
            # wvn1_ar[i] = out.get('waviness_l1')
            # wvn2_ar[i] = out.get('waviness_l2')
            # ksv1_ar[i] = out.get('ksvector_l1')
            loss_ls.append(out["loss"].item())
            wvn1_ls.append(out.get("waviness_l1"))
            wvn2_ls.append(out.get("waviness_l2"))
            ksv1_ls.append(out.get("ksvector_l1"))
            rcns_ls.append(out.get("reconstruction_loss"))
            # ##
            # if self.config.model_name == 'dkvmn':
            #     right_target = np.asarray(out.get('filtered_target').data.tolist())
            #     right_pred = np.asarray(out.get('filtered_pred').data.tolist())
            #     pred_list.append(right_pred)
            #     target_list.append(right_target)
            # ##
            # out['pred_prob'].shape : (20, 100) (seq_len, batch_size)
            # if out.get('pred_prob', False) is not False:
            #     # print(out['pred_prob'], out['pred_prob'].shape)
            #     pred_mx[i] = out['pred_prob'][-1, :].detach().view(-1).cpu()
            # actu_mx[i] = yseq[:, -1, 1].view(-1).cpu()
            if out.get("filtered_pred", False) is not False:
                pred_ls.append(out["filtered_pred"].reshape(-1))
                actu_ls.append(out["filtered_target"].reshape(-1))
                actu_c_ls.append(out["filtered_target_c"].reshape(-1))
            # ksvector_l1 = torch.sum(torch.abs((Sdq * pred_vect) - (Sdqa))) \
            #     / (Sdq.shape[0] * Sdq.shape[1] * Sdq.shape[2])
            if out.get("Sdq", False) is not False:
                # pred_v_mx[i] = (out['Sdq'] * out['pred_vect'])[-1, :, :]\
                #     .detach().view(-1).cpu()
                # actu_v_mx[i] = out['Sdqa'][-1, :, :].view(-1).cpu()
                pred_v_ls.append(
                    (out["Sdq"] * out["pred_vect"])[-1, :, :].detach().view(-1).cpu()
                )
                actu_v_ls.append(out["Sdqa"][-1, :, :].view(-1).cpu())
            # if only_eval:
            #     for p, a, q in zip(pred_mx[i], actu_mx[i], yseq[:, -1, 0].view(-1).cpu()):
            #         q_all_count[q.item()] += 1
            #         q_cor_count[q.item()] += int(a)
            #         q_pred_list[q.item()].append(p)

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

        assert len(actu_ls) > 0 and len(pred_ls) > 0, f"{len(actu_ls)},{len(pred_ls)}"

        fpr, tpr, _thresholds = metrics.roc_curve(
            torch.cat(actu_ls).detach().cpu().numpy().reshape(-1),
            torch.cat(pred_ls).detach().cpu().numpy().reshape(-1),
            pos_label=1,
        )
        auc = metrics.auc(fpr, tpr)
        fpr, tpr, _thresholds = metrics.roc_curve(
            torch.cat(actu_c_ls).detach().cpu().numpy().reshape(-1),
            torch.cat(pred_ls).detach().cpu().numpy().reshape(-1),
            pos_label=1,
        )
        auc_c = metrics.auc(fpr, tpr)
        # if self.config.model_name == 'dkvmn':
        #     auc = metrics.roc_auc_score(all_target, all_pred)  # for DKVMN
        # KSVector AUC
        # fpr_v, tpr_v, _thresholds_v = metrics.roc_curve(
        #     torch.cat(actu_v_ls).detach().cpu().numpy().reshape(-1),
        #     torch.cat(pred_v_ls).detach().cpu().numpy().reshape(-1), pos_label=1)
        # auc_ksv = metrics.auc(fpr_v, tpr_v)

        indicators = {
            "loss": mean(loss_ls),
            "auc": auc,
            "auc_c": auc_c,
            # 'ksv_auc': auc_ksv,
            "waviness_l1": mean(wvn1_ls) if wvn1_ls[0] != None else 0,
            "waviness_l2": mean(wvn2_ls) if wvn2_ls[0] != None else 0,
            "ksvector_l1": mean(ksv1_ls) if ksv1_ls[0] != None else 0,
            "reconstruction_loss": mean(rcns_ls) if ksv1_ls[0] != None else 0,
        }
        # if only_eval:
        #     indicators['qa_relation'] = (q_all_count, q_cor_count, q_pred_list)
        return indicators

    def _train_model_simple(self, train_dl):
        """最小構成を見て基本を思い出す"""
        for epoch in range(1, self.config.epoch_size + 1):
            self.model.train()
            for i, (xseq, yseq, mask) in enumerate(train_dl):
                out = self.model.forward(xseq, yseq, mask, opt=self.opt)

    def test_model(self, test_dl, subname: str, do_report=False):
        self.logger.info("Starting test")
        with torch.no_grad():
            self.model.eval()
            indicators = self.exec_core(dl=test_dl, opt=None, only_eval=True)
            v_loss, v_auc = indicators["loss"], indicators["auc"]
            v_auc_c = indicators["auc_c"]

            if do_report:
                self.report("test_auc", v_auc)
                self.report("test_auc_c", v_auc_c)
            self.logger.info("\tTest Loss: {:.6}\tAUC: {:.6}".format(v_loss, v_auc))
            # self.logger.info('\tTest KSV AUC: {:.6}'.format(indicators['ksv_auc']))
            self.logger.info("\tTest KSV Loss: {:.6}".format(indicators["ksvector_l1"]))
            if self.config.waviness or self.config.reconstruction_and_waviness:
                self.logger.info(
                    "\tW1: {:.6}\tW2: {:.6}".format(
                        indicators["waviness_l1"], indicators["waviness_l2"]
                    )
                )
            if self.config.reconstruction or self.config.reconstruction_and_waviness:
                self.logger.info(
                    "\tr1: {:.6}".format(indicators["reconstruction_loss"])
                )
                self.logger.info("\tTest AUC(C): {:.6}".format(v_auc_c))

            # Reverse Prediction
            seq_size = self.config.sequence_size
            simu = [[0] * i + [1] * (seq_size - i) for i in range(seq_size + 1)[::-1]]
            # simu = [[1]*i + [0]*(seq_size - i) for i in range(seq_size+1)]
            # simu = [[0]*i + [1]*(seq_size - i) for i in range(seq_size)] + [[1]*i + [0]*(seq_size - i) for i in range(seq_size)]
            good, bad = 0, 0
            good_bad = []
            simu_res = dict()
            simu_ndcg = []
            for v in range(self.config.n_skills):
                xs = []
                preds = []
                for s in simu:
                    res = self.model.forward(
                        torch.Tensor([(v, a) for a in s]).unsqueeze(0),
                        torch.Tensor([(v, a) for a in s]).unsqueeze(0),
                        torch.BoolTensor([True] * seq_size).unsqueeze(0),
                    )
                    preds.append(res["pred_prob"][-1].item())
                    xs.append(sum(s))
                # RP soft
                _gb = int(preds[-1] > preds[0])
                good_bad.append(_gb)
                if _gb:
                    good += 1
                else:
                    bad += 1
                # RP hard
                simu_ndcg.append(ndcg(np.asarray([xs]), np.asarray([preds])))
                # raw data
                simu_res[v] = (xs, preds)
            self.logger.info("RP soft \t good:bad = {}:{}".format(good, bad))
            self.logger.info(
                "RP hard \t nDCG = {:.4f}±{:.4f}".format(
                    mean(simu_ndcg), stdev(simu_ndcg)
                )
            )
            # RP soft
            self.report.set_value(
                "RPsoft",
                {
                    "good": good,
                    "bad": bad,
                    "s_good": xs[-1],
                    "s_bad": xs[0],
                    "goodbad": good_bad,
                },
            )
            # RP hard
            self.report.set_value("RPhard", simu_ndcg)
            # raw data
            self.report.set_value("simu_pred", simu_res)
            # Referse prediction *reversed* (from oracle to failing)
            # simu = [[0] * i + [1] * (seq_size - i) for i in range(seq_size + 1)[::-1]]
            simu = [[1] * i + [0] * (seq_size - i) for i in range(seq_size + 1)[::-1]]
            # simu = [[1]*i + [0]*(seq_size - i) for i in range(seq_size+1)]
            # simu = [[0]*i + [1]*(seq_size - i) for i in range(seq_size)] + [[1]*i + [0]*(seq_size - i) for i in range(seq_size)]
            simu_res = dict()
            simu_ndcg = []
            for v in range(self.config.n_skills):
                xs = []
                preds = []
                for s in simu:
                    res = self.model.forward(
                        torch.Tensor([(v, a) for a in s]).unsqueeze(0),
                        torch.Tensor([(v, a) for a in s]).unsqueeze(0),
                        torch.BoolTensor([True] * seq_size).unsqueeze(0),
                    )
                    preds.append(res["pred_prob"][-1].item())
                    xs.append(sum(s))
                # RP hard
                simu_ndcg.append(ndcg(np.asarray([xs]), np.asarray([preds])))
                # raw data
                simu_res[v] = (xs, preds)
            self.logger.info(
                "RP hard *reversed* \t nDCG = {:.4f}±{:.4f}".format(
                    mean(simu_ndcg), stdev(simu_ndcg)
                )
            )
            # RP hard (reversed)
            self.report.set_value("RPhard_reversed", simu_ndcg)

            # Inverted Performance (Reverse Predictionと同じ)
            ip_all_res = []
            for q in range(self.config.n_skills):
                # change the length of synthetic input sequence from 2 to 50
                ip_res = dict()
                for ss in range(seq_size + 1):
                    sequence = [(q, 1 * (_s >= ss)) for _s in range(seq_size)]
                    res = self.model.forward(
                        torch.Tensor(sequence).unsqueeze(0),
                        torch.Tensor(sequence).unsqueeze(0),
                        torch.BoolTensor([True] * seq_size).unsqueeze(0),
                    )
                    # IP
                    ip_res[str(ss)] = res["pred_prob"].view(-1).detach().cpu().tolist()
                ip_all_res.append(ip_res)
            # raw data
            self.report.set_value("inverted_performance", ip_all_res)

