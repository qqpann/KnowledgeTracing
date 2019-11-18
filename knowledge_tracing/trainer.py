import torch

import logging
import numpy as np
from math import log, ceil

from src.data import prepare_data, prepare_dataloader
from model.eddkt import EncDecDKT, get_loss_batch_encdec
from model.basedkt import BaseDKT, get_loss_batch_basedkt
from model.seq2seq import get_Seq2Seq, get_loss_batch_seq2seq


class Trainer(object):
    def __init__(self, config):
        self.config = config
        self.logger = self.get_logger()
        self.device = self.get_device()
        self.model, self.loss_batch, self.train_dl, self.eval_dl = self.get_model()
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
        # =========================
        # Parameters
        # =========================
        batch_size = self.config.batch_size
        n_hidden, n_skills, n_layers = 200, self.config.n_skills, 2
        n_output = n_skills
        PRESERVED_TOKENS = 2  # PAD, SOS
        onehot_size = 2 * n_skills + PRESERVED_TOKENS
        n_input = ceil(log(2 * n_skills))

        INPUT_DIM, ENC_EMB_DIM, ENC_DROPOUT = onehot_size, n_input, 0.6
        OUTPUT_DIM, DEC_EMB_DIM, DEC_DROPOUT = onehot_size, n_input, 0.6
        HID_DIM, N_LAYERS = n_hidden, n_layers
        N_SKILLS = n_skills
        # =========================
        # Prepare models, LossBatch, and Data
        # =========================
        if self.config.model_name == 'encdec':
            model = EncDecDKT(
                INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT,
                OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT,
                N_SKILLS,
                self.device).to(self.device)
            loss_batch = get_loss_batch_encdec(
                self.config.extend_forward, ks_loss=self.config.ks_loss)
            train_dl, eval_dl = prepare_data(
                self.config.source_data, 'encdec', n_skills, PRESERVED_TOKENS,
                min_n=3, max_n=self.config.sequence_size, batch_size=batch_size, device=self.device, sliding_window=0,
                params={'extend_backward': self.config.extend_backward, 'extend_forward': self.config.extend_forward})
        elif self.config.model_name == 'seq2seq':
            model = get_Seq2Seq(
                onehot_size, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT,
                OUTPUT_DIM, DEC_EMB_DIM, DEC_DROPOUT, self.device)
            loss_batch = get_loss_batch_seq2seq(
                self.config.extend_forward, ks_loss=self.config.ks_loss)
            train_dl, eval_dl = prepare_data(
                self.config.source_data, 'encdec', n_skills, PRESERVED_TOKENS,
                min_n=3, max_n=self.config.sequence_size, batch_size=batch_size, device=self.device, sliding_window=0,
                params={'extend_backward': self.config.extend_backward, 'extend_forward': self.config.extend_forward})

        elif self.config.model_name == 'basernn':
            model = BaseDKT(
                self.config,
                self.device, self.config.model_name, n_input, n_hidden, n_output, n_layers, batch_size
            ).to(self.device)
            loss_batch = get_loss_batch_basedkt(
                onehot_size, n_input, batch_size, self.config.sequence_size, self.device)
            train_dl, eval_dl = prepare_data(
                self.config.source_data, 'base', n_skills, preserved_tokens='?',
                min_n=3, max_n=self.config.sequence_size, batch_size=batch_size, device=self.device, sliding_window=0)
        elif self.config.model_name == 'baselstm':
            model = BaseDKT(
                self.config,
                self.device, self.config.model_name, n_input, n_hidden, n_output, n_layers, batch_size
            ).to(self.device)
            loss_batch = get_loss_batch_basedkt(
                n_skills, onehot_size, n_input, batch_size, self.config.sequence_size, self.device)
            train_dl, eval_dl = prepare_dataloader(self.config, device=self.device)
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

        return model, loss_batch, train_dl, eval_dl

    def get_opt(self, model):
        opt = torch.optim.SGD(model.parameters(), lr=self.config.lr)
        return opt

    def train_model(self):
        self.logger.info('Starting train')
        train_loss_list = []
        bset_eval_auc = 0.

        # start_time = time.time()
        for epoch in range(1, self.config.epoch_size + 1):
            self.model.train()

            current_epoch_train_loss = []
            for xseq, yseq in self.train_dl:
                loss = self.model.loss_batch(xseq, yseq, opt=self.opt)
                current_epoch_train_loss.append(loss.item())

                # stop at first batch if debug
                if self.config.debug:
                    break

            if epoch % 10 == 0:
                loss_array = np.array(current_epoch_train_loss)
                if epoch % 100 == 0:
                    self.logger.info('TRAIN Epoch: {} Loss: {}'.format(
                        epoch, loss_array.mean()))
                train_loss_list.append(loss.mean())

    def valid_model(self):
        with torch.no_grad():
            self.model.eval()

            val_pred = []
            val_actual = []
            current_eval_loss = []
            for args in self.eval_dl:
                loss = self.loss_batch(
                    self.model, *args, opt=None)
                current_eval_loss.append(loss.item())
                val_pred.append(pred)
                val_actual.append(actu)

                # stop at first batch if debug
                if self.config.debug:
                    break

            loss = np.array(current_eval_loss)
            if epoch % 100 == 0:
                logger.log(logging.INFO,
                           'EVAL  Epoch: {} Loss: {}'.format(epoch,  loss.mean()))
            eval_loss_list.append(loss.mean())

            # AUC, Recall, F1
            # TODO: viewしない？　最後の1個で？
            y = torch.cat(val_actual).view(-1).cpu()
            pred = torch.cat(val_pred).view(-1).cpu()
            # AUC
            fpr, tpr, thresholds = metrics.roc_curve(
                y, pred, pos_label=1)
            logger.log(logging.INFO,
                        'EVAL  Epoch: {} AUC: {}'.format(epoch, metrics.auc(fpr, tpr)))
            auc = metrics.auc(fpr, tpr)
            eval_auc_list.append(auc)

            save_model(config, model, auc,epoch)
            save_log(
                config,
                (x, train_loss_list, train_auc_list,
                    eval_loss_list, eval_auc_list),
                auc, epoch
            )
            if auc > bset_eval_auc:
                bset_eval_auc = auc
                report['best_eval_auc'] = bset_eval_auc
                report['best_eval_auc_epoch'] = epoch

    #       if epoch % 10 == 0:
    #           x.append(epoch)
    #           if epoch % 100 == 0:
    #               logger.log(logging.INFO,
    #                          f'{timeSince(start_time, epoch / config.epoch_size)} ({epoch} {epoch / config.epoch_size * 100})')

    #   if config.plot_lc:
    #       fname = model_fname
    #       save_learning_curve(x, train_loss_list, train_auc_list,
    #                           eval_loss_list, eval_auc_list, config)