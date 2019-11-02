import numpy as np

class Trainer(object):
    def __init__(self, config, model, loss_batch, logger, loss, opt, train_dl):
        self.config = config
        self.model = model
        self.loss_batch = loss_batch
        self.logger = logger
        self.loss = loss
        self.opt = opt
        self.train_dl = train_dl

    def train_model(self):
        train_loss_list = []
        train_auc_list = []
        eval_loss_list = []
        eval_auc_list = []
        eval_recall_list = []
        eval_f1_list = []
        x = []
        bset_eval_auc = 0.

        # start_time = time.time()
        for epoch in range(1, self.config.epoch_size + 1):
            print_train = epoch % 10 == 0
            print_eval = epoch % 10 == 0
            print_auc = epoch % 10 == 0

            self.model.train()

            current_epoch_train_loss = []
            for args in self.train_dl:
                loss = self.loss_batch(self.model, self.loss, *args, opt=self.opt)
                current_epoch_train_loss.append(loss.item())

                # stop at first batch if debug
                if self.config.debug:
                    break

            if print_train:
                loss_array = np.array(current_epoch_train_loss)
                if epoch % 100 == 0:
                    self.logger.info('TRAIN Epoch: {} Loss: {}'.format(epoch, loss_array.mean()))
                train_loss_list.append(loss.mean())
