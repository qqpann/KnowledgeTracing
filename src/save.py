import torch

import json
import pickle
import matplotlib.pyplot as plt
import seaborn as sns


def save_model(config, model, auc, epoch):
    checkpointsdir = config.resultsdir / 'checkpoints'
    checkpointsdir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), checkpointsdir /
               f'{config.model_name}_auc{auc:.4f}_e{epoch}.model')


def save_log(config, data, auc, epoch):
    lc_datadir = config.resultsdir / 'lc_data'
    lc_datadir.mkdir(exist_ok=True)
    with open(lc_datadir / f'{config.model_name}_auc{auc:.4f}_e{epoch}.pickle', 'wb') as f:
        pickle.dump(data, f)


def save_report(config, report):
    lc_datadir = config.resultsdir / 'report'
    lc_datadir.mkdir(exist_ok=True)
    with open(lc_datadir / f'{config.model_name}.json', 'w') as f:
        json.dump(report, f, indent=2)


def save_hm_fig(config, sns_fig):
    hmdir = config.resultsdir / 'heatmap'
    hmdir.mkdir(exist_ok=True)
    sns_fig.savefig(hmdir / f'{config.model_name}.png')


def save_learning_curve(x, train_loss_list, train_auc_list, eval_loss_list, eval_auc_list, config):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if train_loss_list:
        ax.plot(x, train_loss_list, label='train loss')
    if train_auc_list:
        ax.plot(x, train_auc_list, label='train auc')
    if eval_loss_list:
        ax.plot(x, eval_loss_list, label='eval loss')
    ax.plot(x, eval_auc_list, label='eval auc')
    ax.legend()
    ax.set_ylim(0., 1.)
    lcdir = config.resultsdir / 'learning_curve'
    lcdir.mkdir(exist_ok=True)
    plt.savefig(lcdir / f'{config.model_name}.png')


def save_pred_accu_relation(config, x, y):
    pardir = config.resultsdir / 'pa_relation'
    pardir.mkdir(exist_ok=True)

    with open(pardir / f'{config.model_name}_xy.pkl', 'wb') as f:
        pickle.dump((x, y), f)

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    sns.scatterplot(x, y)
    fig.savefig(pardir / f'{config.model_name}_scatter.png')
