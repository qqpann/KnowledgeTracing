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


def save_learning_curve(idclist_dic, config):
    lcdir = config.resultsdir / 'learning_curve'
    lcdir.mkdir(exist_ok=True)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = idclist_dic['epoch']
    for k in ['train_loss', 'train_auc', 'eval_loss', 'eval_auc']:
        ax.plot(x, idclist_dic[k], label=k.replace('_', ' '))
    ax.legend()
    ax.set_ylim(0., 1.)
    plt.savefig(lcdir / f'{config.model_name}_lc.png')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = idclist_dic['epoch']
    for k in ['ksvector_l1', 'waviness_l1', 'waviness_l2']:
        if len(x) == len(idclist_dic[k]):
            ax.plot(x, idclist_dic[k], label=k)
    ax.legend()
    # ax.set_ylim(.0, .1)
    plt.savefig(lcdir / f'{config.model_name}_loss.png')


def save_pred_accu_relation(config, x, y):
    pardir = config.resultsdir / 'pa_relation'
    pardir.mkdir(exist_ok=True)

    with open(pardir / f'{config.model_name}_xy.pkl', 'wb') as f:
        pickle.dump((x, y), f)

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    sns.scatterplot(x, y)
    fig.savefig(pardir / f'{config.model_name}_scatter.png')
