import json
import pickle

import matplotlib.pyplot as plt
import seaborn as sns
import torch


def save_model(config, model, fname):
    outdir = config.resultsdir / 'checkpoints' / config.starttime
    outdir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), outdir / fname)


def save_log(config, data, auc, epoch):
    outdir = config.resultsdir / 'lc_data' / config.starttime
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / f'{config.model_name}_auc{auc:.4f}_e{epoch}.pickle', 'wb') as f:
        pickle.dump(data, f)


def save_report(config, report):
    outdir = config.resultsdir / 'report' / config.starttime
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / 'report.json', 'w') as f:
        json.dump(report, f, indent=2)


def save_hm_fig(config, sns_fig):
    outdir = config.resultsdir / 'heatmap' / config.starttime
    outdir.mkdir(parents=True, exist_ok=True)
    sns_fig.savefig(outdir / f'{config.model_name}.png')


def save_learning_curve(idclist_dic, config):
    outdir = config.resultsdir / 'learning_curve' / config.starttime
    outdir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = idclist_dic['epoch']
    for k in ['train_loss', 'train_auc', 'eval_loss', 'eval_auc']:
        ax.plot(x, idclist_dic[k], label=k.replace('_', ' '))
    ax.legend()
    ax.set_ylim(0., 1.)
    plt.savefig(outdir / f'{config.model_name}_lc.png')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = idclist_dic['epoch']
    for k in ['ksvector_l1', 'waviness_l1', 'waviness_l2']:
        if len(x) == len(idclist_dic[k]):
            ax.plot(x, idclist_dic[k], label=k)
    ax.legend()
    # ax.set_ylim(.0, .1)
    plt.savefig(outdir / f'{config.model_name}_loss.png')


def save_pred_accu_relation(config, x, y):
    outdir = config.resultsdir / 'pa_relation' / config.starttime
    outdir.mkdir(parents=True, exist_ok=True)

    with open(outdir / f'{config.model_name}_xy.pkl', 'wb') as f:
        pickle.dump((x, y), f)

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    sns.scatterplot(x, y)
    fig.savefig(outdir / f'{config.model_name}_scatter.png')
