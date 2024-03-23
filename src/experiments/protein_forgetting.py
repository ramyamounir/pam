import torch
import os, json
import numpy as np
from tqdm import tqdm

import sys; sys.path.append('./')
from src.models.single_tpc import SingleLayertPC
from src.models.double_tpc import MultilayertPC
from src.models.mcahn import ModernAsymmetricHopfieldNetwork
from src.models.pam import PamModel
from src.models.pam_utils import SDR
from src.experiments.utils import checkdir
from src.datasets.protein import ProteinSequence

def get_ious(gt, rec):
    if isinstance(rec[0], SDR):
        return torch.mean(torch.tensor([g.iou(r) for g, r in zip(gt, rec)])).item()
    else:
        return torch.mean(torch.tensor([torch.sum(torch.logical_and(g,r)).item()/torch.sum(torch.logical_or(g,r)).item() for g,r in zip((gt+1.0)/2.0, (rec+1.0)/2.0)])).item()


def get_norm_ious(gt, rec):
    if isinstance(rec[0], SDR):

        norm_ious = []
        for g, r in zip(gt, rec):
            g_density = g.S/g.N
            r_density = r.S/r.N
            expected_iou = (g_density*r_density)/(g_density+r_density-(g_density*r_density))
            iou = g.iou(r)
            norm_iou = iou - expected_iou/(1.0 - expected_iou)
            norm_ious.append(norm_iou.item())
        return torch.mean(torch.tensor(norm_ious)).item()


    else:

        norm_ious = []
        for g, r in zip((gt+1.0)/2.0, (rec+1.0)/2.0):

            g_density = sum(g)/len(g)
            r_density = sum(r)/len(r)
            expected_iou = (g_density*r_density)/(g_density+r_density-(g_density*r_density))
            iou = torch.sum(torch.logical_and(g, r))/torch.sum(torch.logical_or(g, r))
            norm_iou = iou - expected_iou/(1.0 - expected_iou)
            norm_ious.append(norm_iou.item())
        return torch.mean(torch.tensor(norm_ious)).item()

def get_n_errors(gt, rec):
    if isinstance(rec[0], SDR):
        gts = torch.cat([g.bin.float() for g in gt])
        recs = torch.cat([r.bin.float() for r in rec])
        n_errors = torch.sum(gts!=recs)
        error_prob = n_errors/torch.numel(gts)
        return error_prob
    else:
        return torch.sum(gt!=rec)/torch.numel(gt)


def get_model(model, N, specs):

    if model.startswith('PAM'):
        net = PamModel(N, specs['K'], specs['S'], specs['conn_den'], 0.0, 1.0, 100, False)

    elif model.startswith('PC'):

        if specs['L'] == 1:
            net = SingleLayertPC(N, specs['lr'], 'binary',specs['learn_iters'])
        elif specs['L'] == 2:
            net = MultilayertPC(N*2, N, specs['lr'], specs['inf_lr'], specs['learn_iters'], specs['inf_iters'])

    elif model.startswith('HN'):
        net = ModernAsymmetricHopfieldNetwork(N, 'binary', sep=int(specs['D']))

    return net

def train_model(seq, net):

    if isinstance(net, PamModel):
        net.train_seq(seq)
    elif isinstance(net, SingleLayertPC):
        X_polar = torch.stack([x.bin.float() for x in seq])*2.0-1.0
        net.train_seq(X_polar)
    elif isinstance(net, MultilayertPC):
        X_polar = torch.stack([x.bin.float() for x in seq])*2.0-1.0
        net.train_seq(X_polar)
    elif isinstance(net, ModernAsymmetricHopfieldNetwork):
        pass
    else:
        raise ValueError("model not found")


def backward_transfer(seqs, net):

    metrics = []
    for seq in seqs:

        if isinstance(net, PamModel):
            recall = net.recall_predictive(seq)
            # metric = get_n_errors(seq[1:], recall)
            # metric = get_ious(seq[1:], recall)
            metric = get_norm_ious(seq[1:], recall)
        elif isinstance(net, SingleLayertPC):
            X_polar = torch.stack([x.bin.float() for x in seq])*2.0-1.0
            recall = net.recall_seq(X_polar, 'online')
            # metric = get_n_errors(X_polar[1:], recall[1:])
            # metric = get_ious(X_polar[1:], recall[1:])
            metric = get_norm_ious(X_polar[1:], recall[1:])

        elif isinstance(net, MultilayertPC):
            X_polar = torch.stack([x.bin.float() for x in seq])*2.0-1.0
            recall = net.recall_seq(X_polar, 'online')
            # metric = get_n_errors(X_polar[1:], recall[1:])
            # metric = get_ious(X_polar[1:], recall[1:])
            metric = get_norm_ious(X_polar[1:], recall[1:])

        elif isinstance(net, ModernAsymmetricHopfieldNetwork):
            X_polar = torch.stack([x.bin.float() for x in seq])*2.0-1.0
            recall = net.recall_seq(X_polar, 'online')
            # metric = get_n_errors(X_polar[1:], recall[1:])
            # metric = get_ious(X_polar[1:], recall[1:])
            metric = get_norm_ious(X_polar[1:], recall[1:])

        metrics.append(metric)

    return metrics


def compute(N, model, specs, seed=1):

    S = 5 if specs['S']==5 else int(N*specs['S']/100)
    X = ProteinSequence(N, S, seed=seed)
    net = get_model(model, N, specs)

    backwards = []
    for Xi_id, Xi in enumerate(X.seqs[:10]):
        print(f'seq: {Xi_id}')
        train_model(Xi, net)
        backward_errors = backward_transfer(X.seqs[:Xi_id+1], net)
        backwards.append(backward_errors)

    return backwards




if __name__ == "__main__":

    save_base_dir = f'results/{os.path.splitext(os.path.basename(__file__))[0]}/run_001'
    assert checkdir(save_base_dir, careful=False), f'path {save_base_dir} exists'

    models = ['PAM-1', 'PAM-8', 'PAM-16', 'PAM-24', 'PC-1', 'HN-1-50', 'HN-2-50']
    # models = ['HN-2-5']
    specs = {
            'PAM-1':{'K':1, 'S':5, 'conn_den':0.8},
            'PAM-8':{'K':8, 'S':5, 'conn_den':0.8},
            'PAM-16':{'K':16, 'S':5, 'conn_den':0.8},
            'PAM-24':{'K':24, 'S':5, 'conn_den':0.8},
            'PC-1':{'L':1, 'S':5, 'lr':1e-4, 'learn_iters':800},
            'PC-2':{'L':2, 'S':5, 'lr':1e-4, 'learn_iters':800, 'inf_lr':1e-4, 'inf_iters': 800},
            'HN-1-5': {'D':1, 'S':5},
            'HN-2-5': {'D':2, 'S':5},
            'HN-1-50': {'D':1, 'S':50},
            'HN-2-50': {'D':2, 'S':50},
             }

    results = {}
    for i, model in enumerate(models):
        print(f'Model: {model}')

        conf = dict(N = 100, 
                    model=model, 
                    specs=specs[model],
                    seed=1)

        model_result = compute(**conf)
        results[models[i]] = model_result

    print(results)
    json.dump(results, open(os.path.join(save_base_dir, 'results.json'), 'w'))

    args = dict(models=models, N=conf['N'], specs=specs)
    torch.save(dict(args=args, results=results), os.path.join(save_base_dir, 'results.pth'))

