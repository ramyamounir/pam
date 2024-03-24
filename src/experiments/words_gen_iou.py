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
from src.datasets.words import WordSequence

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




def get_model(model, N, specs):

    if model.startswith('PAM'):
        net = PamModel(N, specs['K'], specs['S'], specs['conn_den'], 0.0, 1.0, 100, True)

    elif model.startswith('PC'):

        if specs['L'] == 1:
            net = SingleLayertPC(N, specs['lr'], 'binary',specs['learn_iters'])
        elif specs['L'] == 2:
            net = MultilayertPC(N*2, N, specs['lr'], specs['inf_lr'], specs['learn_iters'], specs['inf_iters'])

    elif model.startswith('HN'):
        net = ModernAsymmetricHopfieldNetwork(N, 'binary', sep=int(specs['D']))

    return net


def train_model(seqs, net):

    if isinstance(net, PamModel):
        for i in range(len(seqs)):
            net.train_seq(seqs[i])

    elif isinstance(net, SingleLayertPC):

        X_polar = []
        for seq in seqs:
            X_polar.append(torch.stack([x.bin.float() for x in seq])*2.0-1.0)
        X_polar = torch.stack(X_polar)

        net.train_batched_seqs(X_polar)

    elif isinstance(net, MultilayertPC):
        X_polar = []
        for seq in seqs:
            X_polar.append(torch.stack([x.bin.float() for x in seq])*2.0-1.0)
        X_polar = torch.stack(X_polar)
        print(X_polar.shape)
        quit()

        net.train_seqs(X_polar)

    elif isinstance(net, ModernAsymmetricHopfieldNetwork):
        pass

    else:
        raise ValueError("model not found")


def gen_words(seqs, net):
    if isinstance(net, PamModel):
        return [net.recall_generative_random(seq[0], 3) for seq in seqs]

    elif isinstance(net, SingleLayertPC):
        X_polar = []
        for seq in seqs:
            X_polar.append(torch.stack([x.bin.float() for x in seq])*2.0-1.0)
        X_polar = torch.stack(X_polar)

        return [net.recall_seq(seq, query='offline') for seq in X_polar]

    elif isinstance(net, MultilayertPC):
        pass


    elif isinstance(net, ModernAsymmetricHopfieldNetwork):
        X_polar = []
        for seq in seqs:
            X_polar.append(torch.stack([x.bin.float() for x in seq])*2.0-1.0)
        X_polar = torch.stack(X_polar)

        return [net.recall_seqs(seq, X_polar, query='offline') for seq in X_polar]


def evaluate(net, words, seqs):

    if isinstance(net, PamModel):
        ious = []
        for word in words:
            iou = max([get_norm_ious(gt[1:], word[1:]) for gt in seqs])
            ious.append(iou)
        return sum(ious)/len(ious)

    elif isinstance(net, SingleLayertPC):

        X_polar = []
        for seq in seqs:
            X_polar.append(torch.stack([x.bin.float() for x in seq])*2.0-1.0)
        X_polar = torch.stack(X_polar)

        ious = []
        for word in words:
            # iou = max([get_ious(gt[1:], word[1:]) for gt in X_polar])
            iou = max([get_norm_ious(gt[1:], word[1:]) for gt in X_polar])
            ious.append(iou)
        return sum(ious)/len(ious)

    elif isinstance(net, MultilayertPC):
        pass

    elif isinstance(net, ModernAsymmetricHopfieldNetwork):

        X_polar = []
        for seq in seqs:
            X_polar.append(torch.stack([x.bin.float() for x in seq])*2.0-1.0)
        X_polar = torch.stack(X_polar)

        ious = []
        for word in words:
            iou = max([get_norm_ious(gt[1:], word[1:]) for gt in X_polar])
            ious.append(iou)

        return sum(ious)/len(ious)



def compute(N, model, num_words_inc, specs, seed=1):

    S = 5 if specs['S']==5 else int(N*specs['S']/100)

    ious = []
    for num_words in num_words_inc:
        X = WordSequence(N, S, num_words=num_words, seed=seed)
        net = get_model(model, N, specs)

        train_model(X.seqs, net)
        generated_words = gen_words(X.seqs, net)
        ious.append(evaluate(net, generated_words, X.seqs))

    return ious




if __name__ == "__main__":

    save_base_dir = f'results/{os.path.splitext(os.path.basename(__file__))[0]}/run_002'
    assert checkdir(save_base_dir, careful=False), f'path {save_base_dir} exists'

    models = ['PAM-1', 'PAM-8', 'PAM-16', 'PAM-24', 'PC-1', 'HN-1-5', 'HN-1-50', 'HN-2-5', 'HN-2-50']
    num_words_inc = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
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
                    num_words_inc=num_words_inc,
                    specs=specs[model],
                    seed=1)

        model_result = compute(**conf)
        results[models[i]] = model_result
        print(results[models[i]])

    print(results)
    json.dump(results, open(os.path.join(save_base_dir, 'results.json'), 'w'))

    args = dict(models=models, num_words_inc=num_words_inc, N=conf['N'], specs=specs)
    torch.save(dict(args=args, results=results), os.path.join(save_base_dir, 'results.pth'))

