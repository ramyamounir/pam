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
from src.data.cifar import load_sequence_cifar
from src.data.binary import generate_correlated_SDR_patterns, add_noise_SDR_patterns


def get_n_errors(gt, rec):
    gts = torch.cat([g.bin.float() for g in gt])
    recs = torch.cat([r.bin.float() for r in rec])
    n_errors = torch.sum(gts!=recs)
    return n_errors


class Finder():
    def __init__(self, start, step=2, th=0.01):
        self.start = start
        self.lower_bound = None
        self.upper_bound = None
        self.pointer = start
        self.value = None

        self.step = step
        self.th = th

        self.checked = set()

    def get_point(self):
        if self.value == None:
            self.checked.add(self.pointer)
            return self.start, False


        if self.value < self.th:
            self.lower_bound = self.pointer

            if self.upper_bound != None:
                self.pointer = (self.pointer + self.upper_bound)//2
            else:
                self.pointer = self.pointer+ self.step

            final = self.pointer in self.checked
            self.checked.add(self.pointer)

            return self.pointer, final

        elif self.value >= self.th:
            self.upper_bound = self.pointer

            if self.lower_bound != None:
                self.pointer = (self.lower_bound + self.pointer) // 2
            else:
                self.pointer = max(self.pointer-self.step,2)


            final = self.pointer in self.checked
            self.checked.add(self.pointer)

            return self.pointer, final




def search_Pmax(N, P, model, specs, es, seeds):

    # generate data, couple seed with k
    S = 5 if specs['S']==5 else int(N*specs['S']/100)
    X = generate_correlated_SDR_patterns(P, N, specs['b'], S, seed=0)
    if model.startswith('PAM'):
        net = PamModel(N, specs['K'], S, specs['conn_den'], 0.0, 1.0, 1000, True)
        net.train_seq(X)

    elif model.startswith('PC'):
        X_polar = torch.stack([x.bin.float() for x in X])*2.0-1.0
        net = SingleLayertPC(N, specs['lr'], 'binary', specs['learn_iters'])
        losses = net.train_seq(X_polar, verbose=False)

    elif model.startswith('HN'):
        X_polar = torch.stack([x.bin.float() for x in X])*2.0-1.0
        net = ModernAsymmetricHopfieldNetwork(N, 'binary', sep=int(specs['D']))

    
    # number of sweeps for each N and each P to reduce randomness
    errors = []
    for e in es:

        e = int(e*(S/5.0))

        n_errors = 0
        for seed in tqdm(range(seeds)):

            # add noise to original sequence
            X_orig_polar = torch.stack([x.bin.float() for x in X])*2.0-1.0
            X_e = add_noise_SDR_patterns(X, e)

            if model.startswith('PAM'):
                recall = net.recall_generative(X_e)
                n_errors += get_n_errors(X[1:], recall)

            elif model.startswith('PC'):
                X_polar = torch.stack([x.bin.float() for x in X_e])*2.0-1.0
                recall = net.recall_seq(X_polar, query='online')
                n_errors += torch.sum(X_orig_polar[1:]!=recall[1:])

            elif model.startswith('HN'):
                X_polar = torch.stack([x.bin.float() for x in X_e])*2.0-1.0
                recall = net.recall_seq(X_polar, query='online')
                n_errors += torch.sum(X_orig_polar[1:]!=recall[1:])

        error_prob = n_errors / ((P - 1) * N * seeds)
        errors.append(error_prob.item())
        print(error_prob)

    return errors




if __name__ == "__main__":

    save_base_dir = f'results/{os.path.splitext(os.path.basename(__file__))[0]}/run_003'
    assert checkdir(save_base_dir, careful=False), f'path {save_base_dir} exists'

    # models = ['PAM-8', 'PAM-16', 'PAM-24', 'PC-1', 'HN-1-5', 'HN-1-50', 'HN-2-5', 'HN-2-50']
    models = ['PAM-8', 'PAM-16', 'PAM-24', 'PC-1']
    # models = ['HN-1-5', 'HN-1-50', 'HN-2-5', 'HN-2-50']
    es = [0, 1, 2, 3, 4, 5]

    specs = {
            'PAM-8':{'K':8, 'S':5, 'b':0.0, 'conn_den':1.0},
            'PAM-16':{'K':16, 'S':5, 'b':0.0, 'conn_den':1.0},
            'PAM-24':{'K':24, 'S':5, 'b':0.0, 'conn_den':1.0},
            'PC-1':{'L':1, 'S':5, 'b':0.0, 'lr':1e-4, 'learn_iters':800},
            'HN-1-5': {'D':1, 'S':5, 'b':0.0},
            'HN-1-50': {'D':1, 'S':50, 'b':0.0},
            'HN-2-5': {'D':2, 'S':5, 'b':0.0},
            'HN-2-50': {'D':2, 'S':50, 'b':0.0},
             }


    results = {}
    for i, model in enumerate(models):

        conf = dict(N = 100, 
                    P = 10,
                    model=model, 
                    specs = specs[model],
                    es = es,
                    seeds=10)
        errors = search_Pmax(**conf)
        results[models[i]] = errors


    print(results)
    json.dump(results, open(os.path.join(save_base_dir, 'results.json'), 'w'))

    args = dict(models = models, es=es, specs=specs, N=conf['N'], P=conf['P'], seeds=conf['seeds'])
    torch.save(dict(args=args, results=results), os.path.join(save_base_dir, 'results.pth'))

