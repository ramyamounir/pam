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
from src.datasets.cifar import load_sequence_cifar
from src.datasets.binary import generate_correlated_SDR_patterns


def get_S(x):
    min_x, max_x, min_y, max_y = 128, 1024, 0.05, 0.1

    m = torch.tensor((min_y-max_y)/(max_x-min_x))
    c = max_y - (m*min_x)

    return torch.clamp(torch.tensor(int(x*(m*x+c))), 4, 50)

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




def search_Pmax(N, tolerance, model, starts, specs, bs, seeds):
    
    # number of sweeps for each N and each P to reduce randomness
    Pmaxs = []

    for b_ix, b in enumerate(bs):
        print('==========================')

        # set an initial value for Pmax so we can detect if the upper bound of P is exceeded
        losses_N = []
        finder = Finder(starts[b_ix], step=10, th=0.01)
        P, final = finder.get_point()

        while True:

            print(f'Current Model:{model}, Current N:{N}, Current b:{b}, Current P:{P}')
            n_errors = 0
            for seed in tqdm(range(seeds)):

                # generate data, couple seed with k
                S = 5 if specs['S']==5 else int(N*specs['S']/100)
                X = generate_correlated_SDR_patterns(P, N, b.item(), S, seed=seed)

                if model.startswith('PAM'):
                    net = PamModel(N, specs['K'], S, specs['conn_den'], 0.0, 1.0, 100, False)
                    net.train_seq(X)
                    recall = net.recall_predictive(X)

                    n_errors += get_n_errors(X[1:], recall)

                elif model.startswith('PC'):
                    X_polar = torch.stack([x.bin.float() for x in X])*2.0-1.0
                    net = SingleLayertPC(N, specs['lr'], 'binary', specs['learn_iters'])
                    losses = net.train_seq(X_polar)
                    recall = net.recall_seq(X_polar, query='online')

                    n_errors += torch.sum(X_polar[1:]!=recall[1:])

                elif model.startswith('HN'):
                    X_polar = torch.stack([x.bin.float() for x in X])*2.0-1.0
                    net = ModernAsymmetricHopfieldNetwork(N, 'binary', sep=int(specs['D']))
                    recall = net.recall_seq(X_polar, query='online')

                    n_errors += torch.sum(X_polar[1:]!=recall[1:])

            # compute the probability of errors as the percentage of mismatched bits across K sweeps
            error_prob = n_errors / ((P - 1) * N * seeds)
            print(f' error:{error_prob}')
            finder.value = error_prob

            P, final = finder.get_point()
            if final:
                Pmax = P
                break

        print(f'Pmax:{Pmax}')
        

        # collect Pmax
        Pmaxs.append(int(Pmax))
    
    return Pmaxs




if __name__ == "__main__":

    save_base_dir = f'results/{os.path.splitext(os.path.basename(__file__))[0]}/run_003'
    assert checkdir(save_base_dir, careful=False), f'path {save_base_dir} exists'

    tolerance = 0.01
    models = ['PAM-8', 'PC-1', 'HN-1-5', 'HN-1-50', 'HN-2-5', 'HN-2-50']
    bs = torch.arange(0.0, 0.6, 0.1)

    specs = {
            'PAM-8':{'K':8, 'S':5, 'b':0.0, 'conn_den':0.8},
            'PC-1':{'L':1, 'S':5, 'b':0.0, 'lr':1e-4, 'learn_iters':800},
            'HN-1-5': {'D':1, 'S':5, 'b':0.0},
            'HN-1-50': {'D':1, 'S':50, 'b':0.0},
            'HN-2-5': {'D':2, 'S':5, 'b':0.0},
            'HN-2-50': {'D':2, 'S':50, 'b':0.0},
             }


    starts = {
            'PAM-8':[1800, 1790, 1780, 1755, 1750, 1700],
            'PC-1':[100, 50, 30, 30, 30, 30],
            'HN-1-5':[ 3, 3, 3, 3, 3, 3],
            'HN-1-50':[ 18, 10, 6, 3, 3, 3],
            'HN-2-5':[ 3, 3, 3, 3, 3, 3],
            'HN-2-50':[ 595, 10, 6, 3, 3, 3],
            }


    results = {}
    for i, model in enumerate(models):

        conf = dict(N = 100, 
                    tolerance=tolerance, 
                    model=model, 
                    starts=starts[model],
                    specs = specs[model],
                    bs=bs,
                    seeds=10)
        Pmaxs = search_Pmax(**conf)
        results[models[i]] = Pmaxs


    print(results)
    json.dump(results, open(os.path.join(save_base_dir, 'results.json'), 'w'))

    args = dict(tolerance = tolerance, models = models, starts=starts, bs=bs, specs=specs, seeds=conf['seeds'])
    torch.save(dict(args=args, results=results), os.path.join(save_base_dir, 'results.pth'))

