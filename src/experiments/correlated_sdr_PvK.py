import torch
import os, json
import numpy as np
from tqdm import tqdm
import dask

import sys; sys.path.append('./')
from src.models.single_tpc import SingleLayertPC
from src.models.double_tpc import MultilayertPC
from src.models.mcahn import ModernAsymmetricHopfieldNetwork
from src.models.pam import PamModel
from src.models.pam_utils import SDR
from src.experiments.utils import checkdir
from src.data.cifar import load_sequence_cifar
from src.data.binary import generate_correlated_SDR_patterns




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




def search_Pmax(Ns, tolerance, model, starts, specs, seeds):

    # number of sweeps for each N and each P to reduce randomness
    Pmaxs = []
    for N_ix, N in enumerate(Ns):
        print('==========================')

        # set an initial value for Pmax so we can detect if the upper bound of P is exceeded
        losses_N = []
        # start_val = 2 if len(Pmaxs) ==0 else Pmaxs[-1]
        finder = Finder(starts[N_ix], step=10, th=0.01)
        P, final = finder.get_point()


        while True:

            print(f'Current Model:{model}, Current N:{N}, Current P:{P}')
            n_errors = 0
            for seed in tqdm(range(seeds)):

                # generate data, couple seed with k
                S = 5 if specs['S']==5 else int(N*specs['S']/100)
                X = generate_correlated_SDR_patterns(P, N, specs['b'], S, seed=seed)

                if model.startswith('PAM'):
                    net = PamModel(N, specs['K'], S, specs['conn_den'], 0.0, 1.0, 100, False)
                    net.train_seq(X)
                    recall = net.recall_predictive(X)

                    n_errors += get_n_errors(X[1:], recall)

                elif model.startswith('HN'):
                    X_polar = torch.stack([x.bin.float() for x in X])*2.0-1.0
                    net = ModernAsymmetricHopfieldNetwork(N, 'binary', sep=int(specs['D']))
                    recall = net.recall_seq(X_polar, query='online')

                    n_errors += torch.sum(X_polar[1:]!=recall[1:])
                else:
                    raise ValueError("Model not found")


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


def get_starts():

    def try_PAM(P, N, K):
        X = generate_correlated_SDR_patterns(P, N, 0.0, 5, seed=1)
        net = PamModel(N, K, 5, 0.8, 0.0, 1.0, 100, False)
        net.train_seq(X)
        recall = net.recall_seq(X, query='offline')
        return get_n_errors(X[1:], recall[:-1]) / ((P - 1) * N)

    N = 50
    K = 24
    jobs = []
    for p in [1800, 2000, 2100, 2300]:
        job = dask.delayed(try_PAM)(p, N, K)
        jobs.append(job)

    print(dask.compute(*jobs))



if __name__ == "__main__":


    save_base_dir = f'results/{os.path.splitext(os.path.basename(__file__))[0]}/run_003'
    assert checkdir(save_base_dir, careful=False), f'path {save_base_dir} exists'

    tolerance = 0.01
    models =  ['PAM-8', 'PAM-16', 'PAM-24', 'HN-2-5', 'HN-2-50']
    Ns = [10, 20, 30, 40]

    specs = {
            'PAM-8':{'K':8, 'S':5, 'b':0.0, 'conn_den':0.8},
            'PAM-16':{'K':16, 'S':5, 'b':0.0, 'conn_den':0.8},
            'PAM-24':{'K':24, 'S':5, 'b':0.0, 'conn_den':0.8},
            'HN-2-5': {'D':2, 'S':5, 'b':0.0},
            'HN-2-50': {'D':2, 'S':50, 'b':0.0},
             }


    starts = {
            'PAM-8':[7, 23, 82, 191],
            'PAM-16':[8, 79, 289, 575],
            'PAM-24':[20, 190, 525, 1200],
            'HN-2-5':[9, 9, 9, 9],
            'HN-2-50':[9, 27, 53, 111],
            }


    results = {}
    for i, model in enumerate(models):

        conf = dict(Ns = Ns, 
                    tolerance=tolerance, 
                    model=model, 
                    starts=starts[model],
                    specs = specs[model],
                    seeds=10)
        Pmaxs = search_Pmax(**conf)
        results[models[i]] = Pmaxs


    print(results)
    json.dump(results, open(os.path.join(save_base_dir, 'results.json'), 'w'))

    args = dict(tolerance = tolerance, models = models, Ns=Ns, starts=starts, specs=specs, seeds=conf['seeds'])
    torch.save(dict(args=args, results=results), os.path.join(save_base_dir, 'results.pth'))


