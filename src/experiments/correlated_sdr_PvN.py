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
from src.data.binary import generate_correlated_SDR_patterns




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
    def __init__(self, start, inc_step=2, th=0.01):
        self.lower_bound = start
        self.upper_bound = None
        self.pointer = start
        self.value = 0.0

        self.inc_step = inc_step
        self.th = th

        self.checked = set()

    def get_point(self):
        if self.value < self.th:
            self.lower_bound = self.pointer

            if self.upper_bound != None:
                self.pointer = (self.pointer + self.upper_bound)//2
            else:
                self.pointer = self.pointer+ self.inc_step

            final = self.pointer.item() in self.checked
            self.checked.add(self.pointer.item())

            return self.pointer, final

        elif self.value >= self.th:
            self.upper_bound = self.pointer

            self.pointer = (self.lower_bound + self.pointer) // 2

            final = self.pointer.item() in self.checked
            self.checked.add(self.pointer.item())

            return self.pointer, final





def search_Pmax(Ns, tolerance, model, learn_iters, lr, b, seeds):
    
    # number of sweeps for each N and each P to reduce randomness
    Pmaxs = []


    for N in Ns:
        print('==========================')

        # set an initial value for Pmax so we can detect if the upper bound of P is exceeded
        losses_N = []
        start_val = 2 if len(Pmaxs) ==0 else Pmaxs[-1]
        finder = Finder(start_val, inc_step=10*(N//10), th=0.01)
        P, final = finder.get_point()

        while True:

            print(f'Current Model:{model}, Current N:{N}, Current P:{P}')
            n_errors = 0
            for seed in tqdm(range(seeds)):

                # generate data, couple seed with k
                S = get_S(N)
                # S = N//2
                X = generate_correlated_SDR_patterns(P, N, b, S, seed=seed)

                if model.startswith('PAM'):
                    net = PamModel(N, int(model.split('-')[-1]), S, 0.9, 0.0, 1.0, 100)
                    net.train_seq(X)
                    recall = net.recall_seq(X, query='offline')

                    n_errors += get_n_errors(X[1:], recall[:-1])

                elif model.startswith('PC'):
                    X_polar = torch.stack([x.bin.float() for x in X])*2.0-1.0
                    net = SingleLayertPC(N, lr, 'binary', learn_iters)
                    losses = net.train_seq(X_polar, verbose=False)
                    recall = net.recall_seq(X_polar, query='offline')

                    n_errors += torch.sum(X_polar[1:]!=recall[1:])

                elif model.startswith('HN'):
                    X_polar = torch.stack([x.bin.float() for x in X])*2.0-1.0
                    net = ModernAsymmetricHopfieldNetwork(N, 'binary', sep=int(model.split('-')[-1]))
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

    save_base_dir = f'results/{os.path.splitext(os.path.basename(__file__))[0]}/run_001'
    assert checkdir(save_base_dir, careful=True), f'path {save_base_dir} exists'

    tolerance = 0.01
    models = ['PAM-8', 'PC', 'HN-1', 'HN-2']
    Ns = torch.arange(10, 110, 10)

    results = {}
    for i, model in enumerate(models):

        conf = dict(Ns = Ns, 
                    tolerance=tolerance, 
                    model=model, 
                    learn_iters=800, 
                    lr=1e-4, 
                    b=0.0,
                    seeds=10)
        Pmaxs = search_Pmax(**conf)
        results[models[i]] = Pmaxs


    print(results)
    json.dump(results, open(os.path.join(save_base_dir, 'results.json'), 'w'))

