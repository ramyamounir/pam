import torch
import os, json
import numpy as np
from tqdm import tqdm

import sys; sys.path.append('./')
from src.models.pam import PamModel
from src.utils.sdr import SDR
from src.utils.exps import checkdir, accuracy_SDR, accuracy_POLAR, set_seed
from src.utils.configs import get_pam_configs, get_pc_configs, get_hn_configs
from src.models.single_tpc import SingleLayertPC
from src.models.double_tpc import MultilayertPC
from src.models.mcahn import ModernAsymmetricHopfieldNetwork
from src.datasets.binary import generate_correlated_SDR_patterns


class Finder():
    def __init__(self, start, step=2, th=0.9):
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


        if self.value > self.th:
            self.lower_bound = self.pointer

            if self.upper_bound != None:
                self.pointer = (self.pointer + self.upper_bound)//2
            else:
                self.pointer = self.pointer+ self.step

            final = self.pointer in self.checked
            self.checked.add(self.pointer)

            return self.pointer, final

        elif self.value <= self.th:
            self.upper_bound = self.pointer

            if self.lower_bound != None:
                self.pointer = (self.lower_bound + self.pointer) // 2
            else:
                self.pointer = max(self.pointer-self.step,2)


            final = self.pointer in self.checked
            self.checked.add(self.pointer)

            return self.pointer, final




def search_Pmax(N, tolerance, model, start, specs, Bs, seed):
    
    # number of sweeps for each N and each P to reduce randomness
    Pmaxs = []


    # for N_ix, N in enumerate(Ns):
    for b_ix, b in enumerate(Bs):

        # set an initial value for Pmax so we can detect if the upper bound of P is exceeded
        losses_N = []
        # start_val = torch.tensor(start[N_ix]) if len(Pmaxs) ==0 else Pmaxs[-1]
        finder = Finder(start[b_ix], step=50, th=tolerance)
        P, final = finder.get_point()

        while True:

            # set seed for reproducibility
            set_seed(seed)

            # generate data, couple seed with k
            W = int(specs['W']*N) if specs['W_type']=='percentage' else specs['W']
            X = generate_correlated_SDR_patterns(P, N, b.item(), W)

            if model.startswith('PAM'):
                net = PamModel(N_c=N, N_k=specs['N_k'], W=W, **specs['configs'])
                net.learn_sequence(X)
                recall = net.generate_sequence_offline(X[0], len(X)-1)
                iou = accuracy_SDR(X[1:], recall[1:])


            elif model.startswith('PC'):
                X_polar = torch.stack([x.bin.float() for x in X])*2.0-1.0

                if specs['L'] == 1:
                    net = SingleLayertPC(input_size=N, **specs['configs'])
                elif specs['L'] == 2:
                    net = MultilayertPC(hidden_size=N*2, output_size=N, **specs['configs'])

                losses = net.train_seq(X_polar)
                recall = torch.sign(net.recall_seq(X_polar, query='offline'))
                iou = accuracy_POLAR(X_polar[1:], recall[1:])

            elif model.startswith('HN'):
                X_polar = torch.stack([x.bin.float() for x in X])*2.0-1.0

                net = ModernAsymmetricHopfieldNetwork(input_size=N, **specs['configs'])
                recall = net.recall_seq(X_polar, query='offline')
                iou = accuracy_POLAR(X_polar[1:], recall[1:])


            finder.value = iou
            P, final = finder.get_point()
            if final:
                Pmax = P
                break

        print(f'Seed:{seed}, Current Model:{model}, Current N:{N}, Current B:{b} Pmax:{Pmax}')

        # collect Pmax
        Pmaxs.append(int(Pmax))
    
    return Pmaxs

def main(save_base_dir, seed):

    tolerance = 0.9
    models = ['PAM-4', 'PAM-8', 'PC-1', 'HN-1-5', 'HN-1-50', 'HN-2-5', 'HN-2-50']

    bs = torch.arange(0.0, 0.6, 0.1)
    specs = {
            'PAM-4':{'N_k':4, 'W':5, 'W_type':'fixed', 'configs': get_pam_configs()},
            'PAM-8':{'N_k':8, 'W':5, 'W_type':'fixed', 'configs': get_pam_configs()},
            'PC-1':{'L':1, 'W':5, 'W_type':'fixed', 'configs': get_pc_configs(l=1)},
            'PC-2':{'L':2, 'W':5, 'W_type':'fixed', 'configs': get_pc_configs(l=2)},
            'HN-1-5': {'W':5, 'W_type':'fixed', 'configs': get_hn_configs(d=1)},
            'HN-2-5': {'W':5, 'W_type':'fixed', 'configs': get_hn_configs(d=2)},
            'HN-1-50': {'W':0.5, 'W_type':'percentage', 'configs': get_hn_configs(d=1)},
            'HN-2-50': {'W':0.5, 'W_type':'percentage', 'configs': get_hn_configs(d=2)},
             }

    starts = {
              'PAM-4':[500, 480, 460, 440, 420, 400],
              'PAM-8':[1000, 950, 900, 850, 800, 750],
              'PC-1':[90, 90, 90, 90, 90, 90],
              'PC-2':[90, 90, 90, 90, 90, 90],
              'HN-1-5':[2, 2, 2, 2, 2, 2],
              'HN-2-5':[2, 2, 2, 2, 2, 2],
              'HN-1-50':[50, 50, 50, 50, 50, 50],
              'HN-2-50':[600, 600, 600, 600, 600, 600],
              }


    results = {}
    for i, model in enumerate(models):

        conf = dict(N = 100, 
                    tolerance=tolerance, 
                    model=model, 
                    start=starts[model],
                    specs=specs[model],
                    Bs = bs,
                    seed=seed)
        Pmaxs = search_Pmax(**conf)
        results[models[i]] = Pmaxs


    print(results)
    json.dump(results, open(os.path.join(save_base_dir, f'results_{str(seed).zfill(3)}.json'), 'w'))
    args = dict(tolerance = tolerance, models = models, N=conf['N'], Bs=bs, starts=starts, specs=specs, seed=conf['seed'])
    torch.save(dict(args=args, results=results), os.path.join(save_base_dir, f'results_{str(seed).zfill(3)}.pth'))



if __name__ == "__main__":

    save_base_dir = f'results/{os.path.splitext(os.path.basename(__file__))[0]}/run_002'
    assert checkdir(save_base_dir, careful=False), f'path {save_base_dir} exists'

    for i in range(10):
        main(save_base_dir=save_base_dir, seed=i)


