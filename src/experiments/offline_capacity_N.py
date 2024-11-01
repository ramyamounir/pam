import torch
import os, json
import numpy as np
from tqdm import tqdm

import sys; sys.path.append('./')
from src.models.pam import PamModel
from src.utils.sdr import SDR
from src.utils.exps import checkdir, accuracy_SDR, accuracy_POLAR, accuracy_BIN, set_seed
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




def search_Pmax(Ns, tolerance, model, start, specs, seed):
    
    # number of sweeps for each N and each P to reduce randomness
    Pmaxs = []


    for N_ix, N in enumerate(Ns):

        # set an initial value for Pmax so we can detect if the upper bound of P is exceeded
        finder = Finder(start[N_ix], step=10, th=tolerance)
        P, final = finder.get_point()

        while True:

            # set seed for reproducibility
            set_seed(seed)

            # generate data, couple seed with k
            W = int(specs['W']*N) if specs['W_type']=='percentage' else specs['W']
            X = generate_correlated_SDR_patterns(P, N, specs['b'], W)

            if model.startswith('PAM'):
                net = PamModel(N_c=N, N_k=specs['N_k'], W=W, **specs['configs'])
                net.learn_sequence(X)
                recall = net.generate_sequence_offline(X[0], len(X)-1)
                # recall = net.generate_sequence_online(X)
                iou = accuracy_SDR(X[1:], recall[1:])

            elif model.startswith('PC'):
                X_polar = torch.stack([x.bin.float() for x in X])*2.0-1.0

                if specs['L'] == 1:
                    net = SingleLayertPC(input_size=N, **specs['configs'])
                elif specs['L'] == 2:
                    net = MultilayertPC(hidden_size=N*2, output_size=N, **specs['configs'])

                losses = net.train_seq(X_polar)
                recall = torch.sign(net.recall_seq(X_polar, query='offline'))
                # recall = torch.sign(net.recall_seq(X_polar, query='online'))
                iou = accuracy_POLAR(X_polar[1:], recall[1:])

            elif model.startswith('HN'):
                X_polar = torch.stack([x.bin.float() for x in X])*2.0-1.0

                net = ModernAsymmetricHopfieldNetwork(input_size=N, **specs['configs'])
                recall = net.recall_seq(X_polar, query='offline')
                # recall = net.recall_seq(X_polar, query='online')
                iou = accuracy_POLAR(X_polar[1:], recall[1:])

            finder.value = iou
            P, final = finder.get_point()
            if final:
                Pmax = P
                break

        print(f'Seed:{seed}, Current Model:{model}, Current N:{N}, Pmax:{Pmax}')

        # collect Pmax
        Pmaxs.append(int(Pmax))
    
    return Pmaxs


def main(save_base_dir, seed):

    tolerance = 0.9
    models = ['PAM-4', 'PAM-8', 'PC-1', 'HN-1-5', 'HN-1-50', 'HN-2-5', 'HN-2-50']

    Ns = torch.arange(10, 110, 10)
    specs = {
            'PAM-4':{'N_k':4, 'b':0.0, 'W':5, 'W_type':'fixed', 'configs': get_pam_configs()},
            'PAM-8':{'N_k':8, 'b':0.0, 'W':5, 'W_type':'fixed', 'configs': get_pam_configs()},
            'PC-1':{'L':1, 'b':0.0, 'W':5, 'W_type':'fixed', 'configs': get_pc_configs(l=1)},
            'PC-2':{'L':2, 'b':0.0, 'W':5, 'W_type':'fixed', 'configs': get_pc_configs(l=2)},
            'HN-1-5': {'W':5, 'b':0.0, 'W_type':'fixed', 'configs': get_hn_configs(d=1)},
            'HN-2-5': {'W':5, 'b':0.0, 'W_type':'fixed', 'configs': get_hn_configs(d=2)},
            'HN-1-50': {'W':0.5, 'b':0.0, 'W_type':'percentage', 'configs': get_hn_configs(d=1)},
            'HN-2-50': {'W':0.5, 'b':0.0, 'W_type':'percentage', 'configs': get_hn_configs(d=2)},
             }

    starts = {
              'PAM-4':[15, 40, 50, 100, 150, 225, 250, 300, 400, 500],
              'PAM-8':[35, 100, 150, 200, 300, 450, 550, 700, 800, 1000],
              'PC-1':[4, 9, 20, 25, 37, 44, 51, 63, 75, 90],
              'PC-2':[4, 9, 20, 25, 37, 44, 51, 63, 75, 90],
              'HN-1-5':[2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
              'HN-2-5':[2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
              'HN-1-50':[5, 6, 8, 13, 15, 15, 20, 20, 20, 25],
              'HN-2-50':[10, 30, 60, 110, 150, 230 ,350, 400, 550, 675],
              }

    results = {}
    for i, model in enumerate(models):

        conf = dict(Ns = Ns, 
                    tolerance=tolerance, 
                    model=model, 
                    start=starts[model],
                    specs=specs[model],
                    seed=seed)
        Pmaxs = search_Pmax(**conf)
        results[models[i]] = Pmaxs


    print(results)
    json.dump(results, open(os.path.join(save_base_dir, f'results_{str(seed).zfill(3)}.json'), 'w'))
    args = dict(tolerance = tolerance, models = models, Ns=Ns, starts=starts, specs=specs, seed=conf['seed'])
    torch.save(dict(args=args, results=results), os.path.join(save_base_dir, f'results_{str(seed).zfill(3)}.pth'))



if __name__ == "__main__":

    save_base_dir = f'results/{os.path.splitext(os.path.basename(__file__))[0]}/run_002'
    assert checkdir(save_base_dir, careful=False), f'path {save_base_dir} exists'

    for i in range(10):
        main(save_base_dir=save_base_dir, seed=i)



