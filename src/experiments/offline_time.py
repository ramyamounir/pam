import torch
import os, json, time
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




def compute(N, Ps, model, specs, seed):
    
    # number of sweeps for each N and each P to reduce randomness
    times = []


    for P_ix, P in enumerate(Ps):

        # set seed for reproducibility
        set_seed(seed)

        # generate data, couple seed with k
        W = int(specs['W']*N) if specs['W_type']=='percentage' else specs['W']
        X = generate_correlated_SDR_patterns(P, N, 0.0, W)

        if model.startswith('PAM'):
            time_start = time.time()

            net = PamModel(N_c=N, N_k=specs['N_k'], W=W, **specs['configs'])
            net.learn_sequence(X)
            recall = net.generate_sequence_offline(X[0], len(X)-1)

            time_end = time.time()
            times.append(time_end - time_start)

        elif model.startswith('PC'):
            X_polar = torch.stack([x.bin.float() for x in X])*2.0-1.0
            time_start = time.time()

            if specs['L'] == 1:
                net = SingleLayertPC(input_size=N, **specs['configs'])
            elif specs['L'] == 2:
                net = MultilayertPC(hidden_size=N*2, output_size=N, **specs['configs'])

            losses = net.train_seq(X_polar)
            recall = torch.sign(net.recall_seq(X_polar, query='offline'))

            time_end = time.time()
            times.append(time_end - time_start)


        elif model.startswith('HN'):
            X_polar = torch.stack([x.bin.float() for x in X])*2.0-1.0
            time_start = time.time()

            net = ModernAsymmetricHopfieldNetwork(input_size=N, **specs['configs'])
            recall = net.recall_seq(X_polar, query='online')

            time_end = time.time()
            times.append(time_end - time_start)

    return times


def main(save_base_dir, seed):

    models = ['PAM-1', 'PAM-4', 'PAM-8', 'PAM-16', 'PAM-24', 'PC-1', 'PC-2', 'HN-2-50']

    Ps = [10, 50, 100]
    specs = {
            'PAM-1':{'N_k':1, 'b':0.0, 'W':5, 'W_type':'fixed', 'configs': get_pam_configs()},
            'PAM-4':{'N_k':4, 'b':0.0, 'W':5, 'W_type':'fixed', 'configs': get_pam_configs()},
            'PAM-8':{'N_k':8, 'b':0.0, 'W':5, 'W_type':'fixed', 'configs': get_pam_configs()},
            'PAM-16':{'N_k':16, 'b':0.0, 'W':5, 'W_type':'fixed', 'configs': get_pam_configs()},
            'PAM-24':{'N_k':24, 'b':0.0, 'W':5, 'W_type':'fixed', 'configs': get_pam_configs()},
            'PC-1':{'L':1, 'b':0.0, 'W':5, 'W_type':'fixed', 'configs': get_pc_configs(l=1)},
            'PC-2':{'L':2, 'b':0.0, 'W':5, 'W_type':'fixed', 'configs': get_pc_configs(l=2)},
            'HN-1-5': {'W':5, 'b':0.0, 'W_type':'fixed', 'configs': get_hn_configs(d=1)},
            'HN-2-5': {'W':5, 'b':0.0, 'W_type':'fixed', 'configs': get_hn_configs(d=2)},
            'HN-1-50': {'W':0.5, 'b':0.0, 'W_type':'percentage', 'configs': get_hn_configs(d=1)},
            'HN-2-50': {'W':0.5, 'b':0.0, 'W_type':'percentage', 'configs': get_hn_configs(d=2)},
             }


    results = {}
    for i, model in enumerate(models):

        conf = dict(N = 100, 
                    Ps = Ps,
                    model=model, 
                    specs=specs[model],
                    seed=seed)
        times = compute(**conf)
        results[models[i]] = times
        print(models[i], results[models[i]])


    print(results)
    json.dump(results, open(os.path.join(save_base_dir, f'results_{str(seed).zfill(3)}.json'), 'w'))
    args = dict(models=models, Ps=Ps, specs=specs, seed=conf['seed'])
    torch.save(dict(args=args, results=results), os.path.join(save_base_dir, f'results_{str(seed).zfill(3)}.pth'))



if __name__ == "__main__":

    save_base_dir = f'results/{os.path.splitext(os.path.basename(__file__))[0]}/run_002'
    assert checkdir(save_base_dir, careful=False), f'path {save_base_dir} exists'

    main(save_base_dir=save_base_dir, seed=0)



