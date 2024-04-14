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
from src.datasets.binary import generate_correlated_SDR_patterns, add_noise_SDR_patterns


def compute(N, P, model, specs, Es, seed):
    
    # learn the sequence
    set_seed(seed)

    # generate data, couple seed with k
    W = int(specs['W']*N) if specs['W_type']=='percentage' else specs['W']
    X = generate_correlated_SDR_patterns(P, N, specs['b'], W)

    if model.startswith('PAM'):
        net = PamModel(N_c=N, N_k=specs['N_k'], W=W, **specs['configs'])
        net.learn_sequence(X)

    elif model.startswith('PC'):
        X_polar = torch.stack([x.bin.float() for x in X])*2.0-1.0

        if specs['L'] == 1:
            net = SingleLayertPC(input_size=N, **specs['configs'])
        elif specs['L'] == 2:
            net = MultilayertPC(hidden_size=N*2, output_size=N, **specs['configs'])

        losses = net.train_seq(X_polar)

    elif model.startswith('HN'):
        X_polar = torch.stack([x.bin.float() for x in X])*2.0-1.0

        net = ModernAsymmetricHopfieldNetwork(input_size=N, **specs['configs'])


    ious = []

    for e_ix, e in enumerate(Es):

        # set seed for reproducibility
        set_seed(seed)

        # generate noisy sequence
        X_e = add_noise_SDR_patterns(X, int(e*W))

        if model.startswith('PAM'):
            recall = net.generate_sequence_online(X_e)
            ious.append(accuracy_SDR(X[1:], recall[1:]).item())

        elif model.startswith('PC'):
            X_e_polar = torch.stack([x.bin.float() for x in X_e])*2.0-1.0
            recall = torch.sign(net.recall_seq(X_e_polar, query='online'))
            ious.append(accuracy_POLAR(X_polar[1:], recall[1:]).item())

        elif model.startswith('HN'):
            X_e_polar = torch.stack([x.bin.float() for x in X_e])*2.0-1.0
            recall = net.recall_seq(X_e_polar, query='online')
            ious.append(accuracy_POLAR(X_polar[1:], recall[1:]).item())

        # print(f'Seed:{seed}, Current Model:{model}, Current N:{N}, iou:{ious}')
    
    return ious


def main(save_base_dir, P, b, seed):

    models = ['PAM-4', 'PAM-8', 'PC-1', 'PC-2', 'HN-1-50', 'HN-2-50']

    es = torch.tensor([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    specs = {
            'PAM-4':{'N_k':4, 'b':b, 'W':5, 'W_type':'fixed', 'configs': get_pam_configs()},
            'PAM-8':{'N_k':8, 'b':b, 'W':5, 'W_type':'fixed', 'configs': get_pam_configs()},
            'PAM-16':{'N_k':16, 'b':b, 'W':5, 'W_type':'fixed', 'configs': get_pam_configs()},
            'PC-1':{'L':1, 'b':b, 'W':5, 'W_type':'fixed', 'configs': get_pc_configs(l=1)},
            'PC-2':{'L':2, 'b':b, 'W':5, 'W_type':'fixed', 'configs': get_pc_configs(l=2)},
            'HN-1-5': {'W':5, 'b':b, 'W_type':'fixed', 'configs': get_hn_configs(d=1)},
            'HN-2-5': {'W':5, 'b':b, 'W_type':'fixed', 'configs': get_hn_configs(d=2)},
            'HN-1-50': {'W':0.5, 'b':b, 'W_type':'percentage', 'configs': get_hn_configs(d=1)},
            'HN-2-50': {'W':0.5, 'b':b, 'W_type':'percentage', 'configs': get_hn_configs(d=2)},
             }


    results = {}
    for i, model in enumerate(models):

        conf = dict(N = 100, 
                    P = P,
                    model=model, 
                    specs=specs[model],
                    Es = es,
                    seed=seed)
        ious = compute(**conf)
        results[models[i]] = ious

    print(results)
    json.dump(results, open(os.path.join(save_base_dir, f'results_{str(P)}_{str(b)}_{str(seed).zfill(3)}.json'), 'w'))
    args = dict(models = models, b=b, N=conf['N'], P=conf['P'], Es=es, specs=specs, seed=conf['seed'])
    torch.save(dict(args=args, results=results), os.path.join(save_base_dir, f'results_{str(P)}_{str(b)}_{str(seed).zfill(3)}.pth'))



if __name__ == "__main__":

    save_base_dir = f'results/{os.path.splitext(os.path.basename(__file__))[0]}/run_002'
    assert checkdir(save_base_dir, careful=False), f'path {save_base_dir} exists'

    for b in [0.0, 0.3, 0.5]:
        for P in [10, 100, 200]:
            for i in range(10):
                main(save_base_dir=save_base_dir, P=P, b=b, seed=i)


