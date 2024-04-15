import torch
import os, json
import numpy as np
from tqdm import tqdm

import sys; sys.path.append('./')
from src.models.pam import PamModel
from src.models.single_tpc import SingleLayertPC
from src.models.double_tpc import MultilayertPC
from src.models.mcahn import ModernAsymmetricHopfieldNetwork
from src.utils.sdr import SDR
from src.utils.exps import checkdir, accuracy_SDR, accuracy_POLAR, set_seed
from src.utils.configs import get_pam_configs, get_pc_configs, get_hn_configs
from src.datasets.binary import generate_correlated_SDR_patterns, generate_multiple_correlated_SDR_patterns, add_noise_SDR_patterns


def get_model(model, N, W, specs):

    if model.startswith('PAM'):
        net = PamModel(N_c=N, N_k=specs['N_k'], W=W, **specs['configs'])

    elif model.startswith('PC'):

        if specs['L'] == 1:
            net = SingleLayertPC(input_size=N, **specs['configs'])
        elif specs['L'] == 2:
            net = MultilayertPC(hidden_size=N*2, output_size=N, **specs['configs'])

    elif model.startswith('HN'):
        net = ModernAsymmetricHopfieldNetwork(input_size=N, **specs['configs'])

    return net

def train_model(seq, net):

    if isinstance(net, PamModel):
        net.learn_sequence(seq)
    elif isinstance(net, SingleLayertPC) or isinstance(net, MultilayertPC):
        X_polar = torch.stack([x.bin.float() for x in seq])*2.0-1.0
        losses = net.train_seq(X_polar)
    elif isinstance(net, ModernAsymmetricHopfieldNetwork):
        pass
    else:
        raise ValueError("model not found")

def backward_transfer(seqs, net):

    metrics = []
    for seq_id, seq in enumerate(seqs):

        if isinstance(net, PamModel):
            recall = net.generate_sequence_offline(seq[0], len(seq)-1)
            metric = accuracy_SDR(seq[1:], recall[1:])
        elif isinstance(net, SingleLayertPC) or isinstance(net, MultilayertPC):
            X_polar = torch.stack([x.bin.float() for x in seq])*2.0-1.0
            recall = net.recall_seq(X_polar, 'offline')
            metric = accuracy_POLAR(X_polar[1:], recall[1:])
        elif isinstance(net, ModernAsymmetricHopfieldNetwork):
            X_polar = torch.stack([x.bin.float() for x in seq])*2.0-1.0
            recall = net.recall_seq(X_polar, 'offline')
            metric = accuracy_POLAR(X_polar[1:], recall[1:])

        metrics.append(metric)

    return metrics


def compute(N, P, num_seqs, model, bs, specs, seed=0):

    full_results = []
    for b in bs:
        W = int(specs['W']*N) if specs['W_type']=='percentage' else specs['W']
        set_seed(seed)
        X = generate_multiple_correlated_SDR_patterns(num_seqs=num_seqs, P=P, N=N, b=b, W=W)
        net = get_model(model, N, W, specs)

        backwards = []
        for Xi_id, Xi in enumerate(X):
            train_model(Xi, net)
            backward_errors = backward_transfer(X[:Xi_id+1], net)
            backwards.append(backward_errors)

        full_results.append(backwards)

    return full_results




def main(save_base_dir, seed):


    models = ['PAM-1', 'PAM-4', 'PAM-8', 'PC-1', 'HN-1-50', 'HN-2-50']
    bs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    specs = {
            'PAM-1':{'N_k':1, 'W':5, 'W_type':'fixed', 'configs': get_pam_configs()},
            'PAM-4':{'N_k':4, 'W':5, 'W_type':'fixed', 'configs': get_pam_configs()},
            'PAM-8':{'N_k':8, 'W':5, 'W_type':'fixed', 'configs': get_pam_configs()},
            'PAM-16':{'N_k':16, 'W':5, 'W_type':'fixed', 'configs': get_pam_configs()},
            'PAM-24':{'N_k':24, 'W':5, 'W_type':'fixed', 'configs': get_pam_configs()},
            'PC-1':{'L':1, 'W':5, 'W_type':'fixed', 'configs': get_pc_configs(l=1)},
            'PC-2':{'L':2, 'W':5, 'W_type':'fixed', 'configs': get_pc_configs(l=2)},
            'HN-1-5': {'W':5, 'W_type':'fixed', 'configs': get_hn_configs(d=1)},
            'HN-2-5': {'W':5, 'W_type':'fixed', 'configs': get_hn_configs(d=2)},
            'HN-1-50': {'W':0.5, 'W_type':'percentage', 'configs': get_hn_configs(d=1)},
            'HN-2-50': {'W':0.5, 'W_type':'percentage', 'configs': get_hn_configs(d=2)},
             }

    results = {}
    for i, model in enumerate(models):
        print(f'Current model: {model}')

        conf = dict(
                    N = 100, 
                    P = 10,
                    num_seqs=10,
                    model=model, 
                    bs = bs,
                    specs=specs[model],
                    seed=seed)

        model_result = compute(**conf)
        results[models[i]] = model_result

    print(results)
    json.dump(results, open(os.path.join(save_base_dir, f'results_{str(seed).zfill(3)}.json'), 'w'))
    args = dict( models=models, bs=bs, P=conf['P'], N=conf['N'], num_seqs=conf['num_seqs'], specs=specs)
    torch.save(dict(args=args, results=results), os.path.join(save_base_dir, f'results_{str(seed).zfill(3)}.pth'))



if __name__ == "__main__":


    save_base_dir = f'results/{os.path.splitext(os.path.basename(__file__))[0]}/run_002'
    assert checkdir(save_base_dir, careful=False), f'path {save_base_dir} exists'

    for i in range(10):
        main(save_base_dir=save_base_dir, seed=i)



