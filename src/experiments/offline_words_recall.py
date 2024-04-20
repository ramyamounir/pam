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
from src.datasets.words import WordSequence

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


def train_model(seqs, net):

    if isinstance(net, PamModel):
        for seq in seqs:
            net.learn_sequence(seq)

    elif isinstance(net, SingleLayertPC):

        X_polar = []
        for seq in seqs:
            X_polar.append(torch.stack([x.bin.float() for x in seq])*2.0-1.0)
        X_polar = torch.stack(X_polar)

        net.train_batched_seqs(X_polar)

    elif isinstance(net, ModernAsymmetricHopfieldNetwork):
        pass
    else:
        raise ValueError("model not found")


def gen_words(seqs, net):
    if isinstance(net, PamModel):
        return [net.generate_sequence_offline(seq[0], len(seq)-1) for seq in seqs]

    elif isinstance(net, SingleLayertPC):
        X_polar = []
        for seq in seqs:
            X_polar.append(torch.stack([x.bin.float() for x in seq])*2.0-1.0)
        X_polar = torch.stack(X_polar)

        return [net.recall_seq(seq, query='offline') for seq in X_polar]

    elif isinstance(net, ModernAsymmetricHopfieldNetwork):
        X_polar = []
        for seq in seqs:
            X_polar.append(torch.stack([x.bin.float() for x in seq])*2.0-1.0)
        X_polar = torch.stack(X_polar)

        return [net.recall_seqs(seq, X_polar, query='offline') for seq in X_polar]


def decode_words(generated_words, X, net):
    if isinstance(net, PamModel):
        words = []
        for gen_word in generated_words:
            word = ""
            for gw in gen_word:
                max_iou = torch.argmax(torch.tensor([gw.iou(v) for v in X.vocab])).item()
                letter = X.s2w[max_iou]
                word += letter
            words.append(word)
        return words

    else:
        words = []
        for gen_word in generated_words:
            word = ""
            for gw in gen_word:
                gw = SDR.from_bin((gw+1.0)/2.0)
                max_iou = torch.argmax(torch.tensor([gw.iou(v) for v in X.vocab])).item()
                letter = X.s2w[max_iou]
                word += letter
            words.append(word)
        return words


def compute(N, num_words, model, num_gens, specs, seed=1):

    W = int(specs['W']*N) if specs['W_type']=='percentage' else specs['W']

    ious = []
    set_seed(seed)
    X = WordSequence(N, W, num_words=num_words)
    gt_words_set = set(X.word_data)

    net = get_model(model, N, W, specs)
    train_model(X.seqs, net)

    gen_set = set()
    recalls = []
    for gen_id in num_gens:
        generated_words = gen_words(X.seqs, net)
        decoded_words = decode_words(generated_words, X, net)
        gen_set.update(decoded_words)
        recalls.append(len(gt_words_set.intersection(gen_set))/len(gt_words_set))

    return recalls




def main(save_base_dir, seed):

    models = ['PAM-1', 'PAM-4', 'PAM-8', 'PC-1', 'HN-1-50', 'HN-2-50']
    num_gens = [1, 2, 3, 4, 5]
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

    for num_words in [1, 10, 100]:
        print(f'Num words: {num_words}')

        results = {}
        for i, model in enumerate(models):
            print(f'Model: {model}')

            conf = dict(N = 100, 
                        num_words=num_words,
                        model=model, 
                        num_gens=num_gens,
                        specs=specs[model],
                        seed=seed)

            model_result = compute(**conf)
            results[models[i]] = model_result


        print(results)
        json.dump(results, open(os.path.join(save_base_dir, f'results_{num_words}_{str(seed).zfill(3)}.json'), 'w'))
        args = dict(models=models, num_words=conf['num_words'], num_gens=num_gens, N=conf['N'], specs=specs)
        torch.save(dict(args=args, results=results), os.path.join(save_base_dir, f'results_{num_words}_{str(seed).zfill(3)}.pth'))


if __name__ == "__main__":

    save_base_dir = f'results/{os.path.splitext(os.path.basename(__file__))[0]}/run_002'
    assert checkdir(save_base_dir, careful=False), f'path {save_base_dir} exists'

    for i in range(10):
        main(save_base_dir=save_base_dir, seed=i)


