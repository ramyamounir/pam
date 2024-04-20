import sys; sys.path.append('./')
import numpy as np
import torch

from src.utils.exps import to_torch
from src.utils.sdr import SDR


def generate_correlated_SDR_patterns(P, N, b, W):
    # higher b means more correlation -> smaller vocab size
    vocab_size = int(max(round(1.0-b, 3)*P, 1.0))
    vocab = [SDR(N,W) for _ in range(vocab_size)]
    seq_ids = torch.cat([torch.randperm(vocab_size) for _ in range(int(P//vocab_size)+1)])[:P]
    sdrs = [vocab[i] for i in seq_ids]
    return sdrs


def generate_multiple_correlated_SDR_patterns(num_seqs, P, N, b, W):
    return [generate_correlated_SDR_patterns(P, N, b, W) for _ in range(num_seqs)]


def add_noise_SDR_patterns(seq, e):

    noisy_seq = [seq[0]]
    for s in seq[1:]:
        s = SDR.from_SDR(s, e=e)
        noisy_seq.append(s)

    return noisy_seq

def generate_correlated_SDR_from_data(data, P, b):
    assert P <= data['sdrs'].shape[0], "Not enough data, use lower P"

    # sample
    vocab_size = int(max(round(1.0-b, 3)*P, 1.0))
    ixs = torch.randperm(data['sdrs'].shape[0])[:vocab_size]
    seq_ids = torch.cat([ixs for _ in range(int(P//vocab_size)+1)])[:P]
    seq_ids = seq_ids[torch.randperm(len(seq_ids))]

    imgs = ((data['imgs'][seq_ids]+1.0)/2.0)
    sdrs = [SDR(N=s.shape[0], ix=torch.where(s==1.0)[0].cpu()) for s in data['sdrs'][seq_ids]]
    recons = ((data['recons'][seq_ids]+1.0)/2.0)

    return imgs, sdrs, recons

def generate_video_SDR_from_data(data, P, seq_id=None):
    assert P <= len(data['sdrs'][0]), "Not enough data, use lower P"

    if seq_id == None: ix = torch.randperm(len(data['sdrs']))[0]
    else: ix=seq_id

    ix_in = torch.linspace(0, len(data['sdrs'][ix])-1, P).long()
    mask = torch.zeros((len(data['sdrs'][ix]),)).bool()
    mask[ix_in] = True

    imgs = ((data['imgs'][ix][mask]+1.0)/2.0)
    sdrs = [SDR(N=s.shape[0], ix=torch.where(s==1.0)[0].cpu()) for s in data['sdrs'][ix][mask]]
    recons = ((data['recons'][ix][mask]+1.0)/2.0)

    return imgs, sdrs, recons


def generate_multiple_video_SDR_from_data(data, num_seq, P):
    assert P <= len(data['sdrs'][0]), "Not enough data, use lower P"

    seq_ix = torch.randperm(len(data['sdrs']))[:num_seq]

    ix_in = torch.linspace(0, len(data['sdrs'][seq_ix][0])-1, P).long()
    mask = torch.zeros((len(data['sdrs'][seq_ix][0]),)).bool()
    mask[ix_in] = True

    imgs = ((data['imgs'][seq_ix][:,mask]+1.0)/2.0)
    sdrs = [ [SDR(N=s.shape[0], ix=torch.where(s==1.0)[0].cpu()) for s in seq] for seq in data['sdrs'][seq_ix][:,mask]]
    recons = ((data['recons'][seq_ix][:,mask]+1.0)/2.0)

    return imgs, sdrs, recons


    

if __name__ == "__main__":
    # p = generate_correlated_binary_patterns(10, 5, 1.0, 'cuda')
    # print(p)

    p = generate_correlated_SDR_patterns(20, 100, 0.1, 5)
    [print(p1) for p1 in p]
    print('\n')
    p = add_noise_SDR_patterns(p, 0.5)
    [print(p1) for p1 in p]
    quit()

