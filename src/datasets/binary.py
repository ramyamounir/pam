import sys; sys.path.append('./')
from src.experiments.utils import to_torch
import numpy as np
import torch

from src.models.pam_utils import SDR


def generate_correlated_binary_patterns(P, N, b, device, seed=1):
    np.random.seed(seed)
    X = np.zeros((int(P), int(N)))
    template = np.random.choice([-1, 1], size=N)
    prob = (1 + b) / 2
    for i in range(P):
        for j in range(N):
            if np.random.binomial(1, prob) == 1:
                X[i, j] = template[j]
            else:
                X[i, j] = -template[j]


        # revert the sign
        if np.random.binomial(1, 0.5) == 1:
            X[i, j] *= -1

    return to_torch(X, device)

def generate_correlated_SDR_patterns(P, N, b, S, seed=1):
    # higher b means more correlation -> smaller vocab size
    vocab_size = int(max(round(1.0-b, 3)*P, 1.0))
    seq_ids = torch.cat([torch.randperm(vocab_size) for _ in range(int(P//vocab_size)+1)])[:P]

    sdrs = []
    for seed in seq_ids:
        torch.manual_seed(seed)
        sdrs.append(SDR(N, S))

    return sdrs

def generate_multiple_correlated_SDR_patterns(num_seqs, P, N, b, S, seed=1):
    # higher b means more correlation -> smaller vocab size
    vocab_size = int(max(round(1.0-b, 3)*P, 1.0))

    full_sdrs = []
    for _ in range(num_seqs):

        seq_ids = torch.cat([torch.randperm(vocab_size) for _ in range(int(P//vocab_size)+1)])[:P]
        sdrs = []
        for seed in seq_ids:
            torch.manual_seed(seed)
            sdrs.append(SDR(N, S))

        full_sdrs.append(sdrs)

    return full_sdrs


def add_noise_SDR_patterns(seq, e):

    noisy_seq = [seq[0]]
    for s in seq[1:]:
        s = SDR.from_SDR(s, e=e)
        # s = s.add_noise(n=int(e*s.N))
        noisy_seq.append(s)

    return noisy_seq


if __name__ == "__main__":
    # p = generate_correlated_binary_patterns(10, 5, 1.0, 'cuda')
    # print(p)

    p = generate_correlated_SDR_patterns(20, 100, 0.1, 5)
    [print(p1) for p1 in p]
    print('\n')
    p = add_noise_SDR_patterns(p, 0.5)
    [print(p1) for p1 in p]
    quit()

