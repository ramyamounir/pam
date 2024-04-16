import sys; sys.path.append('./')
import numpy as np
import torch

from src.utils.sdr import SDR
from src.utils.exps import to_torch

class ProteinSequence():
    def __init__(self, N, W):
        protein_data = [line.strip() for line in open('data/proteinnet.txt', 'r').readlines()[1:2000:2]]

        # filter unique context for offline evaluation
        protein_dataset = {}
        for p in protein_data:
            if p[0] in protein_dataset: protein_dataset[p[0]].append(p)
            else: protein_dataset[p[0]] = [p]

        # filtered data
        self.protein_data = [v[torch.randperm(len(v))[0]] for k, v in protein_dataset.items()]
        self.unique_ids = sorted(list(set("".join(self.protein_data))))

        self.vocab = [SDR(N, W) for _ in range(len(self.unique_ids))]
        self.p2s = {u:s for u, s in zip(self.unique_ids, range(len(self.vocab)))}
        self.s2p = {s:u for u, s in zip(self.unique_ids, range(len(self.vocab)))}
        self.seqs = self.create_data()

    def create_data(self):
        seqs = []
        for protein_seq in self.protein_data:
            seqs.append([self.vocab[self.p2s[p]] for p in protein_seq])
        return seqs


def load_protein_patterns(N, W, seed=1):

    protein_data = [line.strip() for line in open('data/proteinnet.txt', 'r').readlines()[1:2000:2]]
    unique_ids = sorted(list(set("".join(protein_data))))
    vocab = [SDR(N, W) for _ in range(len(unique_ids))]

    p2s = {u:s for u, s in zip(unique_ids, range(len(vocab)))}
    s2p = {s:u for u, s in zip(unique_ids, range(len(vocab)))}



if __name__ == "__main__":

    seqs = load_protein_patterns(100, 5)

    for seq in seqs:
        print(len(seq))


