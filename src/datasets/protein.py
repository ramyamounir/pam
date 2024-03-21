import sys; sys.path.append('./')
from src.experiments.utils import to_torch
import numpy as np
import torch

from src.models.pam_utils import SDR

class ProteinSequence():
    def __init__(self, N, S, seed=1):
        self.protein_data = [line.strip() for line in open('data/proteinnet.txt', 'r').readlines()[1:2000:2]]
        self.unique_ids = sorted(list(set("".join(self.protein_data))))

        self.vocab = [SDR(N, S) for _ in range(len(self.unique_ids))]
        self.p2s = {u:s for u, s in zip(self.unique_ids, range(len(self.vocab)))}
        self.s2p = {s:u for u, s in zip(self.unique_ids, range(len(self.vocab)))}
        self.seqs = self.create_data()

    def create_data(self):
        seqs = []
        for protein_seq in self.protein_data:
            seqs.append([self.vocab[self.p2s[p]] for p in protein_seq])
        return seqs


def load_protein_patterns(N, S, seed=1):

    protein_data = [line.strip() for line in open('data/proteinnet.txt', 'r').readlines()[1:2000:2]]
    unique_ids = sorted(list(set("".join(protein_data))))
    vocab = [SDR(N, S) for _ in range(len(unique_ids))]

    p2s = {u:s for u, s in zip(unique_ids, range(len(vocab)))}
    s2p = {s:u for u, s in zip(unique_ids, range(len(vocab)))}



if __name__ == "__main__":

    seqs = load_protein_patterns(100, 5)

    for seq in seqs:
        print(len(seq))


