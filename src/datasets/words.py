import sys; sys.path.append('./')
import numpy as np
import torch

from src.utils.sdr import SDR


class WordSequence():
    def __init__(self, N, W, num_words=100, seed=1):
        self.word_data = [line.strip() for line in open('data/words.txt', 'r').readlines() if len(line.strip())==4][:num_words]
        self.unique_ids = sorted(list(set("".join(self.word_data))))

        self.vocab = [SDR(N, W) for _ in range(len(self.unique_ids))]
        self.w2s = {u:s for u, s in zip(self.unique_ids, range(len(self.vocab)))}
        self.s2w = {s:u for u, s in zip(self.unique_ids, range(len(self.vocab)))}
        self.seqs = self.create_data()

    def create_data(self):
        seqs = []
        for word_seq in self.word_data:
            seqs.append([self.vocab[self.w2s[w]] for w in word_seq])
        return seqs


if __name__ == "__main__":
    ws = WordSequence(100, 5)

