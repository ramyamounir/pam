import torch
import torch.nn as nn

import sys; sys.path.append('./')
from basics import SDR
from experiments.base.create_vocab import Vocab


class Generator():
    def __init__(self, vocab_path, vocab_len, len_seq, num_seq, num_datasets):
        self.vocab_path = vocab_path
        self.vocab = Vocab.load_from(vocab_path)
        self.vocab_len = vocab_len
        self.len_seq = len_seq
        self.num_seq = num_seq
        self.num_datasets = num_datasets

        self.current_dataset = 0

        # checks
        assert vocab_len<=len(self.vocab), "chosen vocab length should be less than or equal to full vocab size"
        assert num_seq>0, "Number of sequences must be greater than zero"
        assert len_seq>0, "Length of sequences must be greater than zero"
        assert num_datasets>0, "Number of datasets must be greater than zero"

    def create(self):
        self.seqs = torch.randint(0, self.vocab_len, size=(self.num_datasets, self.num_seq, self.len_seq))

    def set_dataset(self, dataset_ix):
        self.current_dataset = dataset_ix

    def save(self, path):
        to_save = dict(vocab_path=self.vocab_path, vocab_len=self.vocab_len, vocab_size=self.vocab_len, len_seq=self.len_seq, num_seq=self.num_seq, num_datasets=self.num_datasets)
        to_save['seqs'] = self.seqs
        torch.save(to_save, path)

    @staticmethod
    def load_from(path):
        loaded = torch.load(path)
        gen = Generator(vocab_path=loaded['vocab_path'], vocab_len=loaded['vocab_len'], len_seq=loaded['len_seq'], num_seq=loaded['num_seq'], num_datasets=loaded['num_datasets'])
        gen.seqs = loaded['seqs']
        return gen

    def __iter__(self):
        for seq in self.seqs[self.current_dataset]:
            for val_ix, val in enumerate(seq):
                yield self.vocab[val], val_ix==len(seq)-1

    def get_stream_AR(self, shuffle=True):
        data = self.seqs[self.current_dataset]
        ixs = torch.arange(data.shape[0]*data.shape[1])
        if shuffle: ixs = ixs[torch.randperm(len(ixs))]

        for ix in ixs:
            seq_id = ix//data.shape[-1]
            point_id = ix%data.shape[-1]
            yield self.vocab.getsdrs(data[seq_id][:point_id+1])


    def get_stream_full(self, shuffle=True):
        data = self.seqs[self.current_dataset]
        ixs = torch.arange(data.shape[0])
        if shuffle: ixs = ixs[torch.randperm(len(ixs))]

        for ix in ixs:
            yield self.vocab.getsdrs(data[ix])

    def __repr__(self):
        return f"Generator of {self.num_datasets} datasets, with {self.num_seq} sequences and {self.len_seq} length each"



if __name__ == "__main__":

    for l in range(10,110,10):
        g = Generator(vocab_path='saved/sparse_100_128_10.pth', vocab_len=10, len_seq=l, num_seq=1, num_datasets=10)
        g.create()
        g.save(f'../sequence_memory/saved/seq_{str(l).zfill(3)}.pth')

    quit()


    g.set_dataset(0)
    for i,b in g:
        print(i,b)
    quit()


    g = Generator.load_from('saved/seq.pth')
    print(g.seqs)



