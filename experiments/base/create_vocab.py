import torch
import torch.nn as nn

# import sys;sys.path.append('../..')
import sys; sys.path.append('./')
from basics import SDR


class Vocab():
    def __init__(self, kind, dim, size, sparsity=0.02):
        self.kind = kind
        self.dim = dim
        self.size = size
        self.sparsity = sparsity

        self.create()

    def create(self):
        if self.kind=='sparse':
            self.voc = [SDR(self.dim, int(self.dim*self.sparsity)) for _ in range(self.size)]
        elif self.kind =='dense':
            self.voc = nn.Embedding(self.size, self.dim)
        else:
            raise NotImplementedError("Choose either sparse or dense")

    def save(self, path):
        to_save = dict(kind=self.kind, dim=self.dim, size=self.size, sparsity=self.sparsity)
        if self.kind=='sparse':
            to_save['vals'] = [sdr.save() for sdr in self.voc]
        elif self.kind =='dense':
            to_save['vals'] = self.voc.weight.data
        else:
            raise NotImplementedError("Choose either sparse or dense")

        torch.save(to_save, path)

    @staticmethod
    def load_from(path):
        loaded = torch.load(path)
        v = Vocab(kind=loaded['kind'], dim=loaded['dim'], size=loaded['size'], sparsity=loaded['sparsity'])

        if v.kind == 'sparse':
            for sdr, val in zip(v.voc, loaded['vals']):
                sdr.load(val)
        elif v.kind == 'dense':
            v.voc.weight.data = loaded['vals']
        else:
            raise NotImplementedError("Invalid file")

        return v

    def __getitem__(self, ix):
        if isinstance(ix, slice):
            ix = list(range(ix.stop)[ix])

        if self.kind == 'sparse':
            return torch.stack([self.voc[i].bin for i in ix]).float()
            # return self.voc[ix]
        elif self.kind == 'dense':
            return self.voc(ix)
        else:
            raise NotImplementedError("Invalid file")


    def __repr__(self):
        msg = f"{self.kind} vocab with {self.size} entries, {self.dim} dim"
        if self.kind == 'sparse':
            msg += f", and {self.sparsity} sparsity"
        return msg

    def __len__(self):
        return self.size


if __name__ == "__main__":

    v = Vocab(kind='sparse', dim=128, size=1000, sparsity=0.1)
    v.create()
    v.save('saved/sparse_100_128_10.pth')


    # v = Vocab.load_from('saved/sparse.pth')
    # print(v)
