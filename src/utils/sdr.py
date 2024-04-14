import torch
import numpy as np
from random import shuffle


class SDR():

    def __init__(self, N, S=None, ix=None):
        assert ((S == None and ix != None) or (S != None and ix == None)), "Must define either S or ix, and not both"

        self.N = self.to_tensor(N)
        self.S = S

        if self.S != None:
            self.S = self.to_tensor(S)
            self.ix = torch.randperm(N)[:S]
        else:
            self.ix = self.to_tensor(ix).unique()
            self.S = len(self.ix)

        self.parameters = ['N', 'S', 'ix']

    def to_tensor(self, val):
        if isinstance(val, torch.Tensor):
            return val
        elif isinstance(val, int):
            return torch.tensor(val)
        elif isinstance(val, list):
            return torch.tensor(val)
        elif isinstance(val, np.asarray):
            return torch.from_numpy(val)
        else:
            raise NotImplementedError()

    @property
    def val(self): 
        return self.ix


    @property
    def shape(self): 
        return self.ix.shape

    @property
    def not_val(self):
        full = torch.arange(self.N)
        return full[torch.logical_not(torch.isin(full, self.ix))]

    @property
    def bin(self):
        res = torch.zeros((self.N), dtype=torch.bool)
        res[self.ix] = True
        return res

    @property
    def not_bin(self):
        return torch.logical_not(self.bin)


    @staticmethod
    def from_nodes_threshold(nodes, threshold=0.5):
        return SDR(len(nodes), ix=torch.argwhere(nodes>=threshold)[:,0])

    @staticmethod
    def from_nodes_topk(nodes, k=None, largest=True):
        if k==None: k=int(0.1*len(nodes))
        assert k <= len(nodes), "use smaller k for topK"
        return SDR(len(nodes), ix=torch.topk(nodes,k,dim=0,largest=largest)[1])

    @staticmethod
    def from_nodes_th_k(nodes, threshold=0.5, k=5):
        return SDR.from_nodes_threshold(nodes, threshold=threshold).intersect(SDR.from_nodes_topk(nodes, k=k))

    @staticmethod
    def from_bin(val):
        return SDR(len(val), ix=torch.argwhere(val==True))

    @staticmethod
    def from_SDR(sdr, e=0.5):
        """ 
        Clones SDR with specific noise factor e
        e -> number of active bits changed
        """

        assert 0<= e <= sdr.S, "invalid noise"

        ix = sdr.ix
        same_ix = ix[torch.randperm(len(sdr))[:len(ix)-e]]
        available = list(set(range(sdr.N)) - set(ix.numpy().tolist()))
        shuffle(available)
        changed_ix = torch.tensor(available[:sdr.S-len(same_ix)])

        return SDR(sdr.N, ix = torch.cat([same_ix, changed_ix]).long())


    def add_noise(self, n=1):
        change_ix = torch.randperm(self.N)[:n]
        binary = self.bin.clone()
        binary[change_ix] = torch.logical_not(binary[change_ix])
        return SDR.from_bin(binary)

    def to_nodes(self, pad=0, reverse=False):
        if pad>0: 
            if reverse:
                return F.pad(self.not_bin.float(), (0,pad), "constant", 0).unsqueeze(-1)
            else:
                return F.pad(self.bin.float(), (0,pad), "constant", 0).unsqueeze(-1)

        if reverse:
            return self.not_bin.float().unsqueeze(-1)
        else:
            return self.bin.float().unsqueeze(-1)



    def encode(self, location, model):
        assert (location.N % self.N == 0 and model.N % self.N == 0), "location and model must be factors of feature"
        minicolumn_size = location.N // self.N

        feature_expanded = self.bin.repeat_interleave(minicolumn_size).reshape(self.N, minicolumn_size).clone()
        encoders = torch.logical_or(location.bin.bool(), model.bin.bool()).reshape(self.N, minicolumn_size)
        result = torch.logical_and(feature_expanded, encoders)

        ix = torch.argwhere(torch.logical_and(feature_expanded.sum(dim=-1).bool(), 
                                              torch.logical_not(result.sum(dim=-1).bool())
                                              ))
        result[ix] = True
        return SDR.from_bin(result.reshape(-1))

    def expand(self, minicolumn_size):
        return SDR.from_bin(self.bin.bool().repeat_interleave(minicolumn_size))

    def reduce(self, minicolumn_size):
        return SDR.from_bin(self.bin.reshape(-1,minicolumn_size).sum(-1).bool())

    def intersect(self, other):
        return SDR(self.N, ix=torch.tensor(np.intersect1d(self.val,other.val)))

    def choose(self, S):
        return SDR(self.N, ix=self.ix[torch.randperm(len(self.ix))[:S]])

    def overlap(self, other):
        return len(self.intersect(other))

    def match(self, other, threshold):
        return self.overlap(other) > threshold

    def iou(self, other):
        return self.overlap(other)/len(self+other)

    def __iter__(self):
        vals = self.ix[torch.randperm(len(self.ix))]
        for i in vals:
            yield SDR(self.N, ix=i)

    def __add__(self, other):
        return SDR(self.N, ix=torch.tensor(np.union1d(self.val,other.val)))

    def __sub__(self, other):
        inter = self.intersect(other)
        binary = self.bin
        binary[inter.val] = 0
        return SDR.from_bin(binary)

    def __eq__(self, other):
        if other==None or len(self)!=len(other):
            return False

        return torch.all(torch.sort(self.ix)[0] == torch.sort(other.ix)[0])

    def __contains__(self, other):
        return self.intersect(other) == other

    def __repr__(self):
        return str(torch.sort(self.val)[0])

    def __len__(self):
        return self.S

    def save(self, path=None):
        to_save = {}
        for p in self.parameters:
            attr = getattr(self, p)
            to_save[p] = attr if (isinstance(attr, torch.Tensor) or isinstance(attr, int)) else attr.save()

        if path != None: torch.save(to_save, path)
        else: return to_save

    def load(self, path):
        parameters = torch.load(path) if isinstance(path, str) else path

        for name, weight in parameters.items():
            attr = getattr(self, name)
            if isinstance(attr, torch.Tensor) or isinstance(attr, int):
                setattr(self, name, weight)
            else:
                attr.load(weight)
