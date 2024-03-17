import torch
import numpy as np
import random
from random import shuffle

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import erdos_renyi_graph, to_undirected


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
    def from_bin(val):
        return SDR(len(val), ix=torch.argwhere(val==True))

    @staticmethod
    def from_SDR(sdr, b=0.5, seed=1):
        """ 
        Clones SDR with specific correlation factor b
        b=0 -> No correlation (random SDR)
        b=1 -> Full correlation (same SDR)
        """

        ix = sdr.ix
        same_ix = ix[torch.rand_like(ix.float())<b]
        available = list(set(range(sdr.N)) - set(same_ix.numpy().tolist()))
        shuffle(available)
        changed_ix = torch.tensor(available[:sdr.S-len(same_ix)])

        return SDR(sdr.N, ix = torch.cat([same_ix, changed_ix]).long())


    def add_noise(self, n=1):
        change_ix = torch.randperm(self.N)[:n]
        binary = self.bin.clone()
        binary[change_ix] = torch.logical_not(binary[change_ix])
        return SDR.from_bin(binary)

    def to_nodes(self, pad=0):
        if pad>0:
            return F.pad(self.bin.float(), (0,pad), "constant", 0).unsqueeze(-1)

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

        return torch.sort(self.ix)[0] == torch.sort(other.ix)[0]

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


class MyMessagePassing(MessagePassing):
    def __init__(self):
        super(MyMessagePassing, self).__init__(aggr='add')  # "Add" aggregation method
        
    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        return x_j * edge_attr

    def update(self, aggr_out):
        return aggr_out


def find_a_in_b(a, b, inv=False):
    condition = (a[0][:, None] == b[0].view(1, -1)) & (a[1][:, None] == b[1].view(1, -1))

    if inv:
        indices = torch.where(condition.sum(dim=0)==0)[0]
    else:
        indices = torch.where(condition)[1]

    # indicies are relative to b
    return indices


class Connections:
    def __init__(self, in_dim, out_dim, connections_density=0.5, connections_decay=1.00, learning_rate=100):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.connections_density = connections_density
        self.connections_decay = connections_decay
        self.learning_rate = learning_rate
        self.num_nodes = max(in_dim, out_dim)

        self.message_passing = MyMessagePassing()
        self.initialize()

    def initialize(self):

        fully_connected = torch.cartesian_prod(torch.arange(self.in_dim), torch.arange(self.out_dim))
        self.edge_index = fully_connected[torch.randperm(len(fully_connected))[:int(len(fully_connected)*self.connections_density)]].t().contiguous()
        self.edge_attr = torch.ones((self.edge_index.shape[-1],1))*-1
        self.parameters = ['edge_index', 'edge_attr']

    def __call__(self, x_in_sdr):
        assert isinstance(x_in_sdr, SDR), "input must be an SDR"
        x_in = x_in_sdr.to_nodes(pad=self.num_nodes-self.in_dim)
        return self.message_passing(x_in, self.edge_index, self.edge_attr)


    def train(self, x_in_sdr, x_target_sdr):
        active_ix = x_in_sdr.val
        edges_ix = torch.arange(len(self.edge_attr))

        active_edges = torch.isin(self.edge_index[0], active_ix)
        edges_ix = edges_ix[active_edges]

        good_edges = torch.isin(self.edge_index[:,active_edges][1], x_target_sdr.val)
        good_edges_ix = edges_ix[good_edges]
        bad_edges_ix = edges_ix[torch.logical_not(good_edges)]

        # adjust edges
        self.edge_attr[good_edges_ix] += 0.1
        # self.edge_attr[bad_edges_ix] -= 0.01

        # decay
        self.edge_attr -= self.connections_decay

        # clamp
        self.edge_attr[good_edges_ix] = torch.clamp_max(self.edge_attr[good_edges_ix], max=1.0)
        self.edge_attr[bad_edges_ix] = torch.clamp_min(self.edge_attr[bad_edges_ix], min=-1.0)

    def save(self, path=None):
        to_save = {}
        for p in self.parameters:
            attr = getattr(self, p)
            to_save[p] = attr if isinstance(attr, torch.Tensor) else attr.save()

        if path != None: torch.save(to_save, path)
        else: return to_save

    def load(self, path):
        parameters = torch.load(path) if isinstance(path, str) else path

        for name, weight in parameters.items():
            attr = getattr(self, name)
            if isinstance(attr, torch.Tensor):
                setattr(self, name, weight)
            else:
                attr.load(weight)


class Attractors:
    def __init__(self, dim, connections_density=0.5, connections_decay=1.0, learning_rate=100):
        self.dim = dim
        self.connections_density = connections_density
        self.connections_decay = connections_decay
        self.learning_rate = learning_rate

        self.message_passing = MyMessagePassing()
        self.initialize()

    def initialize(self):
        fully_connected = torch.cartesian_prod(torch.arange(self.dim), torch.arange(self.dim))
        self.edge_index = fully_connected[torch.randperm(len(fully_connected))[:int(len(fully_connected)*self.connections_density)]].t().contiguous()
        self.edge_attr = torch.ones((self.edge_index.shape[-1],1))*-1
        self.parameters = ['edge_index', 'edge_attr']

    def __call__(self, x_in_sdr):
        assert isinstance(x_in_sdr, SDR), "input must be an SDR"
        x_in = x_in_sdr.to_nodes()
        return self.message_passing(x_in, self.edge_index, self.edge_attr)

    def adjust_edges(self, edges, mod):
        edges_ix = find_a_in_b(edges, self.edge_index)
        if len(edges_ix) == 0: return
        self.edge_attr[edges_ix] =  self.edge_attr[edges_ix] + mod

    def process(self, single, union):
        edges_strengthen = single.val[erdos_renyi_graph(len(single), edge_prob=1.0, directed=True)]
        # edges_weaken = to_undirected(torch.cartesian_prod(single.val, (union-single).val).t().contiguous())

        self.adjust_edges(edges_strengthen, 0.1 * self.learning_rate)
        # self.adjust_edges(edges_weaken, -0.005 * self.learning_rate)
        # self.adjust_edges(edges_weaken, -1.0 * self.learning_rate)

        # decay
        self.edge_attr -= self.connections_decay


        self.edge_attr = torch.clamp(self.edge_attr, -1.0, 1.0)

    def save(self, path=None):
        to_save = {}
        for p in self.parameters:
            attr = getattr(self, p)
            to_save[p] = attr if isinstance(attr, torch.Tensor) else attr.save()

        if path != None: torch.save(to_save, path)
        else: return to_save

    def load(self, path):
        parameters = torch.load(path) if isinstance(path, str) else path

        for name, weight in parameters.items():
            attr = getattr(self, name)
            if isinstance(attr, torch.Tensor):
                setattr(self, name, weight)
            else:
                attr.load(weight)




if __name__ == "__main__":
    a = SDR(10, 3)
    print(a)
