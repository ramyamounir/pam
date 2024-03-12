import torch
import torch.nn as nn
import torch.nn.functional as F
from tbw import TBWrapper, TBType
from basics import SDR


class TransformerSequenceMemory(nn.Module):
    def __init__(self, dim, num_layers, max_seq_len=100, batch_first=True):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers

        self.pos_enc = nn.Embedding(self.max_seq_len, self.dim, _freeze=True)
        # self.pos_enc = [SDR(N=self.dim, S=20) for _ in range(self.max_seq_len)]

        encoder_layer = nn.TransformerEncoderLayer(self.dim, 8, batch_first=batch_first)
        self.encoder = nn.TransformerEncoder(encoder_layer, self.num_layers)
        self.lin = nn.Linear(self.dim, self.dim)

    def create_mask(self, input_tensor):
        batch_size, seq_len, _ = input_tensor.size()
        mask = torch.tril(torch.ones((seq_len, seq_len)), diagonal=-1).bool()
        return mask

    def forward(self, x):
        # input [S,D]
        # output [1,D]

        x = x.unsqueeze(0)
        mask = self.create_mask(x)

        # pos = torch.stack([self.pos_enc[p].bin.float() for p in range(x.shape[1])]).unsqueeze(0)
        # x_pos = x + pos

        x_pos = x+self.pos_enc(torch.arange(x.shape[1])).unsqueeze(0)
        res = self.encoder(x_pos, mask=mask)

        return F.sigmoid(self.lin(res[:,-1,:]))



class GRUSequenceMemory(nn.Module):
    def __init__(self, dim, num_layers):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers

        self.gru = nn.GRU(input_size=self.dim, hidden_size=self.dim, num_layers=self.num_layers)
        self.lin = nn.Linear(self.dim, self.dim)

    def forward(self, x):
        # input [S,D]
        # output [1,D]

        out, h = self.gru(x)
        return self.lin(out)
        # return out


class GRUSequenceMemoryCell(nn.Module):
    def __init__(self, dim, num_layers):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers

        self.gru = nn.GRUCell(input_size=self.dim, hidden_size=self.dim)
        self.lin = nn.Linear(self.dim, self.dim)
        self.h = torch.zeros(1,self.dim)

    def forward(self, x, h=None):
        # input [S,D]
        # output [1,D]
        
        if h==None:
            h=self.h

        h = self.gru(x, hx=h)
        return self.lin(h), h.detach()




class LSTMSequenceMemory(nn.Module):
    def __init__(self, dim, num_layers):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=self.dim, hidden_size=self.dim, num_layers=self.num_layers)
        self.lin = nn.Linear(self.dim, self.dim)

        # self.h0 = torch.zeros(self.num_layers, self.dim)
        # self.c0 = torch.zeros(self.num_layers, self.dim)

    def forward(self, x):
        # input [S,D]
        # output [1,D]

        out, (hn, _) = self.lstm(x)
        return self.lin(out)



class SequenceMemory(nn.Module):
    def __init__(self, memory, lr, pred_threshold=0.1, accum=100):
        super().__init__()

        self.memory = memory
        self.pred_threshold = pred_threshold
        self.accum = accum
        self.optim = torch.optim.Adam(self.memory.parameters(), lr=lr)

        self.reset()
        self.train_counter = 0

    def reset(self):
        self.prediction = None
        self.inputs = []

    def predict(self, x):
        inputs = torch.cat(self.inputs, dim=0)
        return self.memory(inputs)

    def optimize(self, loss):

        (loss/self.accum).backward()

        if self.train_counter>self.accum:
            self.optim.step()
            self.optim.zero_grad()
            self.train_counter = 0

    def forward(self, x):

        h = None
        for x_input, target in zip(x[:-1], x[1:]):
            x_input, target = x_input.unsqueeze(0), target.unsqueeze(0)

            for _ in range(1):
                self.prediction, h_out = self.memory(x_input, h)
                loss = F.mse_loss(self.prediction, target)
                self.optimize(loss)
                self.train_counter += 1

            h = h_out

        return self.prediction, loss



if __name__ == "__main__":
    writer = TBWrapper('logs/lstm')
    writer(TBType.SCALAR, 'loss')

    mem = SequenceMemory(memory='lstm', dim=512, num_layers=1, pred_threshold=0.1, train_every=1)
    sequence = torch.randn(20,512)

    optim = torch.optim.Adam(mem.parameters(), lr=1e-2)

    for _ in range(100_000):

        bound_count = 0
        for s in sequence:
            s = s.unsqueeze(0)
            pred, boundary, loss = mem(s)

            print(boundary, loss)
            if not boundary: bound_count += 1
            # if boundary:
            #     break

        if bound_count == len(sequence):
            print('DONE')
            quit()

        print('\n')
        mem.clear()




