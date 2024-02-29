import torch
import torch.nn as nn
import torch.nn.functional as F
from tbw import TBWrapper, TBType


class TransformerSequenceMemory(nn.Module):
    def __init__(self, dim, num_layers, max_seq_len=10, batch_first=True):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers

        self.pos_enc = nn.Embedding(self.max_seq_len, self.dim, _freeze=True)
        encoder_layer = nn.TransformerEncoderLayer(self.dim, 8, batch_first=batch_first)
        self.encoder = nn.TransformerEncoder(encoder_layer, self.num_layers)
        self.lin = nn.Linear(self.dim, self.dim)

    def forward(self, x):
        # input [S,D]
        # output [1,D]

        x = x.unsqueeze(0)
        res = self.encoder(x+self.pos_enc(torch.arange(x.shape[1])).unsqueeze(0))[:,-1,:]
        return self.lin(res)


class LSTMSequenceMemory(nn.Module):
    def __init__(self, dim, num_layers):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=self.dim, hidden_size=self.dim, num_layers=self.num_layers)
        self.lin = nn.Linear(self.dim, self.dim)
        self.h0 = torch.randn(self.num_layers, self.dim)
        self.c0 = torch.randn(self.num_layers, self.dim)

    def forward(self, x):
        # input [S,D]
        # output [1,D]

        _, (hn, _) = self.lstm(x, (self.h0,self.c0))
        return self.lin(hn)


class SequenceMemory(nn.Module):
    def __init__(self, memory='transformer', dim=512, num_layers=2, pred_threshold=0.1, train_every=1):
        super().__init__()

        self.dim = dim
        self.num_layers = num_layers
        self.pred_threshold = pred_threshold
        self.train_every = train_every

        if memory=='transformer':
            self.memory = TransformerSequenceMemory(dim, num_layers)
        elif memory=='lstm':
            self.memory = LSTMSequenceMemory(dim, num_layers)

        self.optim = torch.optim.Adam(self.parameters(), lr=1e-2)

        self.clear()
        self.train_counter = 0

    def clear(self):
        self.prediction = None
        self.inputs = []

    def predict(self, x):
        inputs = torch.cat(self.inputs, dim=0)
        return self.memory(inputs)

    def optimize(self, loss):
        self.train_counter += 1
        (loss/self.train_every).backward()

        if self.train_counter > self.train_every:
            self.optim.step()
            self.optim.zero_grad()
            self.train_counter = 0

    def forward(self, x):

        if self.prediction == None:
            self.inputs = [x]
            self.prediction = self.predict(self.inputs)
            return self.prediction, False, None

        loss = F.mse_loss(self.prediction, x)

        if loss > self.pred_threshold/2:
            self.optimize(loss)

        if loss<self.pred_threshold:
            self.inputs.append(x)
            boundary=False
        else:
            self.inputs = [x]
            boundary=True

        self.prediction = self.predict(self.inputs)
        return self.prediction, boundary, loss



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




