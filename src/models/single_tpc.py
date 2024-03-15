import torch
import torch.nn as nn
import torch.nn.functional as F
import time


class Tanh(nn.Module):
    def forward(self, inp):
        return torch.tanh(inp)

    def deriv(self, inp):
        return 1.0 - torch.tanh(inp) ** 2.0

class Linear(nn.Module):
    def forward(self, inp):
        return inp

    def deriv(self, inp):
        return torch.ones((1,)).to(inp.device)



class SingleLayertPC(nn.Module):
    """Generic single layer tPC"""
    def __init__(self, input_size, lr, data_type, learn_iters, nonlin='tanh'):
        super(SingleLayertPC, self).__init__()


        self.Wr = nn.Linear(input_size, input_size, bias=False)
        if nonlin == 'linear':
            self.nonlin = Linear()
        elif nonlin == 'tanh':
            self.nonlin = Tanh()
        else:
            raise ValueError("no such nonlinearity!")

        self.optim= torch.optim.Adam(self.parameters(), lr=lr)
        
        self.input_size = input_size
        self.lr = lr
        self.data_type = data_type
        self.learn_iters = learn_iters
        
    def init_hidden(self, bsz):
        """Initializing sequence"""
        return nn.init.kaiming_uniform_(torch.empty(bsz, self.input_size)).to(self.Wr.weight.device)
    
    def forward(self, prev):
        pred = self.Wr(self.nonlin(prev))
        return pred

    def update_errs(self, curr, prev):
        """
        curr: current observation
        prev: previous observation
        """
        pred = self.forward(prev)
        err = curr - pred
        return err
    
    def get_energy(self, curr, prev):
        err = self.update_errs(curr, prev)
        energy = torch.sum(err**2)
        return energy

    def train_seq(self, seq):
        seq_len = seq.shape[0]
        losses = []
        start_time = time.time()
        for learn_iter in range(self.learn_iters):
            epoch_loss = 0
            prev = self.init_hidden(1)
            batch_loss = 0
            for k in range(seq_len):
                x = seq[k]
                self.optim.zero_grad()
                energy = self.get_energy(x, prev)
                energy.backward()
                self.optim.step()
                prev = x.clone().detach()

                # add up the loss value at each time step
                epoch_loss += energy.item() / seq_len

            losses.append(epoch_loss)
            if (learn_iter + 1) % 10 == 0:
                print(f'Epoch {learn_iter+1}, loss {epoch_loss}')

        print(f'training PC complete, time: {time.time() - start_time}')
        return losses


    def recall_seq(self, seq, query, noise=0.0):
        """recall function for pc
        
        seq: PxN sequence

        output: (P-1)xN recall of sequence (starting from the second step)
        """

        seq_len, N = seq.shape
        recall = torch.zeros((seq_len, N)).to(self.Wr.weight.device)
        recall[0] = seq[0].clone().detach() + (torch.randn_like(seq[0]) * noise)
        if query == 'online':
            # recall using true image at each step
            recall[1:] = torch.sign(self.forward(seq[:-1])) if self.data_type == 'binary' else self.forward(seq[:-1])
        else:
            # recall using predictions from previous step
            for k in range(1, seq_len):
                recall[k] = torch.sign(self.forward(recall[k-1:k])) if self.data_type == 'binary' else self.forward(recall[k-1:k]) # 1xN

        return recall



class LinearSingleLayertPC(nn.Module):
    """
    Linear version of the single layer tPC;

    This is for the convenience of searching Pmax for binary patterns

    Training is performed across the whole sequence,
    rather than step-by-step i.e., the loss is the sum over
    all timesteps.
    """
    def __init__(self, input_size, learn_iters=100, lr=1e-2):
        super(LinearSingleLayertPC, self).__init__()
        self.Wr = nn.Linear(input_size, input_size, bias=False)
        self.input_size = input_size
        self.learn_iters = learn_iters
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.SGD(list(self.Wr.parameters()), lr=lr)
    
    def forward(self, s):
        pred = self.Wr(s)
        return pred
    
    def recall(self, s):
        pred = self.Wr(s)
        return pred
    
    def get_loss(self, X):
        """X: shape PxN"""
        pred = self.forward(X[:-1]) # (P-1)xN
        loss = self.criterion(pred, X[1:])
        return loss

    def train(self, X):
        losses = []
        for i in range(self.learn_iters):
            self.optimizer.zero_grad()
            loss = self.get_loss(X)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
        return losses



if __name__ == "__main__":


    tpc = SingleLayertPC(input_size=64, lr=2e-5, data_type='continuous', learn_iters=400).cuda()
    seq = torch.randn(128,64).cuda()

    losses = tpc.train_seq(seq)
    rec = tpc.recall_seq(seq, query='online')








