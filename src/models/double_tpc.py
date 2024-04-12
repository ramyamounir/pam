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



class MultilayertPC(nn.Module):
    """Multi-layer tPC class, using autograd"""
    def __init__(self, 
                 hidden_size, 
                 output_size, 
                 learn_lr, 
                 inf_lr, 
                 learn_iters,
                 inf_iters,
                 nonlin='tanh'):


        super(MultilayertPC, self).__init__()
        self.hidden_size = hidden_size
        self.Wr = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Wout = nn.Linear(hidden_size, output_size, bias=False)

        if nonlin == 'linear':
            self.nonlin = Linear()
        elif nonlin == 'tanh':
            self.nonlin = Tanh()
        else:
            raise ValueError("no such nonlinearity!")

        self.optim = torch.optim.Adam(self.parameters(), lr = learn_lr)
        self.learn_lr = learn_lr
        self.inf_lr = inf_lr
        self.learn_iters = learn_iters
        self.inf_iters = inf_iters
    
    def forward(self, prev_z):
        pred_z = self.Wr(self.nonlin(prev_z))
        pred_x = self.Wout(self.nonlin(pred_z))
        return pred_z, pred_x

    def init_hidden(self, bsz):
        """Initializing prev_z"""
        return nn.init.kaiming_uniform_(torch.empty(bsz, self.hidden_size))

    def update_errs(self, x, prev_z):
        pred_z, _ = self.forward(prev_z)
        pred_x = self.Wout(self.nonlin(self.z))
        err_z = self.z - pred_z
        err_x = x - pred_x
        return err_z, err_x
    
    def update_nodes(self, x, prev_z, inf_lr, update_x=False):
        err_z, err_x = self.update_errs(x, prev_z)
        delta_z = err_z - self.nonlin.deriv(self.z) * torch.matmul(err_x, self.Wout.weight.detach().clone())
        self.z -= inf_lr * delta_z
        if update_x:
            delta_x = err_x
            x -= inf_lr * delta_x

    def inference(self, inf_iters, inf_lr, x, prev_z, update_x=False):
        """prev_z should be set up outside the inference, from the previous timestep

        Args:
            train: determines whether we are at the training or inference stage
        
        After every time step, we change prev_z to self.z
        """
        with torch.no_grad():
            # initialize the current hidden state with a forward pass
            self.z, _ = self.forward(prev_z)

            # update the values nodes
            for i in range(inf_iters):
                self.update_nodes(x, prev_z, inf_lr, update_x)
                
    def update_grads(self, x, prev_z):
        """x: input at a particular timestep in stimulus
        
        Could add some sparse penalty to weights
        """
        err_z, err_x = self.update_errs(x, prev_z)
        self.hidden_loss = torch.sum(err_z**2)
        self.obs_loss = torch.sum(err_x**2)
        energy = self.hidden_loss + self.obs_loss
        return energy


    def train_seq(self, seq, verbose=False):
        """
        Function to train multi layer tPC
        """
        seq_len = seq.shape[0]
        losses = []
        start_time = time.time()
        for learn_iter in range(self.learn_iters):
            epoch_loss = 0
            prev_z = self.init_hidden(1).to(self.Wr.weight.device)
            for k in range(seq_len):
                x = seq[k].clone().detach()
                self.optim.zero_grad()
                self.inference(self.inf_iters, self.inf_lr, x, prev_z)
                energy = self.update_grads(x, prev_z)
                energy.backward()
                self.optim.step()
                prev_z = self.z.clone().detach()

                # add up the loss value at each time step
                epoch_loss += energy.item() / seq_len

            losses.append(epoch_loss)
            if verbose and (learn_iter + 1) % 10 == 0:
                print(f'Epoch {learn_iter+1}, loss {epoch_loss}')
            
        if verbose: print(f'training PC complete, time: {time.time() - start_time}')
        return losses


    def recall_seq(self, seq, query):

        seq_len, N = seq.shape
        recall = torch.zeros((seq_len, N)).to(self.Wr.weight.device)
        recall[0] = seq[0].clone().detach()
        prev_z = self.init_hidden(1).to(self.Wr.weight.device)

        self.eval()
        with torch.no_grad():

            if query == 'online':
                # infer the latent state at each time step, given correct previous input
                for k in range(seq_len-1):
                    x = seq[k].clone().detach()
                    self.inference(self.inf_iters, self.inf_lr, x, prev_z)
                    prev_z = self.z.clone().detach()
                    _, pred_x = self.forward(prev_z)
                    recall[k+1] = pred_x

            elif query == 'offline':
                # only infer the latent of the cue, then forward pass
                x = seq[0].clone().detach()
                self.inference(self.inf_iters, self.inf_lr, x, prev_z)
                prev_z = self.z.clone().detach()

                # fast forward pass
                for k in range(1, seq_len):
                    prev_z, pred_x = self.forward(prev_z)
                    recall[k] = pred_x


        self.train()
        return recall







def multilayer_recall(model, seq, inf_iters, inf_lr,  device):

    query = 'online'


    seq_len, N = seq.shape
    recall = torch.zeros((seq_len, N)).to(device)
    recall[0] = seq[0].clone().detach()
    prev_z = model.init_hidden(1).to(device)

    if query == 'online':
        # infer the latent state at each time step, given correct previous input
        for k in range(seq_len-1):
            x = seq[k].clone().detach()
            model.inference(inf_iters, inf_lr, x, prev_z)
            prev_z = model.z.clone().detach()
            _, pred_x = model(prev_z)
            recall[k+1] = pred_x

    elif query == 'offline':
        # only infer the latent of the cue, then forward pass
        x = seq[0].clone().detach()
        model.inference(inf_iters, inf_lr, x, prev_z)
        prev_z = model.z.clone().detach()

        # fast forward pass
        for k in range(1, seq_len):
            prev_z, pred_x = model(prev_z)
            recall[k] = pred_x

    return recall




if __name__ == "__main__":
    mtpc = MultilayertPC(
                 hidden_size=64, 
                 output_size=128, 
                 learn_lr=1e-5, 
                 inf_lr=1e-2, 
                 learn_iters=200,
                 inf_iters=100,
                 ).cuda()

    seq = torch.randn((10, 128)).cuda()

    losses = mtpc.train_seq(seq)
    print(losses)

    rec = mtpc.recall_seq(seq, 'online')
    print(rec.shape)





