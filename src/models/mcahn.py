import torch
import torch.nn as nn
import torch.nn.functional as F




class ModernAsymmetricHopfieldNetwork(nn.Module):
    """
    MAHN. train() function is simply a placeholder since we don't really train these models
    """
    
    def __init__(self, input_size, data_type, sep='linear', beta=1):
        super(ModernAsymmetricHopfieldNetwork, self).__init__()
        self.W = torch.zeros((input_size, input_size))
        self.sep = sep
        self.beta = beta
        self.data_type = data_type
        
    def forward(self, X, s):
        """
        X: stored memories, shape PxN
        s: query, shape (P-1)xN
        output: (P-1)xN matrix
        """
        _, N = X.shape
        if self.sep == 'exp':
            score = torch.exp(torch.matmul(s, X[:-1].t()))
        elif self.sep == 'softmax':
            score = F.softmax(self.beta * torch.matmul(s, X[:-1].t()), dim=1)
        else:
            score = torch.matmul(s, X[:-1].t()) ** int(self.sep)
        output = torch.matmul(score, X[1:])
        
        return output
        
    def train(self, X):
        """
        X: PxN matrix, where P is seq len, N the number of neurons

        Asymmetric HN's weight is the auto-covariance matrix of patterns
        """
        P, N = X.shape
        self.W = torch.matmul(X[1:].T, X[:-1]) / N

        return -1


    def recall_seq(self, seq, query, noise=0.0):
        """recall function for pc
        
        seq: PxN sequence
        mode: online or offline
        binary: true or false

        output: (P-1)xN recall of sequence (starting from the second step)
        """

        seq_len, N = seq.shape
        recall = torch.zeros((seq_len, N))
        recall[0] = seq[0].clone().detach() + (torch.randn_like(seq[0]) * noise)
        if query == 'online':
            # recall using true image at each step
            recall[1:] = torch.sign(self.forward(seq, seq[:-1])) if self.data_type == 'binary' else self.forward(seq, seq[:-1]) # (P-1)xN
        else:
            # recall using predictions from previous step
            prev = seq[0].clone().detach() # 1xN
            for k in range(1, seq_len):
                # prev = torch.sign(model(seq, prev)) if binary else model(seq, prev) # 1xN
                recall[k] = torch.sign(self.forward(seq, recall[k-1:k])) if self.data_type == 'binary' else self.forward(seq, recall[k-1:k]) # 1xN

        return recall
                


if __name__ == "__main__":
    hop = ModernAsymmetricHopfieldNetwork(64, data_type='continuous', sep='softmax').cuda()
    seq = torch.randn((128,64)).cuda()
    rec = hop.recall_seq(seq, query='online')
    print(rec.shape)

