import torch



class Runner():
    def __init__(self, seq_file, model):
        self.seq_file = seq_file
        self.model = model
