import torch
import torch.nn as nn
import numpy as np
import random
from random import shuffle

from src.utils.sdr import SDR


class Connections:
    def __init__(self, 
                 N_c, 
                 N_k,
                 synaptic_density=0.5, 
                 eta_inc = 0.1,
                 eta_dec = 0.1,
                 eta_decay=0.0, 
                 init_mean=0.0,
                 init_std=0.1
                 ):

        self.N_c = N_c
        self.N_k = N_k
        self.synaptic_density = synaptic_density
        self.eta_inc = eta_inc
        self.eta_dec = eta_dec
        self.eta_decay = eta_decay
        self.init_mean = init_mean
        self.init_std = init_std

        self.initialize()

    def clamp(self):
        self.W = self.W.clamp(-1.0, 1.0)

    def generate_binary_tensor(self):
        total_elements = torch.prod(torch.tensor(self.W.shape))
        tensor = torch.zeros_like(self.W, dtype=torch.bool)
        indices = torch.randperm(total_elements)[:int(self.synaptic_density*total_elements)]
        row_indices = indices // self.W.shape[1]
        col_indices = indices % self.W.shape[1]
        tensor[row_indices, col_indices] = 1
        return tensor

    def initialize(self):

        self.W = torch.zeros((self.N_c*self.N_k, self.N_c*self.N_k))
        nn.init.normal_(self.W, mean=self.init_mean, std=self.init_std)

        self.W_mask = self.generate_binary_tensor()
        self.W *= self.W_mask

        self.clamp()
        self.parameters = ['W', 'W_mask']


    def __call__(self, x_in_sdr):
        assert isinstance(x_in_sdr, SDR), "input must be an SDR"
        return self.W.t()@x_in_sdr.to_nodes()

    def train_A(self, x_in_sdr, x_out_sdr):
        assert isinstance(x_in_sdr, SDR), "input must be an SDR"
        assert isinstance(x_out_sdr, SDR), "output must be an SDR"

        x_in_nodes = x_in_sdr.to_nodes()
        inc = x_in_nodes@x_out_sdr.to_nodes(reverse=False).t()
        dec = x_in_nodes@x_out_sdr.to_nodes(reverse=True).t()

        self.W += (self.eta_inc * inc) - (self.eta_dec * dec) - (self.eta_decay)
        self.W *= self.W_mask
        self.clamp()

    def train_B(self, union_sdr, single_sdr):
        assert isinstance(union_sdr, SDR), "union must be an SDR"
        assert isinstance(single_sdr, SDR), "single must be an SDR"

        single_nodes = single_sdr.to_nodes()
        inc = single_nodes@single_nodes.t()
        dec1 = single_nodes@(union_sdr-single_sdr).to_nodes().t()
        dec2 = (union_sdr-single_sdr).to_nodes().t()@single_nodes

        self.W += (self.eta_inc * inc) - (0.5*self.eta_dec * dec1) - (0.5*self.eta_dec * dec2) - (self.eta_decay)
        self.W *= self.W_mask
        self.clamp()


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


class PamModel:

    def __init__(self,
                 N_c, 
                 N_k,
                 W,
                 transition_configs,
                 emission_configs,
                 ):

        self.N_c = N_c
        self.N_k = N_k
        self.W = W
        self.transition_configs = transition_configs
        self.emission_configs = emission_configs

        self.configs = ['N_c', 'N_k', 'W', 'transition_configs', 'emission_configs']
        self.initialize()

    def initialize(self):

        self.transition = Connections(
                            N_c = self.N_c, 
                            N_k = self.N_k,   
                            synaptic_density = self.transition_configs['synaptic_density'],
                            eta_inc = self.transition_configs['eta_inc'], 
                            eta_dec = self.transition_configs['eta_dec'], 
                            eta_decay = self.transition_configs['eta_decay'],
                            init_mean = self.transition_configs['init_mean'],
                            init_std = self.transition_configs['init_std']
                            )

        self.emission = Connections(
                            N_c = self.N_c, 
                            N_k = 1,
                            synaptic_density = self.emission_configs['synaptic_density'],
                            eta_inc = self.emission_configs['eta_inc'], 
                            eta_dec = self.emission_configs['eta_dec'], 
                            eta_decay = self.emission_configs['eta_decay'],
                            init_mean = self.emission_configs['init_mean'],
                            init_std = self.emission_configs['init_std']
                            )

        self.start_sdr = self.create_max_sdr(logits=torch.randn((self.N_k*self.N_c, 1)))
        self.parameters = ['transition', 'emission', 'start_sdr']

    def create_max_sdr(self, logits):
        logits_reshaped = logits.reshape(self.N_c, self.N_k)
        max_indices = torch.argmax(logits_reshaped, dim=1)
        scatter_indices = torch.arange(0, self.N_c) * self.N_k + max_indices
        return SDR(self.N_c*self.N_k, ix=scatter_indices)

    def encode_start(self, feature):
        return SDR.from_bin(torch.logical_and(feature.bin, self.start_sdr.bin))

    def forward_transition(self, z):
        return SDR.from_nodes_threshold( self.transition(z), threshold=self.transition_configs['threshold'])

    def evaluate_transition(self, z, p):
        return p in self.forward_transition(z).reduce(self.N_k)

    def forward_emission(self, z_pred, z_pred_single, iters=100):
        z_pred_single_input = z_pred_single
        for i in range(iters):
            z_pred_out = self.emission(z_pred_single_input)
            z_pred_single_output = SDR.from_nodes_threshold(z_pred_out, threshold=self.emission_configs['threshold'])
            z_pred_single_output = z_pred_single_output.intersect(z_pred)

            if z_pred_single_input == z_pred_single_output: break
            z_pred_single_input = z_pred_single_output

        return z_pred_single_output

    def evaluate_emission(self, z_pred, p, trials):
        for _ in range(trials):
            if not self.forward_emission(z_pred, p.choose(1)) == p:
                return False
        return True

    def learn_sequence(self, sequence):

        z = self.encode_start(sequence[0].expand(self.N_k))
        for p in sequence[1:]:

            # train transition
            a = self.transition(z)
            z_new = p.expand(self.N_k).intersect(self.create_max_sdr(a))
            for i in range(1000):
                self.transition.train_A(z, z_new)
                if self.evaluate_transition(z, p): break

            # train emission
            z_pred = SDR.from_nodes_threshold(self.transition(z), threshold=self.emission_configs['threshold']).reduce(self.N_k)
            for i in range(1000):
                self.emission.train_B(z_pred, p)
                if self.evaluate_emission(z_pred, p, trials=3): break

            # update z
            z = z_new

    def recall_sequence_online(self, sequence):
        recall = []
        z = self.encode_start(sequence[0].expand(self.N_k))
        for p in sequence[1:]:
            a = self.transition(z)

            z_pred = SDR.from_nodes_threshold(a, threshold=self.transition_configs['threshold'])
            recall.append(z_pred.reduce(self.N_k))

            z = p.expand(self.N_k).intersect(self.create_max_sdr(a))

        return recall

    def generate_sequence_online(self, sequence):
        generated = []
        z = self.encode_start(sequence[0].expand(self.N_k))
        for p in sequence[1:]:

            # transition
            a = self.transition(z)
            z_pred = SDR.from_nodes_threshold(a, threshold=self.transition_configs['threshold']).reduce(self.N_k)

            # attractors
            z_pred_single = self.forward_emission(z_pred, p)
            if len(z_pred_single) == 0: z_pred_single = p.choose(self.W)
            generated.append(z_pred_single)

            # update states
            z = z_pred_single.expand(self.N_k).intersect(self.create_max_sdr(a))

        return generated


    def recall_sequence_offline(self, x_in_sdr, seq_len):
        recall = []
        z = self.encode_start(x_in_sdr.expand(self.N_k))
        for _ in range(seq_len):
            z_pred = self.forward_transition(z)
            recall.append(z_pred.reduce(self.N_k))
            z = z_pred

        return recall


    def generate_sequence_offline(self, x_in_sdr, seq_len):
        generated = []

        z = self.encode_start(x_in_sdr.expand(self.N_k))
        for _ in range(seq_len):

            # transition
            a = self.transition(z)
            z_pred = SDR.from_nodes_threshold(a, threshold=self.transition_configs['threshold']).reduce(self.N_k)

            # attractors
            for i in range(100):
                z_pred_single = self.forward_emission(z_pred, z_pred.choose(self.W))
                if len(z_pred_single) >= max(1,self.W-2): break

            if len(z_pred_single) == 0: z_pred_single = z_pred.choose(self.W)
            if len(z_pred_single) > self.W: z_pred_single = z_pred_single.choose(self.W)

            generated.append(z_pred_single)

            # update states
            z = z_pred_single.expand(self.N_k).intersect(self.create_max_sdr(a))

        return generated


    def save(self, path=None):
        to_save = {}
        for p in self.parameters:
            attr = getattr(self, p)
            to_save[p] = attr if isinstance(attr, torch.Tensor) else attr.save(path=None)

        if path != None: 
            to_save['configs'] = {c:getattr(self,c) for c in self.configs}
            torch.save(to_save, path)
        else: return to_save


    def load(self, path):
        parameters = torch.load(path) if isinstance(path, str) else path

        for name, weight in parameters.items():
            if name not in self.parameters: continue

            attr = getattr(self, name)
            if isinstance(attr, torch.Tensor):
                setattr(self, name, weight)
            else:
                attr.load(weight)


    @staticmethod
    def load_model(path):
        loaded = torch.load(path)
        model = PamModel(**loaded['configs'])
        model.load(loaded)
        return model

