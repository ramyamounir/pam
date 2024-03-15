import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.pam_utils import SDR, Connections, Attractors
from tqdm import tqdm


class PamModel():
    def __init__(self, 
                 num_base_neurons=512, 
                 num_neurons_per_minicolumn=4,
                 sparsity=10,
                 connections_density=0.5,
                 connections_decay=0.0,
                 learning_rate=1,
                 persistance=1,
                 ):

        self.num_base_neurons = num_base_neurons
        self.num_neurons_per_minicolumn = num_neurons_per_minicolumn
        self.sparsity = sparsity
        self.num_neurons = self.num_base_neurons*self.num_neurons_per_minicolumn
        self.persistance = persistance

        self.connections = Connections(self.num_neurons, self.num_neurons, connections_density=connections_density, connections_decay=connections_decay, learning_rate=learning_rate)
        self.attractors = Attractors(self.num_base_neurons, connections_density=connections_density, connections_decay=connections_decay, learning_rate=learning_rate)
        self.start_sdr = self.create_start_sdr()

        self.parameters = ['connections', 'attractors', 'start_sdr']
        self.reset()

    def create_start_sdr(self):
        inc = torch.arange(start=0, end=self.num_neurons, step= self.num_neurons_per_minicolumn)
        rand = torch.randint(low=0, high=self.num_neurons_per_minicolumn, size=(self.num_base_neurons,))
        val = (inc + rand).long()
        return SDR(N=self.num_neurons, ix=val)

    def encode_start(self, feature):
        return SDR.from_bin(torch.logical_and(feature.bin, self.start_sdr.bin))

    def predict_start(self, feature):
        input_start = self.encode_start(feature)
        self.prediction = self.connections(input_start)
        self.prev_sdr = input_start

    def calculate_to_train(self, feature, prediction_out, prediction_sdr):

        feature_bin = feature.bin.reshape(self.num_base_neurons, self.num_neurons_per_minicolumn)
        prediction_sdr = prediction_sdr.bin.reshape(self.num_base_neurons, self.num_neurons_per_minicolumn)
        for_training_encoder = torch.zeros_like(feature_bin)

        prediction_out = prediction_out.reshape(self.num_base_neurons, self.num_neurons_per_minicolumn)
        prediction_out_argmax = torch.argmax(prediction_out, dim=-1)

        reduced_feature = feature.reduce(self.num_neurons_per_minicolumn)
        max_feature_preds = prediction_out[reduced_feature.val,prediction_out_argmax[reduced_feature.val]]
        boundary = (max_feature_preds>0.5).sum()/len(max_feature_preds) < 0.8

        for_training_encoder[range(self.num_base_neurons),prediction_out_argmax] = True
        result = torch.logical_and(feature_bin, for_training_encoder)
        return SDR.from_bin(result.reshape(-1)), boundary

    def process_input(self, feature, prediction_out):
        prediction_sdr = SDR.from_nodes_threshold(prediction_out, threshold=0.5)
        return self.calculate_to_train(feature, prediction_out, prediction_sdr)

    def generate_from(self, prediction, gen, it=100):
        for _ in range(it):
            gen = SDR.from_nodes_topk(self.attractors(gen), k=int(self.sparsity)).intersect(prediction)
        return gen

    def generate(self, prediction, it=100):
        gen = prediction.choose(self.sparsity)
        for _ in range(it):
            gen = SDR.from_nodes_topk(self.attractors(gen), k=self.sparsity).intersect(prediction)
        return gen


    def create_output(self, predicted, generated):
        output = dict(
                predicted = SDR.from_nodes_threshold(self.prediction, threshold=0.5).reduce(self.num_neurons_per_minicolumn),
                generated = generated
                )
        return output

    def train_single(self, input_sdr):

        input_sdr_expanded = input_sdr.expand(self.num_neurons_per_minicolumn)

        if self.prediction == None:
            self.predict_start(input_sdr_expanded)
            generated = self.generate(SDR.from_nodes_threshold(self.prediction, threshold=0.5).reduce(self.num_neurons_per_minicolumn))
            return self.create_output(self.prediction, generated)

        processed_input, boundary = self.process_input(input_sdr_expanded, self.prediction)


        # train prediction
        counter = 0
        while counter<=self.persistance:
            self.connections.train(self.prev_sdr, processed_input)
            prediction = self.connections(self.prev_sdr)
            pred_sdr = SDR.from_nodes_threshold(prediction, threshold=0.5)
            overlap = pred_sdr.overlap(processed_input)
            if overlap >= self.sparsity: break
            counter += 1

        # train generative
        counter = 0
        while counter<=self.persistance:
            reduced_pred = pred_sdr.reduce(self.num_neurons_per_minicolumn)
            self.attractors.process(input_sdr, reduced_pred)
            counter += 1

        self.prediction = self.connections(processed_input)
        self.prev_sdr = processed_input
        generated = self.generate(SDR.from_nodes_threshold(self.prediction, threshold=0.5).reduce(self.num_neurons_per_minicolumn))

        return self.create_output(self.prediction, generated)

    def train_seq(self, seq):
        self.reset()
        for s in seq:
            self.train_single(s)


    def infer_single(self, input_sdr):

        input_sdr_expanded = input_sdr.expand(self.num_neurons_per_minicolumn)

        if self.prediction == None:
            self.predict_start(input_sdr_expanded)
            generated = self.generate(SDR.from_nodes_threshold(self.prediction, threshold=0.5).reduce(self.num_neurons_per_minicolumn))
            return self.create_output(self.prediction, generated)

        processed_input, boundary = self.process_input(input_sdr_expanded, self.prediction)

        self.prediction = self.connections(processed_input)
        generated = self.generate(SDR.from_nodes_threshold(self.prediction, threshold=0.5).reduce(self.num_neurons_per_minicolumn))
        return self.create_output(self.prediction, generated)

    def recall_seq(self, seq, query='online'):

        result = []

        self.reset()
        for s_ix, s in enumerate(seq):
            if s_ix >0 and query=='ofline': s = res['predicted']
            res = self.infer_single(s)
            result.append(res['predicted'])

        return result

    def reset(self):
        self.prediction = None
        self.prev_sdr = None

    def save(self, path=None):
        to_save = {}
        for p in self.parameters:
            attr = getattr(self, p)
            to_save[p] = attr if isinstance(attr, torch.Tensor) else attr.save(path=None)


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
    pam_model = PamModel(128, 4, 10, 0.8, 0.0, 1.0, 100)

    seq = [SDR(128, 10) for _ in range(100)]

    # gt
    print('gt:')
    for s in seq:
        print(s)

    # train
    pam_model.train_seq(seq)

    print('\nonline predicted:')
    rec = pam_model.recall_seq(seq, 'online')
    for r in rec:
        print(r)


    print('\noffline predicted:')
    rec = pam_model.recall_seq(seq, 'offline')
    for r in rec:
        print(r)
    








