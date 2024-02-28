import torch
from basics import SDR, Connections, Attractors, Attractors2
from tqdm import tqdm
import random
from utils import plot_circle_grid


class Layer4():
    def __init__(self, 
                 num_base_neurons=1024, 
                 num_neurons_per_minicolumn=4,
                 sparsity=0.02,
                 connections_density=0.5,
                 connections_decay=1.00,
                 ):

        self.num_base_neurons = num_base_neurons
        self.num_neurons_per_minicolumn = num_neurons_per_minicolumn
        self.sparsity = sparsity
        self.num_neurons = self.num_base_neurons*self.num_neurons_per_minicolumn

        self.connections = Connections(self.num_neurons, self.num_neurons, connections_density=connections_density, connections_decay=connections_decay)
        self.attractors = Attractors2(self.num_base_neurons, connections_density=connections_density, connections_decay=connections_decay)
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
            gen = SDR.from_nodes_topk(self.attractors(gen), k=int(self.sparsity)).intersect(prediction)
            # gen = SDR.from_nodes_threshold(self.attractors(gen), threshold=0.5).intersect(prediction)
        return gen

    def create_output(self, predicted, generated, boundary):
        output = dict(
                predicted = SDR.from_nodes_threshold(self.prediction, threshold=0.5).reduce(self.num_neurons_per_minicolumn),
                generated = generated if generated != None else None,
                boundary = boundary)
        return output

    def __call__(self, input_sdr, train=True, gen=False):

        input_sdr_expanded = input_sdr.expand(self.num_neurons_per_minicolumn)

        if self.prediction == None:
            self.predict_start(input_sdr_expanded)
            generated = None if gen==False else self.generate(SDR.from_nodes_threshold(self.prediction, threshold=0.5).reduce(self.num_neurons_per_minicolumn))
            return self.create_output(self.prediction, generated, False)

        processed_input, boundary = self.process_input(input_sdr_expanded, self.prediction)
        if train: self.connections.train(self.prev_sdr, processed_input)

        if boundary:
            self.predict_start(input_sdr_expanded)
            generated = None if gen==False else self.generate(SDR.from_nodes_threshold(self.prediction, threshold=0.5).reduce(self.num_neurons_per_minicolumn))
            return self.create_output(self.prediction, generated, True)

        pred_sdr = SDR.from_nodes_threshold(self.prediction, threshold=0.5)
        self.attractors.process(input_sdr, pred_sdr.reduce(self.num_neurons_per_minicolumn))

        self.prediction = self.connections(processed_input)
        self.prev_sdr = processed_input
        generated = None if gen==False else self.generate(SDR.from_nodes_threshold(self.prediction, threshold=0.5).reduce(self.num_neurons_per_minicolumn))
        return self.create_output(self.prediction, generated, False)


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

    num_base_neurons = 128
    num_neurons_per_minicolumn=4
    sparsity = 0.1
    connections_density=0.5
    connections_decay=1.0
    layer4 = Layer4(num_base_neurons=num_base_neurons, 
                    num_neurons_per_minicolumn=num_neurons_per_minicolumn, 
                    sparsity=sparsity, 
                    connections_density=connections_density,
                    connections_decay=connections_decay)

    input_sdr1 = SDR(N=num_base_neurons, S=int(num_base_neurons*sparsity))
    input_sdr2 = SDR(N=num_base_neurons, S=int(num_base_neurons*sparsity))
    input_sdr3 = SDR(N=num_base_neurons, S=int(num_base_neurons*sparsity))
    input_sdr4a= SDR(N=num_base_neurons, S=int(num_base_neurons*sparsity))
    input_sdr4b= SDR(N=num_base_neurons, S=int(num_base_neurons*sparsity))
    input_sdr4c= SDR(N=num_base_neurons, S=int(num_base_neurons*sparsity))
    input_sdr4d= SDR(N=num_base_neurons, S=int(num_base_neurons*sparsity))
    input_sdr4 = [input_sdr4a, input_sdr4b, input_sdr4c]


    epochs = 100
    for i in tqdm(range(epochs)):
        sdr2_pred, sdr2_gen = layer4(input_sdr1, gen=(i==epochs-1))
        sdr3_pred, sdr3_gen = layer4(input_sdr2, gen=(i==epochs-1))
        sdr4_pred, sdr4_gen = layer4(input_sdr3, gen=(i==epochs-1))
        # sdr5_pred, sdr5_gen = layer4(input_sdr4a if random.random()>0.5 else input_sdr4b)
        sdr5_pred, sdr5_gen = layer4(input_sdr4[torch.randint(0,len(input_sdr4),(1,)).item()])
        # sdr5_pred, sdr5_gen = layer4(input_sdr4a)

        layer4.reset()

    print('##### 2 ####') 
    print('sdr2:', input_sdr2)
    print('sdr2 predicted:', sdr2_pred)
    print('sdr2 generated:', sdr2_gen)


    print('##### 3 ####') 
    print('sdr3:', input_sdr3)
    print('sdr3 predicted:', sdr3_pred)
    print('sdr3 generated:', sdr3_gen)

    print('##### 4 ####') 
    print('sdr4a:', input_sdr4a)
    print('sdr4b:', input_sdr4b)
    print('sdr4c:', input_sdr4c)
    print('sdr4 predicted:', sdr4_pred)
    print('sdr4 generated:', sdr4_gen)


    layer4.attractors.visualize()

