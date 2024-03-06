import torch
from basics import SDR, Connections, Attractors, Attractors2
from tqdm import tqdm
import random
from utils import plot_circle_grid


class Layer4():
    def __init__(self, 
                 num_base_neurons=1024, 
                 num_neurons_per_minicolumn=4,
                 sparsity=10,
                 connections_density=0.5,
                 connections_decay=1.00,
                 learning_rate=100,
                 ):

        self.num_base_neurons = num_base_neurons
        self.num_neurons_per_minicolumn = num_neurons_per_minicolumn
        self.sparsity = sparsity
        self.num_neurons = self.num_base_neurons*self.num_neurons_per_minicolumn

        self.connections = Connections(self.num_neurons, self.num_neurons, connections_density=connections_density, connections_decay=connections_decay, learning_rate=learning_rate)
        self.attractors = Attractors2(self.num_base_neurons, connections_density=connections_density, connections_decay=connections_decay, learning_rate=learning_rate)
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

        # train predictor
        processed_input, boundary = self.process_input(input_sdr_expanded, self.prediction)
        if train: self.connections.train(self.prev_sdr, processed_input)
        if boundary:
            self.predict_start(input_sdr_expanded)
            generated = None if gen==False else self.generate(SDR.from_nodes_threshold(self.prediction, threshold=0.5).reduce(self.num_neurons_per_minicolumn))
            return self.create_output(self.prediction, generated, True)

        # train generative
        reduced_pred = SDR.from_nodes_threshold(self.prediction, threshold=0.5).reduce(self.num_neurons_per_minicolumn)
        if train: self.attractors.process(input_sdr, reduced_pred)
        # pred_gen = self.generate_from(reduced_pred, input_sdr.add_noise(n=5))
        # if pred_gen.overlap(input_sdr) < 8:
        #     self.predict_start(input_sdr_expanded)
        #     generated = None if gen==False else self.generate(SDR.from_nodes_threshold(self.prediction, threshold=0.5).reduce(self.num_neurons_per_minicolumn))
        #     return self.create_output(self.prediction, generated, True)


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
        

class Layer4Continual():
    def __init__(self, 
                 num_base_neurons=512, 
                 num_neurons_per_minicolumn=4,
                 sparsity=10,
                 connections_density=0.5,
                 connections_decay=0.0,
                 learning_rate=1,
                 ):

        self.num_base_neurons = num_base_neurons
        self.num_neurons_per_minicolumn = num_neurons_per_minicolumn
        self.sparsity = sparsity
        self.num_neurons = self.num_base_neurons*self.num_neurons_per_minicolumn

        self.connections = Connections(self.num_neurons, self.num_neurons, connections_density=connections_density, connections_decay=connections_decay, learning_rate=learning_rate)
        self.start_sdr = self.create_start_sdr()

        self.parameters = ['connections', 'start_sdr']
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

    def create_output(self, predicted):
        output = dict(
                predicted = SDR.from_nodes_threshold(self.prediction, threshold=0.5).reduce(self.num_neurons_per_minicolumn)
                )
        return output

    def __call__(self, input_sdr, train=True, gen=False):

        input_sdr_expanded = input_sdr.expand(self.num_neurons_per_minicolumn)

        if self.prediction == None:
            self.predict_start(input_sdr_expanded)
            return self.create_output(self.prediction)


        processed_input, boundary = self.process_input(input_sdr_expanded, self.prediction)
        counter = 0
        while boundary and train and counter<=100:
            self.connections.train(self.prev_sdr, processed_input)
            prediction = self.connections(self.prev_sdr)
            pred_sdr = SDR.from_nodes_threshold(prediction, threshold=0.5)
            overlap = pred_sdr.overlap(processed_input)
            boundary = overlap < self.sparsity
            counter += 1

        self.prediction = self.connections(processed_input)
        self.prev_sdr = processed_input

        return self.create_output(self.prediction)


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

    num_base_neurons = 256
    num_neurons_per_minicolumn=16
    sparsity = 10
    connections_density=0.5
    connections_decay=1e-4
    layer4 = Layer4Continual(num_base_neurons=num_base_neurons, 
                    num_neurons_per_minicolumn=num_neurons_per_minicolumn, 
                    sparsity=sparsity, 
                    connections_density=connections_density,
                    connections_decay=connections_decay)

    sdrs_vocab = [SDR(N=num_base_neurons, S=sparsity) for _ in range(20)] 
    sdrs_ix = torch.randint(0,20, size=(100,))
    sdrs = [sdrs_vocab[i.item()] for i in sdrs_ix]


    for sdr in tqdm(sdrs):
        layer4(sdr)
    layer4.reset()

    res_seq = sdrs[0]
    for i in range(1,len(sdrs)-1):
        res_seq = layer4(res_seq, train=False)['predicted']
        print(res_seq, sdrs[i])

    quit()






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

