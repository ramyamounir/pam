import torch
from glob import glob
from tqdm import tqdm
import os

import sys; sys.path.append('./')
from basics import SDR
from experiments.base.generate_data import Generator
from experiments.base.base_models import LSTMSequenceMemory, TransformerSequenceMemory, GRUSequenceMemory, SequenceMemory 
from utils import checkdir

from dask import delayed as dd
from dask import compute as dc


class Runner():

    def __init__(self, 
                 seq_file, 
                 L, 
                 num_gen,
                 cap,
                 stream,
                 model_type,
                 lr=1.0
                 ):
        self.seq_file = seq_file
        self.L = L
        self.num_gen = num_gen
        self.cap = cap
        self.stream = stream
        self.model_type = model_type
        self.lr = lr

        self.gen = Generator.load_from(seq_file)
        self.N = self.gen.vocab.dim
        self.S = self.gen.vocab.sparsity

        self.shuffle = False
        self.epochs = max(int(cap*(1.0-stream)),1)
        self.persistance = max(int(cap*stream),1)


        self.results = {}

    def create_model(self):
        if self.model_type == 'LSTM':
            memory = LSTMSequenceMemory(dim=self.N, num_layers=self.L)
        elif self.model_type == 'Transformer':
            memory = TransformerSequenceMemory(dim=self.N, num_layers=self.L, max_seq_len=self.gen.seqs.shape[-1])
        elif self.model_type == 'GRU':
            memory = GRUSequenceMemory(dim=self.N, num_layers=self.L)
        else:
            raise NotImplementedError('Model type not found')

        return SequenceMemory(memory=memory, pred_threshold=0.01, persistance=self.persistance, accum=1.0)

    def store(self, dataset_ix):
        self.results[f'{str(dataset_ix).zfill(2)}'] = self.test()

    def test(self):

        results = []

        if self.test_type == 'AP':

            tmp_sdr = []
            for ix, (sdr) in enumerate(self.gen.get_stream_full(shuffle=False)):
                if ix == 0: tmp_sdr.extend(sdr)
                sdr_input = torch.stack([s.bin.float() for s in tmp_sdr])
                pred = self.model.memory(sdr_input)
                pred_sdr = SDR.from_bin((pred.squeeze(0)>0.5).bool())
                results.append(pred_sdr.save())
                tmp_sdr.append(pred_sdr)

        elif self.test_type == 'NA':
            for ix, (sdr) in enumerate(self.gen.get_stream_full(shuffle=False)):
                sdr_input = torch.stack([s.bin.float() for s in sdr])
                pred = self.model.memory(sdr_input)
                pred_sdr = SDR.from_bin((pred.squeeze(0)>0.5).bool())
                results.append(pred_sdr.save())
        else:
            raise NotImplementedError("Kind not implemented")

        return results

    def run(self):

        for dataset_ix in range(self.gen.seqs.shape[0]):
            self.gen.set_dataset(dataset_ix)
            self.model = self.create_model()
            for e in tqdm(range(self.epochs)):
                for ix, (sdr) in enumerate(self.gen.get_stream_full(shuffle=self.shuffle)):
                    sdr_input = torch.stack([s.bin.float() for s in sdr])
                    prediction, loss = self.model(sdr_input)

            self.store(dataset_ix)

        return self.results



def run_experiment(args, exp_id):

    exp_id = str(exp_id).zfill(3)
    print(f'started experiment {exp_id} . . .')
    save_dir = os.path.join(save_base_dir, exp_id)
    checkdir(save_dir)

    runner = Runner(args['seq_file'], 
                    args['L'], 
                    args['num_gen'],
                    args['cap'],
                    args['stream'],
                    args['model_type'],
                    args['lr'],
                    )


    results = runner.run()

    args.update(
            dict(vocab_path=runner.gen.vocab_path,
                 vocab_len=runner.gen.vocab_len, 
                 len_seq=runner.gen.len_seq, 
                 num_seq=runner.gen.num_seq, 
                 num_datasets=runner.gen.num_datasets)
            )

    to_save = dict(args=args, vals=results)
    torch.save(to_save, os.path.join(save_dir, 'result.pth'))
    print(f'finished experiment {exp_id} . . .')



save_base_dir = 'results/sequence_generation/others/run_001'
assert checkdir(save_base_dir, careful=False), f'path {save_base_dir} exists'


# test_types = ['AP']
# model_types = ['GRU', 'Transformer', 'LSTM']
# L_range = dict(GRU=[1,2], LSTM=[1,2], Transformer=[3,6])
# stream_range = torch.linspace(0.0, 1.0, 5)
# seq_files = sorted(glob('results/sequence_generation/gt/voc_001/seq_*.pth'))
# args = dict(lr=1e-3, cap=100)


model_types = ['GRU']
L_range = dict(GRU=[1], LSTM=[1,2], Transformer=[3,6])
stream_range = [0.0]
seq_files = sorted(glob('results/sequence_generation/gt/voc_001/seq_*.pth'))[0:1]
args = dict(lr=1e-3, cap=100, num_gen=10)

fns = []
for model_type in model_types:
    for L in L_range[model_type]:
        for seq_file in seq_files:
            for stream in stream_range:
                args.update(dict(L=L, model_type=model_type, seq_file=seq_file, stream=stream))
                fns.append(dd(run_experiment)(args, len(fns)))


dc(*fns)


