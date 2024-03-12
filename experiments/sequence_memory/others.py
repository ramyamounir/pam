import torch
from glob import glob
from tqdm import tqdm
import os

import sys; sys.path.append('./')
from basics import SDR
from experiments.base.generate_data import Generator
from experiments.base.base_models import LSTMSequenceMemory, TransformerSequenceMemory, GRUSequenceMemory, SequenceMemory, GRUSequenceMemoryCell 
from utils import checkdir

from dask import delayed as dd
from dask import compute as dc


class Runner():

    def __init__(self, 
                 seq_file, 
                 L, 
                 test_type,
                 model_type,
                 save_epochs,
                 lr
                 ):

        self.seq_file = seq_file
        self.L = L
        self.test_type = test_type
        self.model_type = model_type
        self.lr = lr
        self.save_epochs = save_epochs

        self.gen = Generator.load_from(seq_file)
        self.N = self.gen.vocab.dim
        self.S = self.gen.vocab.sparsity

        self.shuffle = True
        self.epochs = save_epochs[-1]

        self.results = {}

    def create_model(self):
        if self.model_type == 'LSTM':
            memory = LSTMSequenceMemory(dim=self.N, num_layers=self.L)
        elif self.model_type == 'Transformer':
            memory = TransformerSequenceMemory(dim=self.N, num_layers=self.L, max_seq_len=self.gen.seqs.shape[-1])
        elif self.model_type == 'GRU':
            memory = GRUSequenceMemoryCell(dim=self.N, num_layers=self.L)
        else:
            raise NotImplementedError('Model type not found')

        return SequenceMemory(memory=memory, lr=self.lr, pred_threshold=0.01, accum=1.0)

    def store(self, dataset_ix, epoch):

        if epoch not in self.results:
            self.results[epoch] = {}

        self.results[epoch][dataset_ix] = self.test()

    def test(self):

        results = []

        if self.test_type == 'AP':

            h = None
            for ix, (sdr) in enumerate(self.gen.get_stream_AR(shuffle=False)):
                if ix==0: 
                    # tmp_sdr = torch.stack([s.bin.float() for s in sdr])
                    pred = torch.stack([s.bin.float() for s in sdr])

                pred, h = self.model.memory(pred, h)#[[-1]]
                # tmp_sdr= torch.cat([tmp_sdr, pred])

                pred = SDR.from_bin((pred.squeeze(0)>0.5).bool())
                results.append(pred.save())

                pred = pred.bin.float().unsqueeze(0)

                # pred_sdr = SDR.from_bin((pred.squeeze(0)>0.5).bool())
                # results.append(pred_sdr.save())



            # tmp_sdr = []
            # for ix, (sdr) in enumerate(self.gen.get_stream_AR(shuffle=False)):
            #     if ix == 0: tmp_sdr.extend(sdr)

            #     sdr_input = torch.stack([s.bin.float() for s in tmp_sdr])
            #     pred = self.model.memory(sdr_input)[[-1]]

            #     pred_sdr = SDR.from_bin((pred.squeeze(0)>0.5).bool())
            #     tmp_sdr.append(pred_sdr)

            #     results.append(pred_sdr.save())

        elif self.test_type == 'NA':
            for ix, (sdr) in enumerate(self.gen.get_stream_AR(shuffle=False)):
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
                for ix, (sdr) in enumerate(self.gen.get_stream_full()):
                    sdr_input = torch.stack([s.bin.float() for s in sdr])
                    prediction, loss = self.model(sdr_input)

                if e+1 in self.save_epochs:
                    self.store(dataset_ix, e+1)
            break

        return self.results



def run_experiment(args, exp_id):

    exp_id = str(exp_id).zfill(3)
    print(f'started experiment {exp_id} . . .')
    save_dir = os.path.join(save_base_dir, exp_id)
    checkdir(save_dir)

    runner = Runner(args['seq_file'], 
                    args['L'], 
                    args['test_type'],
                    args['model_type'],
                    args['save_epochs'],
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



save_base_dir = 'results/sequence_memory/others/run_001'
assert checkdir(save_base_dir, careful=False), f'path {save_base_dir} exists'


# test_types = ['AP', 'NA']
# model_types = ['GRU', 'Transformer', 'LSTM']
# L_range = dict(GRU=[1,2], LSTM=[1,2], Transformer=[3,6])
# stream_range = torch.linspace(0.0, 1.0, 5)
# seq_files = sorted(glob('results/sequence_memory/gt/voc_001/seq_*.pth'))
# args = dict(lr=1e-3, cap=100)


# test_types = ['AP']
# model_types = ['GRU', 'Transformer', 'LSTM']
# L_range = dict(GRU=[1], LSTM=[1], Transformer=[2])
# seq_files = sorted(glob('results/sequence_memory/gt/voc_001/seq_*.pth'))
# args = dict(lr=1e-3, save_epochs=[1, 50, 100, 1000])


test_types = ['AP']
model_types = ['GRU']
L_range = dict(GRU=[1], LSTM=[1], Transformer=[2])
seq_files = sorted(glob('results/sequence_memory/gt/voc_001/seq_*.pth'))[:1]
args = dict(lr=1e-3, save_epochs=[1, 10, 20, 25, 30])


fns = []
for model_type in model_types:
    for L in L_range[model_type]:
        for test_type in test_types:
            for seq_file in seq_files:
                args.update(dict(L=L, model_type=model_type, test_type=test_type, seq_file=seq_file))
                fns.append(dd(run_experiment)(args, len(fns)))


dc(*fns)


