import torch
from glob import glob
from tqdm import tqdm
import os

import sys; sys.path.append('./')
from basics import SDR
from experiments.base.generate_data import Generator
from experiments.base.base_models import LSTMSequenceMemory, SequenceMemory, SequenceMemoryStreaming
from utils import checkdir

from dask import delayed as dd
from dask import compute as dc


class Runner():

    def __init__(self, 
                 seq_file, 
                 L, 
                 test_type,
                 train_type,
                 lr=1.0
                 ):
        self.seq_file = seq_file
        self.L = L
        self.test_type = test_type
        self.train_type = train_type
        self.lr = lr

        self.epochs = 1000

        self.gen = Generator.load_from(seq_file)
        self.N = self.gen.vocab.dim
        self.S = self.gen.vocab.sparsity

        self.results = {}

    def create_model(self):
        memory = LSTMSequenceMemory(dim=self.N, num_layers=self.L)
        if train_type == 'B':
            return SequenceMemory(memory=memory, pred_threshold=0.01)
        elif train_type == 'S':
            return SequenceMemoryStreaming(memory=memory, pred_threshold=0.01)


    def store(self, dataset_ix, epoch):
        tag = f'{str(dataset_ix).zfill(2)},{str(epoch).zfill(len(str(self.epochs)))}'
        self.results[tag] = self.test()

    def test(self):

        self.model.reset()
        results = []


        if self.test_type == 'AP':

            tmp_sdr = []
            for ix, (sdr) in enumerate(self.gen.get_stream_full(shuffle=False)):
                if ix == 0: tmp_sdr.append(sdr[0])
                sdr_input = torch.stack([s.bin.float() for s in tmp_sdr])
                pred = self.model.memory(sdr_input)
                pred_sdr = SDR.from_bin((pred.squeeze(0)>0.5).bool())
                results.append(pred_sdr.save())
                tmp_sdr.append(pred_sdr)

        elif self.test_type == 'NA':
            for ix, (sdr) in enumerate(self.gen.get_stream_full(shuffle=False)):
                sdr_input = torch.stack([s.bin.float() for s in sdr])
                prediction = self.model(sdr_input)
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
                losses = 0.0

                counter = 0
                for ix, (sdr) in enumerate(self.gen.get_stream_full(shuffle=False)):
                    sdr_input = torch.stack([s.bin.float() for s in sdr])
                    prediction, loss = self.model(sdr_input)
                    losses += loss
                    counter += 1

                print(losses/counter)
            quit()

                # self.store(dataset_ix, e)
                # self.model.reset()

        return self.results



def run_experiment(args, exp_id):

    exp_id = str(exp_id).zfill(3)
    print(f'started experiment {exp_id} . . .')
    save_dir = os.path.join(save_base_dir, exp_id)
    checkdir(save_dir)

    runner = Runner(args['seq_file'], 
                        args['L'], 
                        args['test_type'],
                        args['train_type'],
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



save_base_dir = 'results/sequence_memory/lstm'
checkdir(save_base_dir)

# L_range = [1,2]
# test_types = ['AP', 'NA']
# seq_files = sorted(glob('results/sequence_memory/gt/seq_*.pth'))[:3]
# args = dict(lr=1e-2)

L_range = [1]
test_types = ['AP']
train_types = ['B']
seq_files = sorted(glob('results/sequence_memory/gt/seq_*.pth'))[:1]
args = dict(lr=1e-2)


fns = []
for L in L_range:
    for test_type in test_types:
        for train_type in train_types:
            for seq_file in seq_files:
                args.update(dict(L=L, test_type=test_type, train_type=train_type, seq_file=seq_file))
                fns.append(dd(run_experiment)(args, len(fns)))


dc(*fns)


