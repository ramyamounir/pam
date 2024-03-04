import torch
from glob import glob
from tqdm import tqdm

import sys; sys.path.append('./')
from basics import SDR
from experiments.base.generate_data import Generator
from experiments.base.base_models import LSTMSequenceMemory, SequenceMemory





results = {}

def store(seq_file, dataset_ix, neurons_per_minicolumn, epoch, res):
    tag = f'{str(dataset_ix).zfill(2)},{str(neurons_per_minicolumn).zfill(2)},{epoch}'
    if seq_file not in results: results[seq_file] = {}
    if tag not in results[seq_file]: results[seq_file][tag] = []

    results[seq_file][tag].append([res['predicted'].save(), res['generated'].save(), res['boundary']])


for seq_file in sorted(glob('results/sequence_memory/seq_*.pth')):
    seq_file = sorted(glob('results/sequence_memory/seq_*.pth'))[-1]
    print(f'seq file = {seq_file}')

    gen = Generator.load_from(seq_file)

    for dataset_ix in range(gen.seqs.shape[0]):
        print(f'dataset = {dataset_ix}')
        gen.set_dataset(dataset_ix)

        for L in [1, 2, 3]:
            print(f'L = {L}')
            memory = LSTMSequenceMemory(dim=128, num_layers=L)
            model = SequenceMemory(memory=memory, pred_threshold=0.1, train_every=1)

            for e in tqdm(range(300)):

                for sdr, b in gen:
                    pred, boundary, loss = model(sdr.bin.float().unsqueeze(0))
                    print(loss)
                    # if e%10==0: store(seq_file, dataset_ix, K, e, res)

                model.reset()
            print(pred)
            quit()

# need to save autoregressive predictive, autoregressive generated, non-autoregressive sequences
torch.save(results, 'results/sequence_memory/lstm_results.pth')



