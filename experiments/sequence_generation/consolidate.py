import sys; sys.path.append('./')
from experiments.base.create_vocab import Vocab
from basics import SDR
import torch
from glob import glob
import os, re
from tqdm import tqdm


def decode_seqs(vocab, seqs_ix):
    decoded = {i:{} for i in range(seqs_ix.shape[0])}
    for dataset_ix in range(seqs_ix.shape[0]):
        for seq_ix in range(seqs_ix[dataset_ix].shape[0]):
            decoded[dataset_ix][seq_ix] = vocab.getsdrs(seqs_ix[dataset_ix][seq_ix])
    return decoded


def calc_iou_pairs(pred_sdrs, gt_sdrs):

    def iou(a, b):
        if not isinstance(a, SDR):
            sdr_a = SDR(N=256,S=10)
            sdr_a.load(a)
        else: sdr_a = a
        return len(sdr_a.intersect(b)) / len(sdr_a+b)

    return torch.mean(torch.tensor([iou(pred_sdr, gt_sdr) for pred_sdr, gt_sdr in zip(pred_sdrs, gt_sdrs)]))

def process_datatset(result):
    iou = torch.tensor([row[0] for row in result])
    ix = torch.tensor([row[1] for row in result])
    return iou.mean(), torch.isin(torch.arange(10.0), ix).sum()/10.0

def compare(args, gt_seqs, pred_seqs):

    num_datasets = args['num_datasets']
    num_seqs = args['num_seq']
    num_gen = args['num_gen']

    result = {}
    for dataset_ix in tqdm(range(num_datasets)):

        gt_set = gt_seqs[dataset_ix]
        pred_set = pred_seqs[f'{str(dataset_ix).zfill(2)}']

        comparison = []
        for pred_ix, pred in enumerate(pred_set, start=1):
            ious = []
            for gt in gt_set.values():
                ious.append(calc_iou_pairs(pred, gt))
            
            argmax = torch.argmax(torch.tensor(ious)).item()
            max_iou = ious[argmax].item()
            if max_iou>=0.9:
                comparison.append([max_iou, argmax])
            else:
                comparison.append([max_iou, -1])

        result[dataset_ix] = comparison


    ious, percents = {int(i//num_seqs):[] for i in range(num_seqs,num_seqs*num_gen+1,num_seqs)}, \
                     {int(i//num_seqs):[] for i in range(num_seqs,num_seqs*num_gen+1,num_seqs)}

    for dataset_ix in tqdm(range(num_datasets)):
        for r in range(num_seqs-1,num_seqs*num_gen,num_seqs):
            iou, percent = process_datatset(result[dataset_ix][:r+1])
            mul = int((r+1)//num_seqs)
            ious[mul].append(iou.item())
            percents[mul].append(percent.item())

    return args, ious, percents


def get_results(result_file):

    results_loaded = torch.load(result_file)

    vocab = Vocab.load_from(results_loaded['args']['vocab_path'])
    gt_seqs_ix = torch.load(results_loaded['args']['seq_file'])['seqs']
    gt_seqs = decode_seqs(vocab, gt_seqs_ix)

    pred_seqs = results_loaded['vals']

    return compare(results_loaded['args'], gt_seqs, pred_seqs)


model = 'solo'
run = 'run_001'
results_folder = f'results/sequence_generation/{model}/{run}'
results_files = sorted(glob(f'{results_folder}/*/*.pth'))

consolidated_results = {}
for result_file in tqdm(results_files):
    args, ious, percents = get_results(result_file)
    consolidated_results[result_file] = [args, ious, percents]

# for k, v in consolidated_results.items():
#     print(f'Test: {v[0]["test_type"]}, Seq_len: {v[0]["len_seq"]}, K: {v[0]["K"]} ', v[1])

torch.save(consolidated_results, os.path.join(results_folder, 'consolidated_results.pth'))

