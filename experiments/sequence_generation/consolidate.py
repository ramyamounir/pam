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

def compare(gt_seqs, pred_seqs):

    num_datasets = len(gt_seqs)

    result = {}
    for dataset_ix in range(num_datasets):

        gt_set = gt_seqs[dataset_ix]
        pred_set = pred_seqs[f'{str(dataset_ix).zfill(2)}']

        comparison = []
        for pred in pred_set:
            ious = []
            for gt in gt_set.values():
                ious.append(calc_iou_pairs(pred, gt))
            
            argmax = torch.argmax(torch.tensor(ious)).item()
            max_iou = ious[argmax].item()
            comparison.append([max_iou, argmax])

        result[dataset_ix] = comparison

    print(result)
    quit()

    return result


def get_results(result_file):

    results_loaded = torch.load(result_file)

    vocab = Vocab.load_from(results_loaded['args']['vocab_path'])
    gt_seqs_ix = torch.load(results_loaded['args']['seq_file'])['seqs']
    gt_seqs = decode_seqs(vocab, gt_seqs_ix)

    pred_seqs = results_loaded['vals']

    return results_loaded['args'], compare(gt_seqs, pred_seqs)


model = 'solo'
run = 'run_001'
results_folder = f'results/sequence_generation/{model}/{run}'
results_files = sorted(glob(f'{results_folder}/*/*.pth'))

consolidated_results = {}
for result_file in tqdm(results_files):
    args, result = get_results(result_file)
    consolidated_results[result_file] = [args, result]

# for k, v in consolidated_results.items():
#     print(f'Test: {v[0]["test_type"]}, Seq_len: {v[0]["len_seq"]}, K: {v[0]["K"]} ', v[1])

torch.save(consolidated_results, os.path.join(results_folder, 'consolidated_results.pth'))

