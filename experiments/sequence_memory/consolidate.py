import sys; sys.path.append('./')
from experiments.base.create_vocab import Vocab
from basics import SDR
import torch
from glob import glob
import os, re
from tqdm import tqdm


def decode_seqs(vocab, seqs_ix):
    decoded = {}
    for dataset_ix in range(seqs_ix.shape[0]):
        decoded[dataset_ix] = vocab.getsdrs(seqs_ix[dataset_ix][0])
    return decoded


def calc_iou_pairs(pred_sdrs, gt_sdrs):

    def iou(a, b):
        sdr_a = SDR(N=256,S=10)
        sdr_a.load(a)
        return len(sdr_a.intersect(b)) / len(sdr_a+b)

    return torch.mean(torch.tensor([iou(pred_sdr, gt_sdr) for pred_sdr, gt_sdr in zip(pred_sdrs, gt_sdrs)]))

def compare(gt_seqs, pred_seqs):
    num_datasets = len(gt_seqs)
    num_steps = int(len(pred_seqs)/num_datasets)
    save_every = 10

    result = []
    for dataset_ix in range(num_datasets):
        steps_list = []
        for step in range(0,num_steps*save_every,save_every):
            tag = f'{str(dataset_ix).zfill(2)}'
            gt_sdrs = gt_seqs[dataset_ix]
            pred_sdrs = pred_seqs[tag]
            avg_iou = calc_iou_pairs(pred_sdrs[:-1], gt_sdrs[1:])
            steps_list.append(avg_iou.item())
        result.append(torch.tensor(steps_list))

    return torch.stack(result)


def get_results(result_file):

    results_loaded = torch.load(result_file)

    vocab = Vocab.load_from(results_loaded['args']['vocab_path'])
    gt_seqs_ix = torch.load(results_loaded['args']['seq_file'])['seqs']
    gt_seqs = decode_seqs(vocab, gt_seqs_ix)

    pred_seqs = results_loaded['vals']

    return results_loaded['args'], compare(gt_seqs, pred_seqs)


model = 'lstm'
run = 'run_001'
results_folder = f'results/sequence_memory/{model}/{run}'
results_files = sorted(glob(f'{results_folder}/*/*.pth'))

consolidated_results = {}
for result_file in tqdm(results_files):
    args, result = get_results(result_file)
    consolidated_results[result_file] = [args, result]

# for k, v in consolidated_results.items():
#     print(f'Model: {v[0]["model_type"]}, Stream: {v[0]["stream"]}, Test: {v[0]["test_type"]}, Seq_len: {v[0]["len_seq"]}, L: {v[0]["L"]} ', v[1])
# quit()

torch.save(consolidated_results, os.path.join(results_folder, 'consolidated_results.pth'))

