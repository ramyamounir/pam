import sys; sys.path.append('./')
from experiments.base.create_vocab import Vocab
from basics import SDR
import torch
from glob import glob
import os, re

vocab = Vocab.load_from('results/sequence_memory/gt/sparse_100_128_10.pth')

def decode_seqs(seqs):
    decoded = {}
    for dataset_ix in range(seqs.shape[0]):
        decoded[dataset_ix] = vocab.getsdrs(seqs[dataset_ix][0])
    return decoded


def calc_iou_pairs(pred_sdrs, gt_sdrs):

    def iou(a, b):
        sdr_a = SDR(N=128,S=12)
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
            tag = f'{str(dataset_ix).zfill(2)},{str(step).zfill(4)}'
            gt_sdrs = gt_seqs[dataset_ix]
            pred_sdrs = pred_seqs[tag]
            avg_iou = calc_iou_pairs(pred_sdrs[:-1], gt_sdrs[1:])
            steps_list.append(avg_iou.item())
        result.append(torch.tensor(steps_list))
    return torch.stack(result)


def get_results(result_file):
    path_split = result_file.split('/')
    seq_file = re.sub( path_split[2], 'gt', os.path.join(*path_split[:4])) + '.pth'
    gt_seqs = decode_seqs(torch.load(seq_file)['seqs'])
    pred_seqs = torch.load(result_file)

    return compare(gt_seqs, pred_seqs)



model = 'lstm'
results_folder = f'results/sequence_memory/{model}'
results_files = glob(os.path.join(results_folder, '*/*/*.pth'))

consolidated_results = {}
for result_file in results_files:
    result = get_results(result_file)
    consolidated_results[result_file] = result

print(consolidated_results)
quit()
torch.save(consolidated_results, os.path.join(results_folder, 'full_results.pth'))

