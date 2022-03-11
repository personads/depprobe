#!/usr/bin/python3

import argparse, os

from collections import defaultdict

import numpy as np
import torch

from scipy.linalg import subspace_angles


def parse_arguments():
    arg_parser = argparse.ArgumentParser(description='Subspace Angles')
    arg_parser.add_argument('exp_paths', nargs='+', help='paths to experiments')
    return arg_parser.parse_args()


def main():
    args = parse_arguments()

    probe_ids = [
        '_arc._transform.weight', '_arc._distance_transform.weight',
        '_arc._depth_transform.weight',
        '_lbl._mlp.weight'
    ]
    probes = defaultdict(list)  # {probe_id: [probe_exp0, probe_exp1, ...]}
    experiments = []

    # load probes from experiments
    for exp_dir in args.exp_paths:
        print(f"Loading experiment in '{exp_dir}'...")
        experiments.append(os.path.basename(exp_dir).split('-')[0])
        checkpoint = torch.load(os.path.join(exp_dir, 'best.tar'))
        # extract probes
        for pid in probe_ids:
            if pid not in checkpoint['parser_state']: continue
            probes[pid].append(checkpoint['parser_state'][pid].cpu().numpy())
            print(f"Extracted probe '{pid}' {probes[pid][-1].shape}.")

    # iterate over probes
    for pid in sorted(probes):
        print(f"Probe '{pid}':")
        # calculate pairwise SSAs for all experiments
        print('Language\t' + '\t'.join(experiments))
        for eidx1 in range(len(experiments)):
            # language row
            print(experiments[eidx1], end='')
            for eidx2 in range(len(experiments)):
                ssa = np.mean(subspace_angles(probes[pid][eidx1].T, probes[pid][eidx2].T))
                print(f'\t{ssa:.4f}', end='')
            print()


if __name__ == '__main__':
    main()
