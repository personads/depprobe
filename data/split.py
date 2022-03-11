#!/usr/bin/python3

import argparse, os, pickle, re, sys

import numpy as np

from collections import OrderedDict

# local imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data.ud import *
from utils.setup import *


def parse_arguments():
	arg_parser = argparse.ArgumentParser(description='Universal Dependencies - Split Data')
	arg_parser.add_argument('ud_path', help='path to Universal Dependencies directory')
	arg_parser.add_argument('out_path', help='path to output directory')
	arg_parser.add_argument('-p', '--proportions', default='.7,.1,.2', help='train, dev, test proportions (default: ".7,.1,.2")')
	arg_parser.add_argument('-ka', '--keep_all', action='store_true', default=False, help='keep all original splits (default: False)')
	arg_parser.add_argument('-kt', '--keep_test', action='store_true', default=False, help='do not split test data into train/dev (default: False)')
	arg_parser.add_argument('-ms', '--max_sentences', type=int, default=int(2e4), help='maximum number of sentences per split except test (default: 20k)')
	arg_parser.add_argument('-rs', '--seed', type=int, default=42, help='seed for probabilistic components (default: 42)')
	return arg_parser.parse_args()

def main():
	args = parse_arguments()

	# check if output dir exists
	setup_success = setup_output_directory(args.out_path)
	if not setup_success: return

	# setup logging
	setup_logging(os.path.join(args.out_path, 'split.log'))

	# set random seed
	np.random.seed(args.seed)

	# parse split proportions
	proportions = [float(pstr) for pstr in args.proportions.split(',')]
	assert sum(proportions) == 1, f"[Error] Split proportions {proportions} do not sum up to 1."
	assert len(proportions) == 3, f"[Error] There must be three proportions (i.e. train, dev, test)."

	# load Universal Dependencies
	ud = UniversalDependencies.from_directory(args.ud_path, verbose=True)
	logging.info(f"Loaded {ud}.")

	# gather corpus indices by treebanks tb -> tbf -> corpus_idcs
	tb_split_idcs = OrderedDict()
	cursor = 0
	for tbf in ud.get_treebanks():
		# get name of current file's treebank
		tb_name = tbf.get_language() + '/' + tbf.get_treebank_name()
		if tb_name not in tb_split_idcs:
			tb_split_idcs[tb_name] = OrderedDict()
		# extract split from current file
		split_match = re.match(r'.+-(.+?)\.conllu', tbf.get_name())
		if not split_match:
			logging.warning(f"[Warning] Could not identify split of '{tbf.get_name()}'. Skipped.")
			continue
		# set tb -> split to tbf_corpus_idcs
		tb_split_idcs[tb_name][split_match[1]] = list(range(cursor, cursor + len(tbf)))
		# increment cursor by split size
		cursor += len(tbf)

	# split and subsample data
	logging.info("-" * 20)
	split_indices = {'train': [], 'dev': [], 'test': []}
	for tb, splits in tb_split_idcs.items():
		cur_split_idcs = {}
		cur_split_info = defaultdict(list)

		# case: treebank has train, dev and test splits
		if {'train', 'dev', 'test'} == set(splits):
			cur_split_idcs = splits
		# case: treebank has train and test splits
		elif {'train', 'test'} == set(splits):
			# test remains test
			cur_split_idcs['test'] = splits['test']
			# if original splits should be kept, don't re-split
			if args.keep_all:
				cur_split_idcs['train'] = splits['train']
				cur_split_idcs['dev'] = []
			# split train into train/dev
			else:
				# shuffle train data
				np.random.shuffle(splits['train'])
				# split train into train and dev (rescale proportions to train/(train+dev))
				prop_idx = round(len(splits['train']) * (proportions[0]/sum(proportions[0:2])))
				cur_split_idcs['train'] = splits['train'][:prop_idx]
				cur_split_info['train'].append(f'train[:{prop_idx}]')
				cur_split_idcs['dev'] = splits['train'][prop_idx:]
				cur_split_info['dev'].append(f'train[{prop_idx}:]')
		# case: treebank contains only test
		elif {'test'} == set(splits):
			# if test should not be split
			if args.keep_test or args.keep_all:
				# add nothing to train and dev
				cur_split_idcs['train'] = []
				cur_split_idcs['dev'] = []
				# keep test as it is
				cur_split_idcs['test'] = splits['test']
				cur_split_info['test'].append('no split')
			# if test is allowed to be split into train, dev, test
			else:
				# shuffle test data
				np.random.shuffle(splits['test'])
				# split test into train, dev, test
				cursor = 0
				for spidx, split in enumerate(['train', 'dev', 'test']):
					cursor_end = cursor + (round(len(splits['test']) * proportions[spidx]))
					# if test, include all remaining data (in case of rounding issues)
					if split == 'test': cursor_end = len(splits['test'])

					cur_split_idcs[split] = splits['test'][cursor:cursor_end]
					cur_split_info[split].append(f'test[{cursor}:{cursor_end}]')
					cursor = cursor_end
		# case: unknown splits
		else:
			logging.warning(f"  [Warning] Unknown set of splits {tuple(splits.keys())}. Skipped.")
			continue

		# subsample split if it is too large
		for split, idcs in cur_split_idcs.items():
			if split == 'test': continue
			if len(idcs) <= args.max_sentences: continue
			cur_split_idcs[split] = list(np.random.choice(idcs, args.max_sentences, replace=False))
			cur_split_info[split].append(f'sampled from {len(idcs)}')

		# append current splits to overall corpus indices
		split_indices['train'] += cur_split_idcs['train']
		split_indices['dev'] += cur_split_idcs['dev']
		split_indices['test'] += cur_split_idcs['test']

		# print statistics
		num_idcs = sum([len(cur_split_idcs['train']), len(cur_split_idcs['dev']), len(cur_split_idcs['test'])])
		logging.info(f"{tb} (n={num_idcs}):")
		for split in ['train', 'dev', 'test']:
			logging.info(f"  {split.capitalize()}: {len(cur_split_idcs[split])} sentences ({len(cur_split_idcs[split])/num_idcs:.4f})")
			if cur_split_info[split]: logging.info(f"    {' | '.join(cur_split_info[split])}")

	# print overall statistics
	num_idcs = sum([len(split_indices['train']), len(split_indices['dev']), len(split_indices['test'])])
	logging.info(f"UD (n={num_idcs}):")
	for split in ['train', 'dev', 'test']:
		logging.info(f"  {split.capitalize()}: {len(split_indices[split])} sentences ({len(split_indices[split])/num_idcs:.4f})")
		cur_languages = {ud.get_language_of_index(sidx) for sidx in split_indices[split]}
		logging.info(f"    Languages: {len(cur_languages)}")
		logging.info(f"      {', '.join(sorted(cur_languages))}")
		cur_domain_combinations = {tuple(sorted(ud.get_domains_of_index(sidx))) for sidx in split_indices[split]}
		cur_domains = {d for dc in cur_domain_combinations for d in dc}
		logging.info(f"    Domains: {len(cur_domains)}")
		logging.info(f"      {', '.join(sorted(cur_domains))}")
		logging.info(f"    Domain Combinations: {len(cur_domain_combinations)}")
		logging.info(f"      {', '.join([str(dc) for dc in sorted(cur_domain_combinations)])}")

	logging.info("-" * 20)

	# save results
	split_path = os.path.join(args.out_path, 'split.pkl')
	with open(split_path, 'wb') as fp:
		pickle.dump(split_indices, fp)
	logging.info(f"Saved split indices to '{split_path}'.")


if __name__ == '__main__':
	main()
