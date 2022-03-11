#!/usr/bin/python3

import argparse, os, pickle, random, sys

# local imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data.ud import *
from utils.setup import *


def parse_arguments():
	arg_parser = argparse.ArgumentParser(description='Universal Dependencies - Filtered Corpus Creation')
	arg_parser.add_argument('ud_path', help='path to Universal Dependencies directory')
	arg_parser.add_argument('split', help='path to split definition (pickle)')
	arg_parser.add_argument('out_path', help='path to output directory')
	# predicted distribution
	arg_parser.add_argument('-dd', '--domain_distribution', help='path to domain distribution pickle (overrides treebanks metadata) (default: None)')
	# filters
	arg_parser.add_argument('-cm', '--combination_mode', default='all', choices=['all', 'any'], help='combination method of filters (default: all)')
	arg_parser.add_argument('-il', '--include_languages', help='comma-separated list of languages to include (default: All)')
	arg_parser.add_argument('-el', '--exclude_languages', help='comma-separated list of languages to exclude (default: None)')
	arg_parser.add_argument('-it', '--include_treebanks', help='comma-separated list of treebanks to include (default: All)')
	arg_parser.add_argument('-et', '--exclude_treebanks', help='comma-separated list of treebanks to exclude (default: None)')
	# random selection
	arg_parser.add_argument('-r', '--random_sample', help='comma-separated sizes of the (train,dev,test) splits\' random subsamples (default: All)')
	arg_parser.add_argument('-rs', '--random_seed', type=int, default=42, help='seed for all probabilistic components (default: 42)')
	return arg_parser.parse_args()


def load_metadata_filters(field, include_values, exclude_values):
	filters = []

	if include_values:
		include_values = [v.strip() for v in include_values.split(',')]
		filters.append(UniversalDependenciesMetadataFilter(field, include_values, mode='include'))
	if exclude_values:
		exclude_values = [v.strip() for v in exclude_values.split(',')]
		filters.append(UniversalDependenciesMetadataFilter(field, exclude_values, mode='exclude'))

	return filters


def main():
	args = parse_arguments()

	# check if output dir exists
	setup_success = setup_output_directory(args.out_path)
	if not setup_success: return

	# setup logging
	setup_logging(os.path.join(args.out_path, 'filter.log'))

	filters = []

	# load existing split
	with open(args.split, 'rb') as fp:
		split_idcs = pickle.load(fp)
	# construct index-based filter
	relevant_idcs = set(split_idcs['train']) | set(split_idcs['dev']) | set(split_idcs['test'])
	filters.append(UniversalDependenciesIndexFilter(relevant_idcs))
	logging.info(f"Loaded data splits {', '.join([f'{s}: {len(idcs)}' for s, idcs in split_idcs.items()])}.")

	# load random subsample sizes
	sample_sizes = None
	if args.random_sample:
		random.seed(args.random_seed)
		sample_sizes = [int(s.strip()) for s in args.random_sample.split(',')]
		sample_sizes = dict(zip(['train', 'dev', 'test'], sample_sizes))
		logging.info(f"Results will be subsampled to max {sample_sizes} using random seed {args.random_seed}.")

	# setup language filters
	filters += load_metadata_filters('Language', args.include_languages, args.exclude_languages)

	# setup treebank filters
	filters += load_metadata_filters('Treebank', args.include_treebanks, args.exclude_treebanks)

	# consolidate filters
	ud_filter = None
	if len(filters) > 0:
		ud_filter = UniversalDependenciesFilterCombination(filters, mode=args.combination_mode)

	# print filter overview
	logging.info(f"Filtering UD based on {args.combination_mode} of the following {len(filters)} filter(s):")
	for filter in filters:
		logging.info(f"  {filter}")

	# load Universal Dependencies
	ud = UniversalDependencies.from_directory(args.ud_path, ud_filter=ud_filter, verbose=True)
	logging.info(f"Loaded {ud} with {len(ud.get_treebanks())} treebanks.")

	# export filtered splits
	filtered_splits = {}
	for split, idcs in split_idcs.items():
		filtered_splits[split] = [idx for idx in idcs if (ud[idx] is not None)]
		# perform subsampling if relevant
		if sample_sizes:
			random.shuffle(filtered_splits[split])
			filtered_splits[split] = sorted(filtered_splits[split][:sample_sizes[split]])
	split_path = os.path.join(args.out_path, 'filtered.pkl')
	with open(split_path, 'wb') as fp:
		pickle.dump(filtered_splits, fp)
	logging.info(f"Saved filtered data splits {', '.join([f'{s}: {len(idcs)}' for s, idcs in filtered_splits.items()])}.")


if __name__ == '__main__':
	main()
