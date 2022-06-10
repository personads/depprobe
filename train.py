#!/usr/bin/python3

import argparse, logging, os, sys, time

from utils.setup import *
from utils.dataset import DepSpaceDataset


def parse_arguments():
	arg_parser = argparse.ArgumentParser(description='Universal Dependencies - Embedding Space Parsing')
	arg_parser.add_argument('ud_path', help='path to Universal Dependencies directory')
	arg_parser.add_argument('out_path', help='path to output directory')
	# parser setup
	arg_parser.add_argument(
		'-lm', '--language_model', default='bert-base-multilingual-cased',
		help='language model name in the transformers library (default: bert-base-multilingual-cased')
	arg_parser.add_argument(
		'-el', '--embedding_layers', nargs='+', type=int, default=[6, 7],
		help='list of embedding layers (0: WordPiece -> 12: Layer 12, default: [6, 7])')
	arg_parser.add_argument(
		'-ec', '--embedding_cache',
		help='path to pre-computed embedding cache or set to "local" for in-memory caching (default: None)')
	arg_parser.add_argument(
		'-ds', '--dependency_size', type=int, default=128,
		help='dimensionality of dependency space transformation (default: 128)')
	arg_parser.add_argument(
		'-pt', '--parser_type', default='depprobe', choices=['structural', 'directed', 'depprobe', 'depprobe-mix'],
		help='parser type (default: depprobe)')
	arg_parser.add_argument(
		'-pd', '--parser_decode', default=False, action='store_true',
		help='set flag to decode parses during training (default: False)')
	# experiment setup
	arg_parser.add_argument(
		'-s', '--split', help='path to data split definition pickle (default: None - full UD)')
	arg_parser.add_argument(
		'-td', '--treebank_directory', default=False, action='store_true',
		help='set flag to load single treebank from directory instead of mix of treebanks from split (default: False)')
	arg_parser.add_argument(
		'-e', '--epochs', type=int, default=100, help='maximum number of epochs (default: 100)')
	arg_parser.add_argument(
		'-es', '--early_stop', type=int, default=5, help='maximum number of epochs without improvement (default: 5, -1 to disable)')
	arg_parser.add_argument(
		'-bs', '--batch_size', type=int, default=32, help='maximum number of sentences per batch (default: 64)')
	arg_parser.add_argument(
		'-lr', '--learning_rate', type=float, default=1e-3, help='learning rate (default: 1e-3)')
	arg_parser.add_argument(
		'-rs', '--seed', type=int, default=42, help='seed for probabilistic components (default: 42)')
	return arg_parser.parse_args()


def main():
	args = parse_arguments()

	# check if output dir exists
	setup_output_directory(args.out_path)

	# setup logging
	setup_logging(os.path.join(args.out_path, 'train.log'))

	# set random seeds
	np.random.seed(args.seed)
	torch.random.manual_seed(args.seed)
	transformers.set_seed(args.seed)

	# setup UD data
	ud, splits, rel_map = setup_data(args.ud_path, args.split, args.treebank_directory)
	train_data = DepSpaceDataset(ud, rel_map, splits['train'], args.batch_size)
	logging.info(f"Loaded training split with {len(train_data)} sentences.")
	# load dev split if early stopping is enabled
	if args.early_stop < 0:
		logging.info("Early stopping disabled. Not loading dev data.")
	else:
		eval_data = DepSpaceDataset(ud, rel_map, splits['dev'], args.batch_size)
		logging.info(f"Loaded dev split with {len(eval_data)} sentences.")

	# setup parser model
	parser = setup_model(
		lm_name=args.language_model, dep_dim=args.dependency_size,
		parser_type=args.parser_type,
		emb_layers=args.embedding_layers,
		emb_cache=args.embedding_cache
	)

	# setup loss
	criterion = setup_criterion(parser_type=args.parser_type)

	# setup optimizer
	optimizer = torch.optim.AdamW(params=parser.get_trainable_parameters(), lr=args.learning_rate)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=0)
	logging.info(f"Optimizing using {optimizer.__class__.__name__} with learning rate {args.learning_rate}.")
	logging.info(f"Scheduler {scheduler.__class__.__name__} reduces learning rate by 0.1 after 1 epoch without improvement.")

	# main training loop
	stats = defaultdict(list)
	stats['time'].append(time.time())
	for ep_idx in range(args.epochs):
		# iterate over batches in training split
		cur_stats = run(
			parser, criterion, optimizer,
			train_data, mode='train', decode=args.parser_decode
		)

		# store and print statistics
		statistics('train', stats, cur_stats, ep_idx, args.epochs)

		# save most recent model
		path = os.path.join(args.out_path, 'newest.tar')
		save_checkpoint(parser, optimizer, ep_idx, stats, path)
		logging.info(f"Saved model from epoch {ep_idx + 1} to '{path}'.")

		# continue to next epoch if early stopping is disabled
		if args.early_stop < 0:
			continue

		# iterate over batches in dev split
		cur_stats = run(
			parser, criterion, None,
			eval_data, mode='eval', decode=True
		)
		stats['time'].append(time.time())

		# store and print statistics
		statistics('eval', stats, cur_stats, ep_idx, args.epochs)
		cur_eval_loss = stats['eval/loss'][-1]

		# save best model
		if cur_eval_loss <= min(stats['eval/loss']):
			path = os.path.join(args.out_path, 'best.tar')
			save_checkpoint(parser, optimizer, ep_idx, stats, path)
			logging.info(f"Saved model with best loss {cur_eval_loss:.4f} to '{path}'.")

		# update scheduler
		scheduler.step(cur_eval_loss)
		# check for early stopping
		if (ep_idx - stats['eval/loss'].index(min(stats['eval/loss']))) >= args.early_stop:
			logging.info(f"No improvement since {args.early_stop} epochs ({min(stats['eval/loss']):.4f} loss). Early stop.")
			break

		stats['time'].append(time.time())

	logging.info(f"Training completed after {ep_idx + 1} epochs and {stats['time'][-1] - stats['time'][0]:.2f} seconds.")
	logging.info(f"Total training time {sum(stats['train/time']):.2f} seconds across {sum(stats['train/tokens'])} ({sum(stats['train/time'])/sum(stats['train/tokens']):.2f} tokens/sec).")


if __name__ == '__main__':
	main()
