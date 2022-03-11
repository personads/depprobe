#!/usr/bin/python3

import argparse, logging, os, sys

import torch
import transformers

# local imports
from data.ud import *
from utils.setup import *


def parse_arguments():
	arg_parser = argparse.ArgumentParser(description='Probe Prediction')
	arg_parser.add_argument('exp_path', help='path to experiment directory')
	arg_parser.add_argument('target', help='path to target CoNLL-U file')
	arg_parser.add_argument('out_path', help='path to output CoNLL-U file')
	arg_parser.add_argument(
		'-lm', '--language_model', default='bert-base-multilingual-cased',
		help='language model name in the transformers library (default: bert-base-multilingual-cased')
	arg_parser.add_argument(
		'-el', '--embedding_layers', nargs='+', type=int, default=[6, 7],
		help='list of embedding layers (0: WordPiece -> 12: Layer 12, default: [6, 7])')
	arg_parser.add_argument(
		'-ds', '--dependency_size', type=int, default=128,
		help='dimensionality of dependency space transformation (default: 128)')
	arg_parser.add_argument(
		'-pt', '--parser_type', default='depprobe', choices=['structural', 'directed', 'depprobe'],
		help='parser type (default: depprobe)')
	arg_parser.add_argument(
		'-bs', '--batch_size', type=int, default=64, help='maximum number of sentences per batch (default: 64)')
	return arg_parser.parse_args()


def main():
	args = parse_arguments()

	# check if checkpoint exists
	if not os.path.exists(args.exp_path):
		print(r"[Error] Could not find experiment directory '{args.exp_path}'. Exiting.")
		exit(1)
	# check if output file exists
	if os.path.exists(args.out_path):
		response = None
		while response not in ['y', 'n']:
			response = input(f"File '{args.out_path}' already exists. Overwrite? [y/n] ")
		if response == 'n':
			exit(1)

	# setup logging
	setup_logging(os.path.join(args.exp_path, 'predict.log'))

	# setup target UD data
	ud_target = UniversalDependenciesTreebank.from_conllu(args.target, name=os.path.basename(args.target))
	logging.info(f"Loaded target {ud_target}.")

	# load state dict of best checkpoint
	checkpoint = torch.load(os.path.join(args.exp_path, 'best.tar'))

	# setup parser model
	parser = setup_model(
		lm_name=args.language_model, dep_dim=args.dependency_size,
		parser_type=args.parser_type,
		emb_layers=args.embedding_layers,
		state_dict=checkpoint['parser_state'])
	parser.eval()

	# main inference loop
	cursor = 0
	while cursor < len(ud_target):
		# set up batch
		start_idx = cursor
		end_idx = min(start_idx + args.batch_size, len(ud_target))
		cursor = end_idx
		sentences = [s.to_words() for s in ud_target[start_idx:end_idx]]

		# forward pass
		with torch.no_grad():
			# get graphs and label logits from parser
			parse = parser(sentences)

		# store predicted dependency data
		for sidx, udidx in enumerate(range(start_idx, end_idx)):
			cur_sentence = ud_target[udidx]
			for widx, word in enumerate(cur_sentence.to_words(as_str=False)):
				# get head predictions
				head = int(parse['graphs'][sidx, widx]) + 1
				word.head = head
				# get label predictions (for non-probe models)
				if 'labels' in parse:
					label = UD_RELATION_TYPES[int(parse['labels'][sidx, widx])]
					word.deprel = label

		# print progress
		sys.stdout.write(f"\r[{(cursor*100)/len(ud_target):.2f}%] Predicting...")
		sys.stdout.flush()
	print("\r", end='')

	# export to CoNLL-U
	with open(args.out_path, 'w', encoding='utf8') as fp:
		fp.write(ud_target.to_conllu())

	logging.info(f"Predicted dependency heads and labels for {len(ud_target)} sentences. Saved output to {args.out_path}.")


if __name__ == '__main__':
	main()
