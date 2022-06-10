#!/usr/bin/python3

import argparse, logging, os, sys

import torch
import transformers

# local imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data.ud import *
from utils.setup import *


def parse_arguments():
	arg_parser = argparse.ArgumentParser(description='Universal Dependencies - Embedding Space Parsing Prediction')
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
		'-pt', '--parser_type', default='depprobe', choices=['structural', 'directed', 'depprobe', 'depprobe-mix'],
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
	if os.path.exists(os.path.join(args.exp_path, 'best.tar')):
		checkpoint = torch.load(os.path.join(args.exp_path, 'best.tar'))
	# otherwise, load newest checkpoint
	else:
		checkpoint = torch.load(os.path.join(args.exp_path, 'newest.tar'))

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
		error_idcs = set()

		# forward pass
		with torch.no_grad():
			# get graphs and label logits from parser
			while len(sentences) > 0:
				# try parser forward pass
				try:
					parse = parser(sentences)
					break
				# remove offending sentence from batch
				except TokenizationError as tok_err:
					error_idcs.add(tok_err.position[0] + len(error_idcs))
					sentences = sentences[:tok_err.position[0]] + sentences[tok_err.position[0] + 1:]
					logging.warning(f"Tokenization error at sentence {tok_err.position[0]}. Skipped.")
					logging.warning(tok_err)

		# re-introduce unparsed sentences
		if len(error_idcs) > 0:
			num_sentences = end_idx - start_idx
			sentence_lengths = [len(s.to_words()) for s in ud_target[start_idx:end_idx]]
			max_len = max(sentence_lengths)

			graphs = torch.zeros((num_sentences, max_len), device=parse['graphs'].device)
			labels = torch.zeros((num_sentences, max_len), device=parse['graphs'].device)

			# fill graphs and labels
			pred_cursor = 0
			for sidx in range(num_sentences):
				# fill with placeholders for unparsed sentences
				if sidx in error_idcs:
					graphs[sidx, :sentence_lengths[sidx]] = 0  # set all heads to 0
					labels[sidx, :sentence_lengths[sidx]] = -1  # set all labels to -1 (map to UNK)
				# fill with original values for parsed sentences
				else:
					graphs[pred_cursor, :sentence_lengths[sidx]] = parse['graphs'][pred_cursor, :sentence_lengths[sidx]]
					labels[pred_cursor, :sentence_lengths[sidx]] = parse['labels'][pred_cursor, :sentence_lengths[sidx]]
					pred_cursor += 1
		# retain all predictions if no errors occurred
		else:
			graphs = parse['graphs']
			labels = parse['labels']

		# store predicted dependency data
		for sidx, udidx in enumerate(range(start_idx, end_idx)):
			cur_sentence = ud_target[udidx]
			for widx, word in enumerate(cur_sentence.to_words(as_str=False)):
				# get head predictions
				head = int(graphs[sidx, widx]) + 1
				word.head = head

				# get label predictions (for non-probe models)
				if 'labels' in parse:
					label_idx = int(labels[sidx, widx])
					label = UD_RELATION_TYPES[label_idx] if label_idx > 0 else 'UNK'
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
