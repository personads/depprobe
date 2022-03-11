#!/usr/bin/python3

import argparse, os, sys

from collections import defaultdict

# local imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data.ud import *


def parse_arguments():
	arg_parser = argparse.ArgumentParser(description='Universal Dependencies - Undirected Unlabeled Attachment Score')
	arg_parser.add_argument('prediction', help='path to predicted CoNLL-U file')
	arg_parser.add_argument('target', help='path to target CoNLL-U file')
	arg_parser.add_argument("-st", "--subtypes", default=False, action="store_true", help="include relation subtypes in evaluation (default: False)")
	return arg_parser.parse_args()


def heads_to_edges(heads, rels, directed=True, labeled=True):
	edges = []

	for cidx, hidx in enumerate(heads):
		# skip root node for UUAS
		if (not directed) and (hidx == -1): continue
		# case: LAS
		if directed and labeled:
			edges.append((cidx, hidx, rels[cidx]))
		# case: UAS
		elif directed and not labeled:
			edges.append((cidx, hidx))
		# case: UUAS (sort edge nodes)
		elif not directed and not labeled:
			edges.append(tuple(sorted([cidx, hidx])))

	return set(edges)


def main():
	args = parse_arguments()

	# setup UD data
	ud_prediction = UniversalDependenciesTreebank.from_conllu(args.prediction, name=os.path.basename(args.prediction))
	ud_target = UniversalDependenciesTreebank.from_conllu(args.target, name=os.path.basename(args.target))
	assert len(ud_prediction) == len(ud_target), f"[Error] Unequal number of predictions and targets ({len(ud_prediction)} != {len(ud_target)})."

	metrics = [('las', 'words'), ('uas', 'words'), ('uuas', 'edges'), ('label', 'words'), ('root', 'roots')]
	matches = defaultdict(int)
	totals = defaultdict(int)
	for sidx in range(len(ud_target)):
		sen_pred = ud_prediction[sidx]
		sen_trgt = ud_target[sidx]

		trgt_graph, trgt_rels = sen_trgt.get_dependencies(include_subtypes=args.subtypes)
		pred_graph, pred_rels = sen_pred.get_dependencies(include_subtypes=args.subtypes)
		if len(trgt_graph) != len(pred_graph):
			print(f"[Error] Unequal number of nodes in predicted and target graph ({len(pred_graph)} != {len(trgt_graph)}). Skipped sentence {sidx}.")
			continue
		if len(trgt_rels) != len(pred_rels):
			print(f"[Error] Unequal number of labels in predictions and target ({len(pred_rels)} != {len(trgt_rels)}). Skipped sentence {sidx}.")
			continue

		sen_matches = defaultdict(int)
		# check LAS
		sen_matches['las'] += len(
			heads_to_edges(trgt_graph, trgt_rels, directed=True, labeled=True)
			&
			heads_to_edges(pred_graph, pred_rels, directed=True, labeled=True)
		)
		# check UAS
		sen_matches['uas'] += len(
			heads_to_edges(trgt_graph, trgt_rels, directed=True, labeled=False)
			&
			heads_to_edges(pred_graph, pred_rels, directed=True, labeled=False)
		)
		# check UUAS
		sen_matches['uuas'] += len(
			heads_to_edges(trgt_graph, trgt_rels, directed=False, labeled=False)
			&
			heads_to_edges(pred_graph, pred_rels, directed=False, labeled=False)
		)
		# check labels
		sen_matches['label'] += sum([1 for prel, trel in zip(pred_rels, trgt_rels) if prel == trel])
		# check roots
		sen_matches['root'] += sum([1 for prel, trel in zip(pred_rels, trgt_rels) if (prel == trel) and (trel == 'root')])

		# update totals
		totals['words'] += len(heads_to_edges(trgt_graph, trgt_rels, directed=True, labeled=True))
		totals['edges'] += len(heads_to_edges(trgt_graph, trgt_rels, directed=False, labeled=False))
		totals['roots'] += sum([1 for trel in trgt_rels if trel == 'root'])

		# aggregate instance-level scores
		for m, _ in metrics:
			matches[m] += sen_matches[m]

	# print global scores
	print('\t'.join([f'{m.upper():<5}' for m, _ in metrics]))
	print('\t'.join([f'{(100 * matches[m])/totals[t]:<5.2f}' for m, t in metrics]))


if __name__ == '__main__':
	main()
