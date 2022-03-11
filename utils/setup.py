import logging, os, sys, time

import numpy as np
import torch
import transformers

# local imports
from data.ud import *
from data.utils import *

from models.embedding import *
from models.probing import *
from models.depprobe import *
from models.loss import *


def setup_output_directory(out_path):
	if os.path.exists(out_path):
		response = None
		while response not in ['y', 'n']:
			response = input(f"Path '{out_path}' already exists. Overwrite? [y/n] ")
		if response == 'n':
			exit(1)
	# if output dir does not exist, create it
	else:
		print(f"Path '{out_path}' does not exist. Creating...")
		os.mkdir(out_path)
	return True


def setup_logging(log_path):
	log_format = '%(message)s'
	log_level = logging.INFO
	logging.basicConfig(filename=log_path, filemode='w', format=log_format, level=log_level)

	logger = logging.getLogger()
	logger.addHandler(logging.StreamHandler(sys.stdout))


def setup_data(ud_path, split_path, skip_root_label=False):
	# load data split definition (if supplied)
	ud_filter = None
	splits = None
	if split_path:
		with open(split_path, 'rb') as fp:
			splits = pickle.load(fp)
		# create filter to load only relevant indices (train, dev)
		relevant_idcs = set(splits['train']) | set(splits['dev'])
		ud_filter = UniversalDependenciesIndexFilter(relevant_idcs)
		logging.info(f"Loaded data splits {', '.join([f'{s}: {len(idcs)}' for s, idcs in splits.items()])} with filter {ud_filter}.")

	# load Universal Dependencies
	ud = UniversalDependencies.from_directory(ud_path, ud_filter=ud_filter, verbose=True)
	# load UD dependency relations and map to indices
	rel_map = {r: i for i, r in enumerate(UD_RELATION_TYPES)}
	# set 'root' to -1 such that it's skipped (only for separate root prediction)
	if skip_root_label:
		rel_map['root'] = -1
	logging.info(f"Loaded {ud} with {len(rel_map)} dependency relations.")

	# use all of UD for each split if none are provided
	if splits is None:
		splits = {split: list(range(len(ud))) for split in ['train', 'dev', 'test']}

	return ud, splits, rel_map


def setup_model(lm_name, dep_dim, parser_type='depprobe', state_dict=None, emb_layers=None, emb_cache=None):
	# load pre-computed embedding cache if specified
	if emb_cache is not None:
		if emb_cache == 'local':
			emb_cache = {}
		else:
			# load pre-computed embeddings {hash: torch.Tensor (sen_len, emb_dim)}
			with open(emb_cache, 'rb') as fp:
				emb_cache = pickle.load(fp)

	# load transformer embedding model
	emb_model = EmbeddingModel(lm_name, layers=emb_layers, cache=emb_cache)

	# build structural probe
	if parser_type == 'structural':
		assert len(emb_layers) == 1, f"[Error] StructuralProbe requires one embedding layer, received {len(emb_layers)}."
		parser = StructuralProbe(
			emb_model=emb_model,
			dep_dim=dep_dim
		)
	# build directed probe
	elif parser_type == 'directed':
		assert len(emb_layers) == 1, f"[Error] DirectedProbe requires one embedding layer, received {len(emb_layers)}."
		parser = DirectedProbe(
			emb_model=emb_model,
			dep_dim=dep_dim
		)
	# build operational parser
	elif parser_type == 'depprobe':
		assert len(emb_layers) == 2, f"[Error] DepProbe requires two embedding layers, received {len(emb_layers)}."
		parser = DepProbe(
			emb_model=emb_model,
			dep_dim=dep_dim,
			dep_rels=UD_RELATION_TYPES
		)
	else:
		logging.error(f"[Error] Unknown model type '{parser_type}.")

	logging.info(f"Constructed '{parser_type}' model:")
	logging.info(parser)

	# load existing state if provided
	if state_dict is not None:
		parser.load_state_dict(state_dict)
		logging.info(f"Loaded weights from predefined state dict.")

	# check CUDA availability
	if torch.cuda.is_available():
		parser.to(torch.device('cuda'))
		logging.info(f"Moved parser to CUDA device ('{torch.device('cuda')}').")

	return parser


def setup_criterion(parser_type='depprobe'):
	# use structural distance loss
	if parser_type == 'structural':
		criterion = StructuralProbingLoss()
		logging.info(
			f"Using {criterion.__class__.__name__} with "
			f"{criterion._distance_loss.__class__.__name__}.")
	# use directed (depth + distance) loss
	elif parser_type == 'directed':
		criterion = DirectedProbingLoss()
		logging.info(
			f"Using {criterion.__class__.__name__} with "
			f"{criterion._depth_loss.__class__.__name__} and {criterion._distance_loss.__class__.__name__}.")
	# use depprobe loss
	else parser_type == 'depprobe':
		criterion = RootedDependencyLoss()
		logging.info(
			f"Using {criterion.__class__.__name__} with "
			f"{criterion._distance_loss.__class__.__name__} and "
			f"{criterion._label_loss.__class__.__name__}.")

	return criterion


def get_accuracies(parse, targets, match_all=True):
	accuracies = {}

	# compute graph accuracy (i.e. UAS)
	if 'graphs' in parse:
		# gather predictions and targets (these fields are always available)
		pred_graphs, trgt_graphs = parse['graphs'].detach(), targets['heads'].detach()

		# calculate mask to ignore padding (same for graphs and labels)
		mask = (trgt_graphs != -2).float()

		head_matches = (pred_graphs == trgt_graphs).float()
		num_head_matches = torch.sum(head_matches * mask, dim=-1)
		accuracies['graph'] = float(torch.sum(num_head_matches) / torch.sum(mask))

	# compute label accuracy
	if 'labels' in parse:
		pred_labels, trgt_labels = parse['labels'].detach(), targets['rels'].detach()
		# only count tokens with correct heads and correct labels
		if match_all:
			num_label_matches = torch.sum((pred_labels == trgt_labels) * mask * head_matches, dim=-1)
		# assume correct heads and count correct labels
		else:
			num_label_matches = torch.sum((pred_labels == trgt_labels) * mask, dim=-1)
		accuracies['label'] = float(torch.sum(num_label_matches) / torch.sum(mask))

	# compute sentence reconstruction accuracy
	if 'sentences' in parse:
		num_correct, num_total = 0, 0
		for sidx in range(len(parse['sentences'])):
			for widx in range(len(parse['sentences'][sidx])):
				if parse['sentences'][sidx][widx] == targets['sentences'][sidx][widx]:
					num_correct += 1
				num_total += 1
		accuracies['word'] = num_correct / num_total

	return accuracies


def run(parser, criterion, optimizer, dataset, mode='train', decode=True):
	stats = defaultdict(list)

	# set model to training mode
	if mode == 'train':
		parser.train()
	# set model to eval mode
	elif mode == 'eval':
		parser.eval()

	# iterate over batches
	for bidx, batch_data in enumerate(dataset):
		sentences, targets, num_remaining = batch_data

		try:
			# when training, perform both forward and backward pass
			if mode == 'train':
				stats['tokens'].append(sum([len(s) for s in sentences]))
				stats['time'].append(time.time())

				# zero out previous gradients
				optimizer.zero_grad()

				# forward pass (use teacher forcing for label prediction)
				parse = parser(sentences, decode=decode)

				# propagate loss
				loss = criterion(parse, targets, use_trgt_graphs=True)
				loss.backward()
				optimizer.step()

				stats['time'][-1] = time.time() - stats['time'][-1]

				# calculate accuracy (assume gold graph for labels)
				accuracies = get_accuracies(parse, targets, match_all=False)

			# when evaluating, perform forward pass without gradients
			elif mode == 'eval':
				with torch.no_grad():
					# forward pass
					parse = parser(sentences)
					# calculate loss
					loss = criterion(parse, targets)
				# calculate accuracies (both heads and labels need to match)
				accuracies = get_accuracies(parse, targets)

		except TokenizationError as tok_err:
			logging.error(f"[Error] {tok_err}. Skipped batch.")
			continue

		# store statistics
		for crit, val in criterion.stats.items():
			stats[crit].append(val)
		stats['loss'].append(float(loss.detach()))
		for acc_key, acc_val in accuracies.items():
			stats[f'acc_{acc_key}'].append(acc_val)
		stats['time'] = [np.sum(stats['time'])]
		stats['tokens'] = [np.sum(stats['tokens'])]

		# print batch statistics
		pct_complete = (1 - (num_remaining/len(dataset)))*100
		sys.stdout.write(
			f"\r[{mode.capitalize()} | Batch {bidx+1} | {pct_complete:.2f}%] "
			f"Acc: {' / '.join([f'{np.mean(v):.2f}' for s, v in sorted(stats.items()) if s.startswith('acc_')]) if decode else 'no-decode'}, "
			f"Loss: {' + '.join([f'{np.mean(v):.2f}' for s, v in sorted(stats.items()) if s.startswith('loss_')])} = "
			f"{np.mean(stats['loss']):.4f}"
		)
		sys.stdout.flush()

	# clear line
	print("\r", end='')

	return stats


def statistics(mode, stats, epoch_stats, ep_idx, epochs):
	# store epoch statistics
	for stat in epoch_stats:
		stats[f'{mode}/{stat}'].append(np.mean(epoch_stats[stat]))

	# print statistics
	logging.info(
		f"[Epoch {ep_idx+1}/{epochs}] {mode.capitalize()} completed with "
		f"AccGraph: {np.round(stats[f'{mode}/acc_graph'][-1], 4) if f'{mode}/acc_graph' in stats else 'None'}, "
		f"AccLabel: {np.round(stats[f'{mode}/acc_label'][-1], 4) if f'{mode}/acc_label' in stats else 'None'}, "
		f"AccWord: {np.round(stats[f'{mode}/acc_word'][-1], 4) if f'{mode}/acc_word' in stats else 'None'}, "
		f"DepthLoss: {np.round(stats[f'{mode}/loss_depth'][-1], 4) if f'{mode}/loss_depth' in stats else 'None'}, "
		f"DistLoss: {np.round(stats[f'{mode}/loss_dist'][-1], 4) if f'{mode}/loss_dist' in stats else 'None'}, "
		f"RootLoss: {np.round(stats[f'{mode}/loss_root'][-1], 4) if f'{mode}/loss_root' in stats else 'None'}, "
		f"LabelLoss: {np.round(stats[f'{mode}/loss_label'][-1], 4) if f'{mode}/loss_label' in stats else 'None'}, "
		f"Loss: {stats[f'{mode}/loss'][-1]:.4f}"
	)


def save_checkpoint(parser, optimizer, epoch, stats, path):
	torch.save({
		'epoch': epoch,
		'stats': stats,
		'parser_state': parser.state_dict(),
		'optimizer': optimizer.state_dict()
	}, path)
	logging.info(f"Saved checkpoint to '{path}'.")
