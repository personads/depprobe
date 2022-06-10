import torch
import torch.nn as nn


#
# DepProbe
#


class DepProbe(nn.Module):
	def __init__(self, emb_model, dep_dim, dep_rels):
		super(DepProbe, self).__init__()
		# internal variables
		self._root_label = dep_rels.index('root')
		# internal models
		self._emb = emb_model
		self._arc = RootedGraphPredictor(self._emb.emb_dim, dep_dim)
		self._lbl = LabelClassifier(self._emb.emb_dim, len(dep_rels), self._root_label)

	def __repr__(self):
		return \
			f'{self.__class__.__name__}:\n' \
			f'  {self._emb}\n' \
			f'  <{self._arc.__class__.__name__}: {self._arc._emb_dim} -> {self._arc._out_dim}>\n' \
			f'  <{self._lbl.__class__.__name__}: {self._lbl._in_dim} -> {self._lbl._num_labels}>'

	def get_trainable_parameters(self):
		return list(self._arc.parameters()) + list(self._lbl.parameters())

	def train(self, mode=True):
		super(DepProbe, self).train(mode)
		self._emb.eval()
		return self

	def forward(self, sentences, decode=True):
		# embed sentences (batch_size, seq_length)
		# -> ([(batch_size, max_length, emb_dim) * 2], (batch_size, max_length))
		# -> ([emb_sentences_lay0, emb_sentences_lay1], att_sentences)
		with torch.no_grad():
			emb_layers, att_sentences = self._emb(sentences)

		# calculate distances in dependency space
		# dep_embeddings: (batch_size, dep_dim)
		# distances: (batch_size, max_len, max_len)
		dep_embeddings, distances = self._arc(emb_layers[0].detach())

		# classify dependency relations
		lbl_logits = self._lbl(emb_layers[1].detach(), att_sentences.detach())

		# construct minimal return set
		results = {
			'dependency_embeddings': dep_embeddings,
			'distances': distances,
			'label_logits': lbl_logits
		}

		# decode labelled dependency graph
		if decode:
			# get roots and labels from logits
			roots, labels = self._lbl.get_labels(lbl_logits.detach())
			# construct MST starting at root
			graphs = self._arc.to_graph(roots.detach(), distances.detach(), att_sentences.detach())

			# add labels and graphs to results
			results['graphs'] = graphs
			results['labels'] = labels

		return results


class DepProbeMix(DepProbe):
	def __init__(self, emb_model, dep_dim, dep_rels):
		super().__init__(emb_model=emb_model, dep_dim=dep_dim, dep_rels=dep_rels)
		# initialize mixture weights as standard average over layers
		self._mix_arc = nn.Parameter(torch.ones(len(self._emb._lm_layers)))
		self._mix_lbl = nn.Parameter(torch.ones(len(self._emb._lm_layers)))
		self._softmax = nn.Softmax(dim=0)

	def get_trainable_parameters(self):
		return list(self._arc.parameters()) + list(self._lbl.parameters()) + [self._mix_arc, self._mix_lbl]

	def forward(self, sentences, decode=True):
		# embed sentences (batch_size, seq_length)
		# -> ([(batch_size, max_length, emb_dim) * 2], (batch_size, max_length))
		# -> ([emb_sentences_lay0, emb_sentences_lay1], att_sentences)
		with torch.no_grad():
			emb_layers, att_sentences = self._emb(sentences)

		# compute weighted sum over embedding layers
		emb_sentences_arc = torch.zeros_like(emb_layers[0])
		emb_sentences_lbl = torch.zeros_like(emb_layers[0])
		for layer_idx in range(len(emb_layers)):
			emb_sentences_arc += self._softmax(self._mix_arc)[layer_idx] * emb_layers[layer_idx].detach()
			emb_sentences_lbl += self._softmax(self._mix_lbl)[layer_idx] * emb_layers[layer_idx].detach()

		# calculate distances in dependency space
		# dep_embeddings: (batch_size, dep_dim)
		# distances: (batch_size, max_len, max_len)
		dep_embeddings, distances = self._arc(emb_sentences_arc)

		# classify dependency relations
		lbl_logits = self._lbl(emb_sentences_lbl, att_sentences.detach())

		# construct minimal return set
		results = {
			'dependency_embeddings': dep_embeddings,
			'distances': distances,
			'label_logits': lbl_logits
		}

		# decode labelled dependency graph
		if decode:
			# get roots and labels from logits
			roots, labels = self._lbl.get_labels(lbl_logits.detach())
			# construct MST starting at root
			graphs = self._arc.to_graph(roots.detach(), distances.detach(), att_sentences.detach())

			# add labels and graphs to results
			results['graphs'] = graphs
			results['labels'] = labels

		return results


#
# Graph Predictors
#


class RootedGraphPredictor(nn.Module):
	def __init__(self, embedding_dim, output_dim):
		super(RootedGraphPredictor, self).__init__()
		self._emb_dim = embedding_dim
		self._out_dim = output_dim
		# trainable parameters
		self._transform = nn.Linear(self._emb_dim, self._out_dim, bias=False)

	def forward(self, emb_sentences):
		dep_embeddings = self._transform(emb_sentences)
		batch_size, max_len, out_dim = dep_embeddings.size()

		# calculate differences
		dup_transformed = dep_embeddings.unsqueeze(2)
		dup_transformed = dup_transformed.expand(-1, -1, max_len, -1)
		dup_transposed = dup_transformed.transpose(1, 2)
		differences = dup_transformed - dup_transposed  # (batch_size, max_len, max_len, dep_dim)
		squared_diffs = differences.pow(2)
		distances = torch.sum(squared_diffs, -1)

		return dep_embeddings, distances

	def to_graph(self, roots, distances, mask):
		graphs = torch.ones_like(mask, dtype=torch.long) * -2  # (batch_size, max_len)

		# iterate over sentences
		for sidx in range(graphs.shape[0]):
			# get current sentence length
			sen_len = int(torch.sum(mask[sidx]))

			# set root node's head to -1
			sen_root = int(roots[sidx].detach())
			graphs[sidx, sen_root] = -1

			# gather initial nodes
			tree_nodes = [sen_root]
			free_nodes = [n for n in range(sen_len) if n != sen_root]

			# while there are free nodes, keep adding to graph
			while free_nodes:
				# look for minimum distance between tree and free nodes
				cur_tree_dists = distances[sidx, tree_nodes, :]  # (num_tree_nodes, max_len)
				cur_dists = cur_tree_dists[:, free_nodes]  # (num_tree_nodes, num_free_nodes)
				min_dist_idx = torch.argmin(cur_dists)  # returns argmin of flattened distances # returns tree node, free node
				min_tree = tree_nodes[min_dist_idx // len(free_nodes)]  # tree node of minimum distance pair
				min_free = free_nodes[min_dist_idx % len(free_nodes)]  # free node of minimum distance pair

				# set head node of free node to tree node (point towards root)
				graphs[sidx, min_free] = min_tree

				# housekeeping
				tree_nodes.append(min_free)
				free_nodes.remove(min_free)

		return graphs


#
# Label Predictors
#


class LabelClassifier(nn.Module):
	def __init__(self, input_dim, num_labels, root_label):
		super(LabelClassifier, self).__init__()
		self._in_dim = input_dim
		self._num_labels = num_labels  # number of labels (e.g. 37)
		self._root_label = root_label  # index of root label
		# trainable parameters
		self._mlp = nn.Linear(self._in_dim, self._num_labels, bias=False)

	def forward(self, emb_sentences, att_sentences):
		# logits for all tokens in all sentences + padding -inf (batch_size, max_len, num_labels)
		logits = torch.ones(
			(att_sentences.shape[0], att_sentences.shape[1], self._num_labels),
			device=emb_sentences.device
		) * float('-inf')
		# get token embeddings of all sentences (total_tokens, emb_dim)
		emb_words = emb_sentences[att_sentences, :]
		# pass through MLP
		logits[att_sentences, :] = self._mlp(emb_words)  # (num_words, num_labels) -> (batch_size, max_len, num_labels)
		return logits

	def get_labels(self, lbl_logits):
		# gather word with highest root probability for each sentence
		roots = torch.argmax(lbl_logits[:, :, self._root_label], dim=-1)  # (batch_size, 1)
		# set root logits to -inf to prevent multiple roots
		lbl_logits_noroot = lbl_logits.detach().clone()
		lbl_logits_noroot[:, :, self._root_label] = torch.ones(
			(lbl_logits.shape[0], lbl_logits.shape[1]),
			device=lbl_logits.device
		) * float('-inf')
		# get predicted labels with maximum probability (padding should have -inf)
		labels = torch.argmax(lbl_logits_noroot, dim=-1)  # (batch_size, max_len)
		# add true root labels
		labels[torch.arange(lbl_logits.shape[0]), roots] = self._root_label
		# add -1 padding
		labels[(lbl_logits[:, :, 0] == float('-inf'))] = -1

		return roots, labels
