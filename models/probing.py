import numpy as np
import torch
import torch.nn as nn

#
# Parsing Models
#


class StructuralProbe(nn.Module):
	def __init__(self, emb_model, dep_dim):
		super().__init__()
		# internal models
		self._emb = emb_model
		self._arc = UndirectedGraphPredictor(self._emb.emb_dim, dep_dim)

	def __repr__(self):
		return \
			f'{self.__class__.__name__}:\n' \
			f'  {self._emb}\n' \
			f'  <{self._arc.__class__.__name__}: {self._arc._emb_dim} -> {self._arc._out_dim}>'

	def get_trainable_parameters(self):
		return list(self._arc.parameters())

	def train(self, mode=True):
		super().train(mode)
		self._emb.eval()
		return self

	def forward(self, sentences, decode=True):
		# embed sentences (batch_size, seq_length) -> (batch_size, max_length, emb_dim)
		with torch.no_grad():
			emb_sentences, att_sentences = self._emb(sentences)

		# calculate distances in dependency space
		# dep_embeddings: (batch_size, dep_dim)
		# distances: (batch_size, max_len, max_len)
		dep_embeddings, distances = self._arc(emb_sentences.detach())

		# construct minimal result set
		results = {
			'dependency_embeddings': dep_embeddings,
			'distances': distances
		}

		# decode undirected graph
		if decode:
			# construct MST starting at node 0 (no explicit directionality)
			edges = self._arc.to_edges(distances.detach(), att_sentences.detach())

			# add undirected graph to results
			results['graphs'] = edges

		return results


class DirectedProbe(nn.Module):
	def __init__(self, emb_model, dep_dim):
		super().__init__()
		# internal models
		self._emb = emb_model
		self._arc = DirectedGraphPredictor(self._emb.emb_dim, dep_dim)

	def __repr__(self):
		return \
			f'{self.__class__.__name__}:\n' \
			f'  {self._emb}\n' \
			f'  <{self._arc.__class__.__name__}: {self._arc._emb_dim} -> {self._arc._out_dim}>'

	def get_trainable_parameters(self):
		return list(self._arc.parameters())

	def train(self, mode=True):
		super().train(mode)
		self._emb.eval()
		return self

	def forward(self, sentences, decode=True):
		# embed sentences (batch_size, seq_length) -> (batch_size, max_length, emb_dim)
		with torch.no_grad():
			emb_sentences, att_sentences = self._emb(sentences)

		# calculate distances in dependency space
		# depths: (batch_size, max_len)
		# distances: (batch_size, max_len, max_len)
		depths, distances, _ = self._arc(emb_sentences)

		# construct minimal result set
		results = {
			'depths': depths,
			'distances': distances
		}

		# decode into directed graph using CLE
		if decode:
			# convert to graph with idx -> head (batch_size, max_len)
			graphs = self._arc.to_graph(depths, distances, att_sentences)

			# add directed graph to results
			results['graphs'] = graphs

		return results


#
# Graph Predictors
#


class UndirectedGraphPredictor(nn.Module):
	def __init__(self, embedding_dim, output_dim):
		super().__init__()
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
		differences = dup_transformed - dup_transposed # (batch_size, max_len, max_len, dep_dim)
		squared_diffs = differences.pow(2)
		distances = torch.sum(squared_diffs, -1)

		return dep_embeddings, distances

	def to_edges(self, distances, mask):
		graphs = torch.ones_like(mask, dtype=torch.long) * -2 # (batch_size, max_len)

		# iterate over sentences
		for sidx in range(graphs.shape[0]):
			# get current sentence length
			sen_len = int(torch.sum(mask[sidx]))

			# always set first node's head to -1 (root)
			graphs[sidx, 0] = -1

			# gather initial nodes
			tree_nodes = [0]
			free_nodes = [n for n in range(sen_len) if n != 0]

			# while there are free nodes, keep adding to graph
			while free_nodes:
				# look for minimum distance between tree and free nodes
				cur_tree_dists = distances[sidx, tree_nodes, :] # (num_tree_nodes, max_len)
				cur_dists = cur_tree_dists[:, free_nodes] # (num_tree_nodes, num_free_nodes)
				min_dist_idx = torch.argmin(cur_dists) # returns argmin of flattened distances # returns tree node, free node
				min_tree = tree_nodes[min_dist_idx // len(free_nodes)] # tree node of minimum distance pair
				min_free = free_nodes[min_dist_idx % len(free_nodes)] # free node of minimum distance pair

				# set head node of free node to tree node (point towards root)
				graphs[sidx, min_free] = min_tree

				# housekeeping
				tree_nodes.append(min_free)
				free_nodes.remove(min_free)

		return graphs


class DirectedGraphPredictor(nn.Module):
	"""
	Distance + Depth-based Graph Predictor

	Chu-Liu-Edmonds Algorithm implementation adapted from AllenNLP. [2]

	[2] https://github.com/allenai/allennlp/blob/v2.6.0/allennlp/nn/chu_liu_edmonds.py
	"""
	def __init__(self, embedding_dim, output_dim):
		super().__init__()
		self._emb_dim = embedding_dim
		self._out_dim = output_dim
		# trainable parameters
		self._depth_transform = nn.Linear(self._emb_dim, self._out_dim, bias=False)
		self._distance_transform = nn.Linear(self._emb_dim, self._out_dim, bias=False)

	def forward(self, emb_sentences):
		batch_size, max_len, emb_dim = emb_sentences.shape

		# depth prediction
		emb_depths = self._depth_transform(emb_sentences)
		# calculate norms
		norms = torch.bmm(
			emb_depths.view(batch_size * max_len, 1, self._out_dim),
			emb_depths.view(batch_size * max_len, self._out_dim, 1))
		norms = norms.view(batch_size, max_len)

		emb_distances = self._distance_transform(emb_sentences)
		# calculate squared differences
		dup_transformed = emb_distances.unsqueeze(2)
		dup_transformed = dup_transformed.expand(-1, -1, max_len, -1)
		dup_transposed = dup_transformed.transpose(1, 2)
		differences = dup_transformed - dup_transposed
		squared_diffs = differences.pow(2)
		distances = torch.sum(squared_diffs, -1)

		return norms, distances, differences

	def to_graph(self, depths, distances, mask):
		graphs = torch.ones_like(depths, dtype=torch.int) * -2 # (batch_size, max_len)

		# iterate over sentences
		for sidx in range(graphs.shape[0]):
			# get current sentence length
			sen_len = int(torch.sum(mask[sidx]))

			# initialize energy matrix
			energy = np.ones((sen_len+1, sen_len+1)) * float('-inf')

			# root node is shallowest
			root_idx = torch.argmin(depths[sidx, :sen_len])
			# set root node to maximum energy
			energy[0, root_idx+1] = 0

			# construct energy matrix
			for hidx in range(sen_len):
				for cidx in range(sen_len):
					# skip self
					if hidx == cidx: continue
					# skip if potential child is shallower than head
					if depths[sidx, cidx] < depths[sidx, hidx]: continue
					# if potential head is shallower than child, add score
					energy[hidx+1, cidx+1] = -distances[sidx, hidx, cidx]

			graph = self.decode_mst(energy, sen_len+1)
			graph = graph[1:] - 1  # remove dummy root node (idx=0) and offset by -1
			graphs[sidx, :sen_len] = torch.tensor(graph, dtype=torch.int)

		return graphs

	def decode_mst(self, energy, length):
		input_shape = energy.shape
		max_length = input_shape[-1]

		# Our energy matrix might have been batched -
		# here we clip it to contain only non padded tokens.
		energy = energy[:length, :length]
		label_id_matrix = None
		# get original score matrix
		original_score_matrix = energy
		# initialize score matrix to original score matrix
		score_matrix = np.array(original_score_matrix, copy=True)

		old_input = np.zeros([length, length], dtype=np.int32)
		old_output = np.zeros([length, length], dtype=np.int32)
		current_nodes = [True for _ in range(length)]
		representatives = []

		for node1 in range(length):
			original_score_matrix[node1, node1] = 0.0
			score_matrix[node1, node1] = 0.0
			representatives.append({node1})

			for node2 in range(node1 + 1, length):
				old_input[node1, node2] = node1
				old_output[node1, node2] = node2

				old_input[node2, node1] = node2
				old_output[node2, node1] = node1

		final_edges = {}

		# The main algorithm operates inplace.
		self.chu_liu_edmonds(
			length, score_matrix, current_nodes, final_edges, old_input, old_output, representatives
		)

		heads = np.zeros([max_length], np.int32)

		for child, parent in final_edges.items():
			heads[child] = parent

		return heads

	def chu_liu_edmonds(self, length, score_matrix,	current_nodes, final_edges,	old_input, old_output, representatives):
		# Set the initial graph to be the greedy best one.
		parents = [-1]
		for node1 in range(1, length):
			parents.append(0)
			if current_nodes[node1]:
				max_score = score_matrix[0, node1]
				for node2 in range(1, length):
					if node2 == node1 or not current_nodes[node2]:
						continue

					new_score = score_matrix[node2, node1]
					if new_score > max_score:
						max_score = new_score
						parents[node1] = node2

		# Check if this solution has a cycle.
		has_cycle, cycle = self._find_cycle(parents, length, current_nodes)
		# If there are no cycles, find all edges and return.
		if not has_cycle:
			final_edges[0] = -1
			for node in range(1, length):
				if not current_nodes[node]:
					continue

				parent = old_input[parents[node], node]
				child = old_output[parents[node], node]
				final_edges[child] = parent
			return

		# Otherwise, we have a cycle so we need to remove an edge.
		# From here until the recursive call is the contraction stage of the algorithm.
		cycle_weight = 0.0
		# Find the weight of the cycle.
		index = 0
		for node in cycle:
			index += 1
			cycle_weight += score_matrix[parents[node], node]

		# For each node in the graph, find the maximum weight incoming
		# and outgoing edge into the cycle.
		cycle_representative = cycle[0]
		for node in range(length):
			if not current_nodes[node] or node in cycle:
				continue

			in_edge_weight = float("-inf")
			in_edge = -1
			out_edge_weight = float("-inf")
			out_edge = -1

			for node_in_cycle in cycle:
				if score_matrix[node_in_cycle, node] > in_edge_weight:
					in_edge_weight = score_matrix[node_in_cycle, node]
					in_edge = node_in_cycle

				# Add the new edge score to the cycle weight
				# and subtract the edge we're considering removing.
				score = (
						cycle_weight
						+ score_matrix[node, node_in_cycle]
						- score_matrix[parents[node_in_cycle], node_in_cycle]
				)

				if score > out_edge_weight:
					out_edge_weight = score
					out_edge = node_in_cycle

			score_matrix[cycle_representative, node] = in_edge_weight
			old_input[cycle_representative, node] = old_input[in_edge, node]
			old_output[cycle_representative, node] = old_output[in_edge, node]

			score_matrix[node, cycle_representative] = out_edge_weight
			old_output[node, cycle_representative] = old_output[node, out_edge]
			old_input[node, cycle_representative] = old_input[node, out_edge]

		# For the next recursive iteration, we want to consider the cycle as a
		# single node. Here we collapse the cycle into the first node in the
		# cycle (first node is arbitrary), set all the other nodes not be
		# considered in the next iteration. We also keep track of which
		# representatives we are considering this iteration because we need
		# them below to check if we're done.
		considered_representatives = []
		for i, node_in_cycle in enumerate(cycle):
			considered_representatives.append(set())
			if i > 0:
				# We need to consider at least one
				# node in the cycle, arbitrarily choose
				# the first.
				current_nodes[node_in_cycle] = False

			for node in representatives[node_in_cycle]:
				considered_representatives[i].add(node)
				if i > 0:
					representatives[cycle_representative].add(node)

		self.chu_liu_edmonds(
			length, score_matrix, current_nodes, final_edges, old_input, old_output, representatives
		)

		# Expansion stage.
		# check each node in cycle, if one of its representatives
		# is a key in the final_edges, it is the one we need.
		found = False
		key_node = -1
		for i, node in enumerate(cycle):
			for cycle_rep in considered_representatives[i]:
				if cycle_rep in final_edges:
					key_node = node
					found = True
					break
			if found:
				break

		previous = parents[key_node]
		while previous != key_node:
			child = old_output[parents[previous], previous]
			parent = old_input[parents[previous], previous]
			final_edges[child] = parent
			previous = parents[previous]

	def _find_cycle(self, parents, length, current_nodes):
		added = [False for _ in range(length)]
		added[0] = True
		cycle = set()
		has_cycle = False
		for i in range(1, length):
			if has_cycle:
				break
			# don't redo nodes we've already
			# visited or aren't considering.
			if added[i] or not current_nodes[i]:
				continue
			# Initialize a new possible cycle.
			this_cycle = set()
			this_cycle.add(i)
			added[i] = True
			has_cycle = True
			next_node = i
			while parents[next_node] not in this_cycle:
				next_node = parents[next_node]
				# If we see a node we've already processed,
				# we can stop, because the node we are
				# processing would have been in that cycle.
				if added[next_node]:
					has_cycle = False
					break
				added[next_node] = True
				this_cycle.add(next_node)

			if has_cycle:
				original = next_node
				cycle.add(original)
				next_node = parents[original]
				while next_node != original:
					cycle.add(next_node)
					next_node = parents[next_node]
				break

		return has_cycle, list(cycle)
