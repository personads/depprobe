import numpy as np
import torch


class DepSpaceDataset:
	def __init__(self, ud, rels, idcs, batch_size):
		self._ud = ud
		self._rels = rels
		self._idcs = idcs
		self._batch_size = batch_size
		# iteration variables
		self._iter_idcs = None
		# cache variables
		self._cache_heads = {}
		self._cache_rels = {}
		self._cache_roots = {}
		self._cache_depths = {}
		self._cache_distances = {}

	def __len__(self):
		return len(self._idcs)

	def __iter__(self):
		self._iter_idcs = set(self._idcs)
		return self

	def __next__(self):
		if len(self._iter_idcs) > 0:
			# get random sample from UD
			batch_idcs = list(np.random.choice(list(self._iter_idcs), min(self._batch_size, len(self._iter_idcs)), replace=False))

			# gather sentences [['word', 'word', ...], ['word', 'word', ...]] (batch_size, var_lens)
			sentences = [s.to_words() for s in self._ud[batch_idcs]]
			max_len = max([len(s) for s in sentences])

			targets = self.get_targets(batch_idcs, max_len)

			self._iter_idcs -= set(batch_idcs)
			num_remaining = len(self._iter_idcs)

			return sentences, targets, num_remaining
		else:
			raise StopIteration

	def get_targets(self, idcs, max_len):
		targets = {
			'heads': torch.ones((len(idcs), max_len), dtype=torch.long) * -2,
			'rels': torch.ones((len(idcs), max_len), dtype=torch.long) * -1,
			'roots': torch.ones((len(idcs), max_len), dtype=torch.long) * -1,
			'depths': torch.ones((len(idcs), max_len)) * -1,
			'distances': torch.ones((len(idcs), max_len, max_len)) * -1
		}

		for bidx, sidx in enumerate(idcs):
			sen_len = len(self._ud[sidx].to_words())
			# compute heads and relations if not in cache
			if (sidx not in self._cache_heads) or (sidx not in self._cache_rels):
				heads, rels = self._ud[sidx].get_dependencies(include_subtypes=False)
				rels = [self._rels[r] for r in rels]  # map relation names to label indices
				self._cache_heads[sidx] = torch.tensor(heads, dtype=torch.long)
				self._cache_rels[sidx] = torch.tensor(rels, dtype=torch.long)
			targets['heads'][bidx, :sen_len] = self._cache_heads[sidx].clone()
			targets['rels'][bidx, :sen_len] = self._cache_rels[sidx].clone()

			# compute roots, depths and distances if not in cache
			if (sidx not in self._cache_roots) or (sidx not in self._cache_depths) or (sidx not in self._cache_distances):
				roots, depths, distances = self.get_dependency_structures(self._cache_heads[sidx])
				self._cache_roots[sidx] = roots
				self._cache_depths[sidx] = depths
				self._cache_distances[sidx] = distances
			targets['roots'][bidx, :sen_len] = self._cache_roots[sidx].clone()
			targets['depths'][bidx, :sen_len] = self._cache_depths[sidx].clone()
			targets['distances'][bidx, :sen_len, :sen_len] = self._cache_distances[sidx].clone()

		# move everything to GPU if available
		if torch.cuda.is_available():
			for tgt_key in ['heads', 'rels', 'roots', 'depths', 'distances']:
				targets[tgt_key] = targets[tgt_key].to(torch.device('cuda'))

		return targets

	def get_dependency_structures(self, heads):
		sen_len = heads.shape[0]
		# set root labels within sentence
		roots = (heads == -1).long()
		# init norms with padding value -1 (sen_len)
		depths = torch.ones_like(heads) * -1
		# init distances with padding values -1 (sen_len, sen_len)
		distances = torch.ones((sen_len, sen_len)) * -1

		# progress through tree
		cur_depth = 0  # start with depth 0
		cur_heads = [-1]  # start with root
		while cur_heads:
			cur_children = []
			# set distances between current heads, their children and the history
			for head in cur_heads:
				# gather current head's children
				children = [i for i in range(sen_len) if heads[i] == head]
				cur_children += children

				# skip further processing for root node
				if head < 0: continue

				# set distance to self to 0
				distances[head, head] = 0

				# set distance from current head to its children to 1
				distances[head, children] = 1
				distances[children, head] = 1

				# propagate existing distances to children
				for node in range(sen_len):
					# init indices to propagate to with all children
					prop_mask = list(children)
					# skip uninitialized distances
					if distances[head, node] < 1: continue
					# if node is child, do not propagate its own distance to itself
					if node in children:
						prop_mask.remove(node)
					# propagate current head's distance + 1 to all relevant child nodes
					distances[prop_mask, node] = distances[head, node] + 1
					distances[node, prop_mask] = distances[head, node] + 1

			# set depth of all child nodes at current depth
			depths[cur_children] = cur_depth

			cur_heads = cur_children
			cur_depth += 1

		return roots, depths, distances
