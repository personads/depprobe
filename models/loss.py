import torch
import torch.nn as nn

#
# Loss Functions
#


class RootedDependencyLoss(nn.Module):
	def __init__(self):
		super(RootedDependencyLoss, self).__init__()
		# set up spatial loss
		self._distance_loss = DependencyDistanceLoss()
		# set up label loss (ignore -1 padding labels)
		self._label_loss = torch.nn.CrossEntropyLoss(ignore_index=-1)
		# stats
		self.stats = {
			'loss_dist': None,
			'loss_label': None
		}

	def forward(self, parse, targets, use_trgt_graphs=False):
		pred_distances, pred_label_logits = parse['distances'], parse['label_logits']

		# calculate distance loss
		dist_loss = self._distance_loss(pred_distances, targets['distances'])
		self.stats['loss_dist'] = float(dist_loss.detach())

		# flatten logits and labels across all sequences
		pred_label_logits = torch.flatten(pred_label_logits, start_dim=0, end_dim=1) # (batch_size * max_len, num_labels)
		flat_trgt_labels = torch.flatten(targets['rels']) # (batch_size * max_len, )
		# calculate cross-entropy loss over label predictions
		label_loss = self._label_loss(pred_label_logits, flat_trgt_labels)
		self.stats['loss_label'] = float(label_loss.detach())

		loss = dist_loss + label_loss

		return loss


class StructuralProbingLoss(nn.Module):
	def __init__(self):
		super().__init__()
		# set up spatial loss
		self._distance_loss = DependencyDistanceLoss()
		# stats
		self.stats = {
			'loss_dist': None
		}

	def forward(self, parse, targets, use_trgt_graphs=False):
		pred_distances = parse['distances']

		# calculate distance loss
		loss = self._distance_loss(pred_distances, targets['distances'])
		self.stats['loss_dist'] = float(loss.detach())

		return loss


class DirectedProbingLoss(nn.Module):
	def __init__(self):
		super().__init__()
		# set up spatial loss
		self._depth_loss = DependencyDepthLoss()
		self._distance_loss = DependencyDistanceLoss()
		# stats
		self.stats = {
			'loss_depth': None,
			'loss_dist': None
		}

	def forward(self, parse, targets, use_trgt_graphs=False):
		pred_depths, pred_distances = parse['depths'], parse['distances']

		# calculate distance loss
		depth_loss = self._depth_loss(pred_depths, targets['depths'])
		dist_loss = self._distance_loss(pred_distances, targets['distances'])
		self.stats['loss_depth'] = float(depth_loss.detach())
		self.stats['loss_dist'] = float(dist_loss.detach())

		loss = depth_loss + dist_loss

		return loss

#
# L1-Distance Loss in Dependency Space
#
# 	Based on Hewitt & Manning, (2019) [1]
#
# 	[1] https://github.com/john-hewitt/structural-probes/blob/master/structural-probes/loss.py
#


class DependencyDepthLoss(nn.Module):
	def __init__(self):
		super(DependencyDepthLoss, self).__init__()

	def forward(self, pred_depths, trgt_depths):
		depth_mask = (trgt_depths != -1) # (batch_size, max_len)
		num_sentences = trgt_depths.shape[0] # scalar
		len_sentences = torch.sum(depth_mask, dim=-1) # (batch_size, )

		# calculate depth loss
		# sum absolute differences between predicted depth and target (both are positive)
		sum_token_depths = torch.sum(
			torch.abs((trgt_depths - pred_depths) * depth_mask),
			dim=-1
		) # (batch_size, )
		# average over individual sentence lengths
		nrm_sentence_depths = sum_token_depths / len_sentences # (batch_size, )
		# mean over batch
		depth_loss = torch.sum(nrm_sentence_depths) / num_sentences # scalar

		return depth_loss


class DependencyDistanceLoss(nn.Module):
	def __init__(self):
		super(DependencyDistanceLoss, self).__init__()

	def forward(self, pred_distances, trgt_distances):
		dists_mask = (trgt_distances != -1).float()  # (batch_size, max_len, max_len)
		num_sentences = float(trgt_distances.shape[0])  # scalar
		len_sentences = torch.sum((trgt_distances != -1)[:, :, 0], dim=-1).float()  # (batch_size, )

		# calculate distance loss
		# mask distances to disregard padding
		trgt_distances_masked = trgt_distances * dists_mask
		pred_distances_masked = pred_distances * dists_mask
		# sum absolute differences between predicted distances (positive because of square) and target (positive)
		sum_token_loss = torch.sum(
			torch.abs(trgt_distances_masked - pred_distances_masked),
			dim=[1, 2]
		)  # (batch_size, )
		# average over all token pairs per sentence
		nrm_sentence_loss = sum_token_loss / (len_sentences ** 2)  # (batch_size, )
		# mean over batch
		dist_loss = torch.sum(nrm_sentence_loss) / num_sentences  # scalar

		return dist_loss
