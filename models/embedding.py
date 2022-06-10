import hashlib, pickle

import torch
import torch.nn as nn
import transformers

#
# Embedding Models
#


class EmbeddingModel(nn.Module):
	def __init__(self, lm_name, layers, cache=None):
		super(EmbeddingModel, self).__init__()
		# load transformer
		self._tok = transformers.AutoTokenizer.from_pretrained(lm_name, use_fast=True, add_prefix_space=True)
		self._lm = transformers.AutoModel.from_pretrained(lm_name, return_dict=True)
		# load cache
		self._cache = cache  # {hash: torch.tensor (num_layers, sen_len, emb_dim)}
		# internal variables
		self._lm_name = lm_name
		self._lm_layers = layers
		# public variables
		self.emb_dim = self._lm.config.hidden_size
		self.num_layers = self._lm.config.num_hidden_layers

	def __repr__(self):
		return f'<{self._lm.__class__.__name__}: "{self._lm_name}", Layers {str(self._lm_layers)}{" , with cache" if self._cache is not None else ""}>'

	def forward(self, sentences):
		# try retrieving embeddings from cache
		emb_cache = self.retrieve(sentences)
		if emb_cache is not None:
			emb_layers, att_words = emb_cache
		else:
			# compute embeddings if not in cache
			tok_sentences = self.tokenize(sentences)
			model_inputs = {
				k: tok_sentences[k] for k in ['input_ids', 'token_type_ids', 'attention_mask']
				if k in tok_sentences
			}

			# perform embedding forward pass
			model_outputs = self._lm(**model_inputs, output_hidden_states=True)
			hidden_states = model_outputs.hidden_states  # tuple(num_layers * (batch_size, max_len, hidden_dim))

			# post-process embeddings from specified layers
			emb_layers, att_words = [], None
			for layer_idx in self._lm_layers:
				emb_pieces = hidden_states[layer_idx] # batch_size, max_len, hidden_dim
				# reduce WordPiece to words
				emb_words, att_words = self.reduce(sentences, tok_sentences, emb_pieces)
				# append to results
				emb_layers.append(emb_words)

			# store embeddings in cache (if cache is enabled)
			if self._cache is not None:
				self.cache(sentences, emb_layers)

		# reduce list of layers to single tuple if only one layer is returned
		emb_layers = emb_layers if len(self._lm_layers) != 1 else emb_layers[0]

		return emb_layers, att_words

	def retrieve(self, sentences):
		if self._cache is None:
			return None

		max_len = max([len(s) for s in sentences])
		emb_layers = [torch.zeros((len(sentences), max_len, self.emb_dim)) for _ in range(len(self._lm_layers))]
		att_words = torch.zeros((len(sentences), max_len), dtype=torch.bool)

		# iterate over sentences
		for sidx, sentence in enumerate(sentences):
			# retrieve sentence embedding using string hash
			sen_hash = hashlib.md5(' '.join(sentence).encode('utf-8')).hexdigest()
			# skip batch if not all sentences are in cache
			if sen_hash not in self._cache:
				return None

			# retrieve embeddings for each layer from cache
			for lidx in range(len(self._lm_layers)):
				emb_layers[lidx][sidx, :len(sentence), :] = self._cache[sen_hash][lidx]  # (sen_len, emb_dim)
			att_words[sidx, :len(sentence)] = True

		# move input to GPU (if available)
		if torch.cuda.is_available():
			emb_layers = [embs.to(torch.device('cuda')) for embs in emb_layers]
			att_words = att_words.to(torch.device('cuda'))

		return emb_layers, att_words

	def cache(self, sentences, emb_layers):
		# detach, duplicate and move embeddings to CPU
		emb_layers = [embs.detach().clone().cpu() for embs in emb_layers]

		# iterate over sentences
		for sidx, sentence in enumerate(sentences):
			# compute sentence hash
			sen_hash = hashlib.md5(' '.join(sentence).encode('utf-8')).hexdigest()

			# initialize cache entry with list over layers
			self._cache[sen_hash] = []
			# iterate over layers
			for lidx in range(len(self._lm_layers)):
				self._cache[sen_hash].append(emb_layers[lidx][sidx, :len(sentence), :])  # (sen_len, emb_dim)

	def tokenize(self, sentences):
		# tokenize batch: {input_ids: [[]], token_type_ids: [[]], attention_mask: [[]], special_tokens_mask: [[]]}
		tok_sentences = self._tok(
			sentences,
			is_split_into_words=True, padding=True, truncation=True,
			return_tensors='pt', return_special_tokens_mask=True, return_offsets_mapping=True
		)
		# move input to GPU (if available)
		if torch.cuda.is_available():
			tok_sentences = {k: v.to(torch.device('cuda')) for k, v in tok_sentences.items()}

		return tok_sentences

	def reduce(self, sentences, tok_sentences, emb_pieces):
		emb_words = torch.zeros_like(emb_pieces)
		att_words = torch.zeros(emb_pieces.shape[:-1], dtype=torch.bool, device=emb_pieces.device)
		max_len = 0
		# iterate over sentences
		for sidx in range(emb_pieces.shape[0]):
			# get string tokens of current sentence
			tokens = self._tok.convert_ids_to_tokens(tok_sentences['input_ids'][sidx])
			offsets = tok_sentences['offset_mapping'][sidx]

			tidx = -1
			for widx, orig_word in enumerate(sentences[sidx]):
				# init aggregate word embedding
				emb_word = torch.zeros(emb_pieces.shape[-1], device=emb_pieces.device)  # (emb_dim,)
				num_tokens = 0
				coverage = 0
				while coverage < len(orig_word):
					tidx += 1
					if tidx >= len(emb_pieces[sidx, :]):
						raise TokenizationError(
							f"More words than pieces {tidx} >= {len(emb_pieces[sidx, :])}.\n"
							f"UD (len={len(sentences[sidx])}): {sentences[sidx]}\n"
							f"LM (len={len(tokens)}): {tokens}",
							position=(sidx, tidx)
						)
					# skip if special tokens ([CLS], [SEQ], [PAD])
					if tok_sentences['special_tokens_mask'][sidx, tidx] == 1: continue

					token_span = offsets[tidx]  # (start_idx, end_idx + 1) within orig_word
					# add WordPiece embedding to current word embedding sum
					emb_word += emb_pieces[sidx, tidx]
					num_tokens += 1
					coverage = token_span[1]

					# exit prematurely if next piece initiates new word (some LMs return less characters than in input)
					if (tidx < len(offsets) - 1) and offsets[tidx + 1][0] == 0:
						break

				# add mean of aggregate WordPiece embeddings and set attention to True
				emb_words[sidx, widx] = emb_word / num_tokens
				att_words[sidx, widx] = True

			# store new maximum sequence length
			max_len = len(sentences[sidx]) if len(sentences[sidx]) > max_len else max_len

		# reduce embedding and attention matrices to new maximum length
		emb_words = emb_words[:, :max_len, :]  # (batch_size, max_len, emb_dim)
		att_words = att_words[:, :max_len]  # (batch_size, max_len)

		return emb_words, att_words

#
# Custom Errors
#


class TokenizationError(Exception):
	def __init__(self, message, position=None):
		super().__init__(message)
		self.position = position
