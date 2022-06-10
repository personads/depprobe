import logging, os, re

from collections import OrderedDict

#
# Primary Universal Dependencies Data Classes
#


class UniversalDependencies:
	def __init__(self, treebanks=[]):
		self._treebanks = treebanks
		self._index_map = self._build_index_map() # corpus_index -> (treebank_index, sentence_index)

	def __repr__(self):
		return f'<UniversalDependencies: {len(self._treebanks)} treebanks, {len(self)} sentences>'

	def __len__(self):
		# returns total number of sentences across all treebanks
		return len(self._index_map)

	def __getitem__(self, key):
		if type(key) is slice:
			return [self._treebanks[tbidx][sidx] for tbidx, sidx in self._index_map[key]]
		elif type(key) is list:
			return [self._treebanks[self._index_map[key_idx][0]][self._index_map[key_idx][1]] for key_idx in key]
		else:
			tbidx, sidx = self._index_map[key]
			return self._treebanks[tbidx][sidx]

	def __setitem__(self, key, val):
		if type(key) is slice:
			for vidx, (tbidx, sidx) in enumerate(self._index_map[key]):
				self._treebanks[tbidx][sidx] = val[vidx]
		elif type(key) is list:
			for kidx, v in zip(key, val):
				tbidx, sidx = self._index_map[kidx]
				self._treebanks[tbidx][sidx] = v
		else:
			tbidx, sidx = self._index_map[key]
			self._treebanks[tbidx][sidx] = val

	def _build_index_map(self):
		index_map = []

		for tbidx, tb in enumerate(self._treebanks):
			tb_sentence_indices = list(range(len(tb))) # [0 ... num_sentences-1]
			tb_index_map = list(zip([tbidx for _ in range(len(tb))], tb_sentence_indices)) # [(tbidx, 0) ... (tbidx, num_sentences-1)]

			index_map += tb_index_map

		return index_map

	def _get_sentences_by_criterion(self, criterion):
		cur_key, cur_sentences = None, []
		for sidx in range(len(self)):
			# check if current key has changed based on criterion function
			if (criterion(sidx) != cur_key) or (sidx == len(self) - 1):
				if cur_key is not None:
					yield cur_key, (cur_sentences + [self[sidx]] if (sidx == len(self) - 1) else cur_sentences)
				# start gathering sentences of new key
				cur_key = criterion(sidx)
				cur_sentences = []
			cur_sentences.append(self[sidx])

	@staticmethod
	def from_directory(path, ud_filter=None, verbose=False):
		treebanks = []
		cursor = 0

		# gather treebank directories
		for tb_dir in sorted(os.listdir(path)):
			tb_path = os.path.join(path, tb_dir)

			# skip non-directories
			if not os.path.isdir(tb_path):
				continue

			# parse TB dirname
			tb_name_match = re.match(r'UD_(.+)-(.+)', tb_dir)
			if not tb_name_match:
				continue
			language = tb_name_match[1].replace('_', ' ')
			tb_name = tb_name_match[2]

			# initialize TB metadata
			tb_meta = {
				'Language': language,
				'Treebank': tb_name
			}

			# iterate over files in TB directory
			for tbf in sorted(os.listdir(tb_path)):
				tbf_path = os.path.join(tb_path, tbf)

				# if README
				if tbf.startswith('README'):
					# parse README metadata
					with open(tbf_path, 'r', encoding='utf8') as fp:
						readme = fp.read()
					metadata = re.search(r'[-=]+ Machine[-\s]readable metadata(.+)', readme, flags=re.DOTALL)
					if metadata is None: continue

					for meta_line in metadata[1].split('\n'):
						meta_line = meta_line.strip()
						# skip comments
						if meta_line.startswith('==='): continue
						# extract metadata from 'key: value'
						if len(meta_line.split(': ')) != 2: continue
						meta_key, meta_value = meta_line.split(': ')
						tb_meta[meta_key] = meta_value

				# skip non-conllu files
				if os.path.splitext(tbf)[1] != '.conllu': continue

				# load treebank
				treebank = UniversalDependenciesTreebank.from_conllu(tbf_path, name=tbf, meta=tb_meta, start_idx=cursor, ud_filter=ud_filter)
				treebanks.append(treebank)
				cursor += len(treebank)

				# print statistics (if verbose)
				if verbose:
					info = f"Loaded {treebank}."
					if logging.getLogger().hasHandlers():
						logging.info(info)
					else:
						print(info)

		return UniversalDependencies(treebanks=treebanks)

	def get_treebanks(self):
		return self._treebanks

	def get_domains(self):
		return sorted({d for tb in self.get_treebanks() for d in tb.get_domains()})

	def get_relations(self, include_subtypes=False):
		relations = set()
		for sidx in range(len(self)):
			sentence = self[sidx]
			if sentence is None: continue
			relations |= set(sentence.get_dependencies(include_subtypes=include_subtypes)[1])

		return sorted(relations)

	def get_language_of_index(self, key):
		return self._treebanks[self._index_map[key][0]].get_language()

	def get_treebank_name_of_index(self, key):
		return self._treebanks[self._index_map[key][0]].get_treebank_name()

	def get_treebank_file_of_index(self, key):
		return self._treebanks[self._index_map[key][0]].get_name()

	def get_domains_of_index(self, key):
		return self._treebanks[self._index_map[key][0]].get_domains()

	def get_sentences_by_language(self):
		for language, sentences in self._get_sentences_by_criterion(self.get_language_of_index):
			yield language, sentences

	def get_sentences_by_treebank(self):
		cursor = 0
		for treebank, sentences in self._get_sentences_by_criterion(self.get_treebank_name_of_index):
			yield f'{self.get_language_of_index(cursor)}-{treebank}', sentences
			cursor += len(sentences)

	def get_sentences_by_file(self):
		for tb_file, sentences in self._get_sentences_by_criterion(self.get_treebank_file_of_index):
			yield tb_file, sentences


class UniversalDependenciesTreebank:
	def __init__(self, sentences=[], name=None, meta={}):
		self._sentences = sentences
		self._name = name
		self._meta = meta

	def __repr__(self):
		return f'<UniversalDependenciesTreebank{f" ({self._name})" if self._name else ""}: {len(self._sentences)} sentences>'

	def __len__(self):
		return len(self._sentences)

	def __getitem__(self, key):
		return self._sentences[key]

	def __setitem__(self, key, val):
		self._sentences[key] = val

	@staticmethod
	def from_conllu(path, name=None, meta=None, start_idx=0, ud_filter=None):
		sentences = []
		with open(path, 'r', encoding='utf8') as fp:
			cur_lines = []
			for line_idx, line in enumerate(fp):
				line = line.strip()
				# on blank line, construct full sentence from preceding lines
				if line == '':
					try:
						# parse sentence from current set of lines
						sentence = UniversalDependenciesSentence.from_conllu(start_idx + len(sentences), cur_lines)
						# if filter is set, set any sentences not matching the filter to None
						if (ud_filter is not None) and (not ud_filter(sentence, meta)): sentence = None
						# append sentence to results
						sentences.append(sentence)
					except Exception as err:
						warn_msg = f"[Warning] UniversalDependenciesTreebank: Unable to parse '{path}' line {line_idx} ({err}). Skipping."
						if logging.getLogger().hasHandlers():
							logging.warning(warn_msg)
						else:
							print(warn_msg)
					cur_lines = []
					continue
				cur_lines.append(line)
		return UniversalDependenciesTreebank(sentences=sentences, name=name, meta=meta)

	def to_tokens(self):
		sentences = []
		for sentence in self:
			sentences.append(sentence.to_tokens())
		return sentences

	def to_words(self):
		sentences = []
		for sentence in self:
			sentences.append(sentence.to_words())
		return sentences

	def to_conllu(self, comments=True, resolve=False):
		return ''.join([s.to_conllu(comments, resolve) for s in self._sentences])

	def get_sentences(self):
		return self._sentences

	def get_name(self):
		return self._name

	def get_treebank_name(self):
		return self._meta.get('Treebank', 'Unknown')

	def get_language(self):
		return self._meta.get('Language', 'Unknown')

	def get_domains(self):
		return sorted(self._meta.get('Genre', '').split(' '))

	def get_statistics(self):
		statistics = {
			'sentences': len(self._sentences),
			'tokens': 0,
			'words': 0,
			'metadata': set()
		}

		for sidx, sentence in enumerate(self):
			statistics['tokens'] += len(sentence.to_tokens(as_str=False))
			statistics['words'] += len(sentence.to_words(as_str=False))
			statistics['metadata'] |= set(sentence.get_metadata().keys())

		statistics['metadata'] = list(sorted(statistics['metadata']))

		return statistics


class UniversalDependenciesSentence:
	def __init__(self, idx, tokens, comments=[]):
		self.idx = idx
		self._tokens = tokens
		self._comments = comments

	def __repr__(self):
		return f"<UniversalDependenciesSentence: ID {self.idx}, {len(self._tokens)} tokens, {len(self._comments)} comments>"

	@staticmethod
	def from_conllu(idx, lines):
		tokens, comments = [], []
		line_idx = 0
		while line_idx < len(lines):
			# check for comment
			if lines[line_idx].startswith('#'):
				comments.append(lines[line_idx])
				line_idx += 1
				continue

			# process tokens
			tkn_words = []
			tkn_line_split = lines[line_idx].split('\t')
			tkn_idx_str = tkn_line_split[0]
			# check for multiword token in 'a-b' format
			num_words = 1
			if '-' in tkn_idx_str:
				tkn_idx_split = tkn_idx_str.split('-')
				# convert token id to tuple signifying range (e.g. (3,4))
				tkn_span = (int(tkn_idx_split[0]), int(tkn_idx_split[1]))
				# collect the number of words in the current span
				while (line_idx + num_words + 1) < len(lines):
					num_words += 1
					# get current index as float due to spans such as '1-2; 1; 2; 2.1; ... 3' (e.g. Arabic data)
					span_str = lines[line_idx+num_words].split('\t')[0]
					if '-' in span_str: break
					span_tkn_idx = float(span_str)
					if int(span_tkn_idx) > tkn_span[1]: break
			# check for multiword token in decimal format '1; 1.1; 1.2; ... 2' or '0.1; 0.2; ... 1' (e.g. Czech data)
			elif re.match(r'^\d+\.\d+', tkn_idx_str)\
				or ((line_idx < (len(lines) - 1)) and re.match(r'^\d+\.\d+\t', lines[line_idx+1])):
				# count words that are part of multiword token
				while (line_idx + num_words) < len(lines):
					if not re.match(r'^\d+\.\d+\t', lines[line_idx+num_words]):
						break
					num_words += 1
				# token span for decimal indices is (a.1, a.n)
				tkn_span_start = float(tkn_idx_str) if re.match(r'^\d+\.\d+', tkn_idx_str) else int(tkn_idx_str) + .1
				tkn_span_end = tkn_span_start + (.1 * (num_words - 1))
				tkn_span = (tkn_span_start, tkn_span_end)
			# if single word token
			else:
				# convert token id to tuple with range 1 (e.g. (3,3))
				tkn_span = (int(tkn_idx_str), int(tkn_idx_str))

			# construct words contained in token
			for word_line in lines[line_idx:line_idx + num_words]:
				tkn_words.append(UniversalDependenciesWord.from_conllu(word_line))
			# construct and append token
			tokens.append(UniversalDependenciesToken(idx=tkn_span, words=tkn_words))
			# increment line index by number of words in token
			line_idx += num_words

		return UniversalDependenciesSentence(idx=idx, tokens=tokens, comments=comments)

	def to_text(self):
		return ''.join([t.to_text() for t in self._tokens])

	def to_tokens(self, as_str=True):
		return [(t.get_form() if as_str else t) for t in self._tokens]

	def to_words(self, as_str=True):
		return [(w.get_form() if as_str else w) for token in self._tokens for w in token.to_words()]

	def to_conllu(self, comments=True, resolve=False):
		conllu = '\n'.join(self._comments) + '\n' if comments and self._comments else ''

		conllu += '\n'.join([t.to_conllu(resolve=resolve) for t in self._tokens])
		conllu += '\n\n'

		return conllu

	def get_dependencies(self, offset=-1, include_subtypes=True):
		heads = [
			(w.head + offset)
			for token in self._tokens for w in token.to_words()
		]
		labels = [
			w.deprel if include_subtypes else w.deprel.split(':')[0]
			for token in self._tokens for w in token.to_words()
		]
		return heads, labels

	def get_pos(self):
		pos = [
			w.upostag for token in self._tokens for w in token.to_words()
		]
		return pos

	def get_comments(self, stripped=True):
		return [c[1:].strip() for c in self._comments]

	def get_metadata(self):
		"""Returns metadata from the comments of a sentence.

		Comment should follow the UD metadata guidelines '# FIELD = VALUE' or '# FIELD VALUE.
		Lines not following this convention are exported in the 'unknown' field.

		Returns a dict of metadata field and value pairs {'FIELD': 'VALUE'}.
		"""
		metadata = {}
		md_patterns = [r'^# ?(.+?) ?= ?(.+)', r'^# ?([^\s]+?)\s([^\s]+)$']
		for comment in self._comments:
			for md_pattern in md_patterns:
				md_match = re.match(md_pattern, comment)
				if md_match:
					metadata[md_match[1]] = md_match[2]
					break
			else:
				metadata['unknown'] = metadata.get('unknown', []) + [comment[1:].strip()]
		return metadata


class UniversalDependenciesToken:
	def __init__(self, idx, words):
		self.idx = idx # expects int or float tuple
		self._words = words # first element is token form, all following belong to potential multiword tokens

	def to_text(self):
		return self._words[0].to_text()

	def to_words(self):
		# if single word token
		if len(self._words) == 1:
			return self._words if self._words[0].head is not None else []
		# if multiword token
		else:
			# return words which have a dependency head
			return [w for w in self._words if w.head is not None]

	def to_conllu(self, resolve=False):
		# resolve multiword tokens into its constituents
		if resolve:
			# if form token has no head (e.g. 'i-j' token), get constituent words
			if (self._words[0].head is None) and (len(self._words) > 1):
				return '\n'.join([w.to_conllu() for w in self._words[1:] if w.head is not None])
			# if form token has head or it is not a multiword token, return itself
			elif self._words[0].head is not None:
				return self._words[0].to_conllu()
			# if token consists of only one word which has no head, omit (e.g. Coptic '0.1')
			else:
				return ''
		# otherwise return full set of words
		else:
			return '\n'.join([w.to_conllu() for w in self._words])

	def get_form(self):
		return self._words[0].get_form()


class UniversalDependenciesWord:
	"""
	ID: Word index, integer starting at 1 for each new sentence; may be a range for tokens with multiple words.
	FORM: Word form or punctuation symbol.
	LEMMA: Lemma or stem of word form.
	UPOSTAG: Universal part-of-speech tag drawn from our revised version of the Google universal POS tags.
	XPOSTAG: Language-specific part-of-speech tag; underscore if not available.
	FEATS: List of morphological features from the universal feature inventory or from a defined language-specific extension; underscore if not available.
	HEAD: Head of the current token, which is either a value of ID or zero (0).
	DEPREL: Universal Stanford dependency relation to the HEAD (root iff HEAD = 0) or a defined language-specific subtype of one.
	DEPS: List of secondary dependencies (head-deprel pairs).
	MISC: Any other annotation.

	[1] https://universaldependencies.org/docs/format.html
	"""
	def __init__(self, idx, form, lemma, upostag, xpostag, feats, head, deprel, deps=None, misc=None):
		self.idx = idx # expects int, float or str
		self.form = form
		self.lemma = lemma
		self.upostag = upostag
		self.xpostag = xpostag
		self.feats = feats # expects dict
		self.head = head # expects int
		self.deprel = deprel # expects str
		self.deps = deps
		self.misc = misc # expects dict

	def __repr__(self):
		return f'<UniversalDependenciesWord: ID {self.idx}, "{self.form}">'

	@staticmethod
	def from_conllu(line):
		# split line and initially convert '_' values to None
		idx_str, form, lemma, upostag, xpostag, feats, head, deprel, deps, misc = [(v if v != '_' else None) for v in line.split('\t')]
		# parse idx string (int 1, decimal 1.1 or string '1-2')
		idx = idx_str
		if re.match(r'^\d+\.\d+$', idx_str): idx = float(idx_str)
		elif re.match(r'^\d+$', idx_str): idx = int(idx_str)
		# parse form and lemma (special case '_')
		form = form if form is not None else '_'
		lemma = lemma if form != '_' else '_'
		# parse dependency head idx (int)
		head = int(head) if head is not None else head
		# parse FEATS dictionaries
		try:
			feats = {f.split('=')[0]:f.split('=')[1] for f in feats.split('|')}
		except:
			feats = {}
		# parse MISC dictionary
		try:
			misc = {m.split('=')[0]:m.split('=')[1] for m in misc.split('|')}
		except:
			misc = {}
		# construct word
		word = UniversalDependenciesWord(
			idx,
			form, lemma, # form and lemma are str
			upostag, xpostag, # upostag and xpostag are str
			feats,
			head, deprel, deps, # dependency information as str
			misc
		)
		return word

	def to_text(self):
		text = self.get_form() + ' ' # form + space by default
		# if 'SpaceAfter=No' remove trailing space
		if ('SpaceAfter' in self.misc) and (self.misc['SpaceAfter'] == 'No'):
			text = text[:-1]

		return text

	def to_conllu(self):
		conllu = ''

		# convert dictionaries
		feats_str = '|'.join([f'{k}={v}' for k, v in sorted(self.feats.items())]) if self.feats else None
		misc_str = '|'.join([f'{k}={v}' for k, v in sorted(self.misc.items())]) if self.misc else None

		conllu_values = [
			self.idx,
			self.form, self.lemma,
			self.upostag, self.xpostag,
			feats_str,
			self.head, self.deprel, self.deps,
			misc_str
		]
		# convert None to '_'
		conllu_values = [str(v) if v is not None else '_' for v in conllu_values]

		conllu = '\t'.join(conllu_values)
		return conllu

	def get_form(self, empty_as_unk=True):
		form = self.form if self.form else ''
		form = form.replace('\xad', '-') # sanitize soft hyphens
		form = form.replace('\x92', '') # sanitize single quotation mark
		form = form.replace('\x97', '') # sanitize acute accent below
		form = form.replace('\U000fe4fa', '') # sanitize Unicode character U+FE4FA
		form = form.replace('\ue402', '') # sanitize Unicode character U+E402
		form = form.replace('\ufeff', '') # sanitize Unicode character U+FEFF (zero width no-break space)
		form = form.replace('�', '') # sanitize Unicode replacement character
		# replace form with [UNK] if empty
		form = form if (len(form) > 0) or (not empty_as_unk) else '[UNK]'
		return form


#
# Universal Dependency Filtering Classes and Functions
#

class UniversalDependenciesFilter:
	def __repr__(self):
		return f"<{self.__class__.__name__}>"

	def __call__(self, sentence, meta):
		return True


class UniversalDependenciesFilterCombination(UniversalDependenciesFilter):
	"""Combines multiple filters."""
	def __init__(self, filters, mode='any'):
		self._filters = filters
		self._mode = mode

	def __repr__(self):
		return f"<{self.__class__.__name__}: match {self._mode} of {len(self._filters)} filters>"

	def __call__(self, sentence, meta):
		if (self._mode == 'all') and not all([filt(sentence, meta) for filt in self._filters]):
			return False
		if (self._mode == 'any') and not any([filt(sentence, meta) for filt in self._filters]):
			return False
		return True


class UniversalDependenciesIndexFilter(UniversalDependenciesFilter):
	"""Filters out any sentences which are not in the set of specified indices (corpus-level)."""
	def __init__(self, indices):
		self._indices = indices
		self._cursor = -1

	def __repr__(self):
		return f"<{self.__class__.__name__}: {len(self._indices)} indices>"

	def __call__(self, sentence=None, meta=None):
		self._cursor += 1
		if self._cursor not in self._indices:
			return False
		return True


class UniversalDependenciesMetadataFilter(UniversalDependenciesFilter):
	"""Filters out sentences based on a value in the treebank metadata."""
	def __init__(self, field, values, mode='include'):
		self._field = field
		self._values = values
		self._mode = mode

	def __repr__(self):
		return f"<{self.__class__.__name__}: {self._mode} {len(self._values)} '{self._field}' value(s)>"

	def __call__(self, sentence, meta):
		if (self._mode == 'include') and (meta[self._field] not in self._values):
			return False
		if (self._mode == 'exclude') and (meta[self._field] in self._values):
			return False
		return True


class UniversalDependenciesDomainFilter(UniversalDependenciesFilter):
	"""Filters out sentences based on the treebank domains or a provided domain distribution."""
	def __init__(self, domains, source=None, mode='include'):
		self._domains = set(domains)
		self._source = source # format {'domains': ['label0', ...], 'domain_dist': np.array}
		self._mode = mode
		self._cursor = -1

	def __repr__(self):
		return f"<{self.__class__.__name__}: {self._mode} {len(self._domains)} domains{'' if self._source is None else ' based on provided distribution'}>"

	def __call__(self, sentence, meta):
		self._cursor += 1
		# if no source is provided, use treebank metadata
		if self._source is None:
			domains = set(meta.get('Genre', '').split(' '))
		# if source is provided use maximum probability assigned domain
		else:
			domains = {self._source['domains'][self._source['domain_dist'][self._cursor].argmax()]}

		if (self._mode == 'include') and (len(domains & self._domains) < 1):
			return False
		if (self._mode == 'exclude') and (len(domains & self._domains) > 0):
			return False
		if (self._mode == 'exact') and (domains != self._domains):
			return False
		return True

#
# Universal Dependency Grouping Classes and Functions
#


class UniversalDependenciesGrouper:
	def __repr__(self):
		return f"<{self.__class__.__name__}>"

	def __call__(self, sentences):
		return OrderedDict()


class UniversalDependenciesCommentGrouper(UniversalDependenciesGrouper):
	"""Groups sentences by a comment pattern into numbered groups.

	Examples:
		* comment('newdoc'): Groups each sentence by the last preceeding 'newdoc' comment, otherwise files under ID = 'unknown'
	"""
	def __init__(self, comment_regex):
		self._comment_regex = comment_regex

	def __call__(self, sentences):
		"""Returns an OrderedDict of sentence lists grouped by the values of the provided metadata key."""
		groups = OrderedDict()
		# initialize current group as 'unknown' in case no key is encountered in the first lines
		num_groups = 0
		cur_group = 'unknown'
		for sentence in sentences:
			for comment in sentence.get_comments():
				# e.g. encountered 'newdoc'
				if re.match(self._comment_regex, comment):
					# set current group to 
					cur_group = f'group-{num_groups}'
					num_groups += 1
					break
			# if group is new, initialize empty list
			if cur_group not in groups:
				groups[cur_group] = []
			# add sentences to the active group
			groups[cur_group].append(sentence)
		return groups


class UniversalDependenciesMetadataGrouper(UniversalDependenciesGrouper):
	"""Groups sentences by a metadata key.

	Examples:
		* metadata('newdoc id'): Groups each sentence by the last preceeding ID specified in 'newdoc id = ID', otherwise files under ID = 'unknown'
		* metadata('source'): Groups each sentence by the same source, e.g. if each sentence is commented with the same 'source = SOURCE'
	"""
	def __init__(self, key, value_regex=''):
		self._key = key
		self._value_regex = value_regex if len(value_regex) > 0 else None

	def __call__(self, sentences):
		"""Returns an OrderedDict of sentence lists grouped by the values of the provided metadata key."""
		groups = OrderedDict()
		# initialize current group as 'unknown' in case no key is encountered in the first lines
		cur_group = 'unknown'
		for sentence in sentences:
			metadata = sentence.get_metadata()
			# e.g. encountered 'newdoc id'
			if self._key in metadata:
				# set current group to appropriate metadata value (e.g. document ID)
				cur_group = metadata[self._key]
				# check whether group must be extracted from the metadata value
				if self._value_regex:
					value_match = re.match(self._value_regex, cur_group)
					if value_match:
						cur_group = ''
						for match_key, match_val in sorted(value_match.groupdict().items()):
							if not match_key.startswith('val'):
								continue
							cur_group += match_val
					else:
						cur_group = 'unknown'
			# if group is new, initialize empty list
			if cur_group not in groups:
				groups[cur_group] = []
			# add sentences to the active group
			groups[cur_group].append(sentence)
		return groups


def parse_grouper(grouper_str):
	grouper_map = {
		'comment': UniversalDependenciesCommentGrouper,
		'metadata': UniversalDependenciesMetadataGrouper
	}

	# match the grouper syntax "grouper_name('arg1', 'arg2')"
	grouper_match = re.match(r'(.+?)\((.+)\)', grouper_str)
	if not grouper_match:
		return None

	# look for grouper in list of constructors
	grouper_key = grouper_match[1]
	if grouper_key not in grouper_map:
		return None

	# parse grouper arguments
	grouper_args = tuple()
	for arg_str in re.split(r",\s*(?=')", grouper_match[2]):
		grouper_args += (arg_str[1:-1], ) # remove surrounding quotes

	# instantiate grouper
	grouper = grouper_map[grouper_key](*grouper_args)

	return grouper

#
# Global Universal Dependencies Variables
#


# all 307 UD 2.8 dependency relations with subtypes
UD_RELATIONS = [
	'acl', 'acl:adv', 'acl:attr', 'acl:cleft', 'acl:fixed', 'acl:inf', 'acl:relat', 'acl:relcl',
	'advcl', 'advcl:abs', 'advcl:cau', 'advcl:cleft', 'advcl:cmpr', 'advcl:cond', 'advcl:coverb', 'advcl:eval', 'advcl:lcl', 'advcl:lto', 'advcl:mcl', 'advcl:pred', 'advcl:relcl', 'advcl:sp', 'advcl:svc', 'advcl:tcl',
	'advmod', 'advmod:arg', 'advmod:cau', 'advmod:comp', 'advmod:deg', 'advmod:det', 'advmod:df', 'advmod:emph', 'advmod:eval', 'advmod:fixed', 'advmod:foc', 'advmod:freq', 'advmod:lfrom', 'advmod:lmod', 'advmod:lmp', 'advmod:locy', 'advmod:lto', 'advmod:mmod', 'advmod:mode', 'advmod:neg', 'advmod:obl', 'advmod:que', 'advmod:tfrom', 'advmod:tlocy', 'advmod:tmod', 'advmod:to', 'advmod:tto',
	'amod', 'amod:att', 'amod:attlvc', 'amod:flat',
	'appos', 'appos:trans',
	'aux', 'aux:aff', 'aux:aspect', 'aux:caus', 'aux:clitic', 'aux:cnd', 'aux:ex', 'aux:imp', 'aux:nec', 'aux:neg', 'aux:opt', 'aux:part', 'aux:pass', 'aux:pot', 'aux:q', 'aux:tense',
	'case', 'case:acc', 'case:adv', 'case:aff', 'case:det', 'case:gen', 'case:loc', 'case:pred', 'case:voc',
	'cc', 'cc:nc', 'cc:preconj',
	'ccomp', 'ccomp:cleft', 'ccomp:obj', 'ccomp:obl', 'ccomp:pmod', 'ccomp:pred',
	'clf',
	'compound', 'compound:a', 'compound:affix', 'compound:dir', 'compound:ext', 'compound:lvc', 'compound:nn', 'compound:preverb', 'compound:prt', 'compound:quant', 'compound:redup', 'compound:smixut', 'compound:svc', 'compound:vo', 'compound:vv',
	'conj', 'conj:expl', 'conj:extend', 'conj:svc',
	'cop', 'cop:expl', 'cop:locat', 'cop:own',
	'csubj', 'csubj:cleft', 'csubj:cop', 'csubj:pass',
	'dep', 'dep:aff', 'dep:agr', 'dep:alt', 'dep:ana', 'dep:aux', 'dep:comp', 'dep:conj', 'dep:cop', 'dep:emo', 'dep:infl', 'dep:mark', 'dep:mod', 'dep:pos', 'dep:redup', 'dep:ss',
	'det', 'det:adj', 'det:noun', 'det:numgov', 'det:nummod', 'det:poss', 'det:predet', 'det:pron', 'det:rel',
	'discourse', 'discourse:emo', 'discourse:filler', 'discourse:intj', 'discourse:sp',
	'dislocated', 'dislocated:cleft', 'dislocated:csubj', 'dislocated:nsubj', 'dislocated:obj', 'dislocated:subj',
	'expl', 'expl:comp', 'expl:impers', 'expl:pass', 'expl:poss', 'expl:pv', 'expl:subj',
	'fixed',
	'flat', 'flat:abs', 'flat:dist', 'flat:foreign', 'flat:name', 'flat:num', 'flat:range', 'flat:repeat', 'flat:sibl', 'flat:title', 'flat:vv',
	'goeswith',
	'iobj', 'iobj:agent', 'iobj:appl', 'iobj:patient',
	'list',
	'mark', 'mark:adv', 'mark:advmod', 'mark:aff', 'mark:prt', 'mark:q', 'mark:rel',
	'nmod', 'nmod:agent', 'nmod:appos', 'nmod:arg', 'nmod:att', 'nmod:attlvc', 'nmod:attr', 'nmod:bahuv', 'nmod:cau', 'nmod:comp', 'nmod:flat', 'nmod:gen', 'nmod:gobj', 'nmod:gsubj', 'nmod:lfrom', 'nmod:lmod', 'nmod:npmod', 'nmod:obj', 'nmod:obl', 'nmod:part', 'nmod:poss', 'nmod:pred', 'nmod:prp', 'nmod:redup', 'nmod:relat', 'nmod:subj', 'nmod:tmod',
	'nsubj', 'nsubj:advmod', 'nsubj:aff', 'nsubj:bfoc', 'nsubj:caus', 'nsubj:cleft', 'nsubj:cop', 'nsubj:ifoc', 'nsubj:lfoc', 'nsubj:lvc', 'nsubj:nc', 'nsubj:obj', 'nsubj:pass', 'nsubj:periph',
	'nummod', 'nummod:det', 'nummod:entity', 'nummod:flat', 'nummod:gov',
	'obj', 'obj:advmod', 'obj:advneg', 'obj:agent', 'obj:appl', 'obj:caus', 'obj:lvc', 'obj:obl', 'obj:periph',
	'obl', 'obl:advmod', 'obl:agent', 'obl:appl', 'obl:arg', 'obl:cau', 'obl:cmp', 'obl:cmpr', 'obl:comp', 'obl:dat', 'obl:freq', 'obl:inst', 'obl:lfrom', 'obl:lmod', 'obl:lmp', 'obl:lto', 'obl:lvc', 'obl:mcl', 'obl:mod', 'obl:npmod', 'obl:orphan', 'obl:own', 'obl:patient', 'obl:pmod', 'obl:poss', 'obl:prep', 'obl:sentcon', 'obl:smod', 'obl:tmod',
	'orphan', 'orphan:missing',
	'parataxis', 'parataxis:appos', 'parataxis:conj', 'parataxis:coord', 'parataxis:deletion', 'parataxis:discourse', 'parataxis:dislocated', 'parataxis:hashtag', 'parataxis:insert', 'parataxis:mod', 'parataxis:newsent', 'parataxis:nsubj', 'parataxis:obj', 'parataxis:parenth', 'parataxis:rel', 'parataxis:rep', 'parataxis:restart', 'parataxis:rt', 'parataxis:sentence', 'parataxis:trans', 'parataxis:url',
	'punct',
	'reparandum',
	'root',
	'vocative', 'vocative:cl', 'vocative:mention',
	'xcomp', 'xcomp:cleft', 'xcomp:ds', 'xcomp:obj', 'xcomp:pred', 'xcomp:sp', 'xcomp:subj'
]

# UD dependency relations without subtypes (N=37)
UD_RELATION_TYPES = sorted({r.split(':')[0] for r in UD_RELATIONS})