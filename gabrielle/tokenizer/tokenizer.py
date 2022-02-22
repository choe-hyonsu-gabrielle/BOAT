import glob
import json
from collections import defaultdict
from tqdm import tqdm

SPECIAL_TOKENS_PRESET = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4, "[S]": 5, "[BGN]": 6, "[END]": 7}


class Tokenizer:
    def __init__(self, vocab_limit=30000, filters='', cased=True, max_length=512, shave_long_tail=True, name=None,
                 oov_token="[UNK]", whitespace_token="[S]", special_tokens_preset=SPECIAL_TOKENS_PRESET):
        assert special_tokens_preset is None or isinstance(special_tokens_preset, dict)
        self.name = name if name else self.__class__.__name__
        self.model = self.__class__.__name__
        self.vocab_size = None
        self.filters = filters  # filtering method is not implemented yet.
        self.cased = cased
        self.shave_long_tail = shave_long_tail
        self.documents = 0
        self.special_tokens = tuple(special_tokens_preset)
        assert oov_token in self.special_tokens or oov_token is None
        assert whitespace_token in self.special_tokens or whitespace_token is None
        self.oov_token = oov_token
        self.whitespace_token = whitespace_token
        self.special_token_strips = [t for t in self.special_tokens if t not in (self.oov_token, self.whitespace_token)]
        self.token_index = dict(special_tokens_preset)
        self.index_token = {v: k for k, v in self.token_index.items()}
        self.vocab_count = None
        # For training
        self.vocab_limit = vocab_limit
        # For truncation
        self.max_length = max_length

    def encode(self, inputs):
        raise NotImplementedError('Must be defined on specific tokenizer class.')
        pass

    def encode_for_transformer(self, inputs):
        raise NotImplementedError('Must be defined on specific tokenizer class.')
        pass

    def decode(self, inputs):
        raise NotImplementedError('Must be defined on specific tokenizer class.')
        pass

    def train_from_files(self, files):
        raise NotImplementedError('Must be defined on specific tokenizer class.')
        pass

    def save_tokenizer(self, save_to):
        _tokenizer = dict(name=self.name,
                          model=self.model,
                          vocab_size=self.vocab_size,
                          documents=self.documents,
                          filters=self.filters,
                          cased=self.cased,
                          shave_long_tail=self.shave_long_tail,
                          special_tokens=self.special_tokens,
                          oov_token=self.oov_token,
                          whitespace_token=self.whitespace_token,
                          special_token_strips=self.special_token_strips,
                          vocab_limit=self.vocab_limit,
                          max_length=self.max_length,
                          vocabulary=self.token_index,
                          vocab_count=self.vocab_count)
        with open(save_to, encoding='utf-8', mode='w') as exporter:
            json.dump(_tokenizer, exporter, ensure_ascii=False)

    def load_pretrained_tokenizer(self, load_from):
        with open(load_from, encoding='utf-8', mode='r') as importer:
            _t = json.load(importer)
        self.name = _t.get('name', None)
        self.model = _t.get('model', None)
        self.vocab_size = _t.get('vocab_size', None)
        self.documents = _t.get('documents', None)
        self.filters = _t.get('filters', None)
        self.cased = _t.get('cased', None)
        self.shave_long_tail = _t.get('shave_long_tail', None)
        self.special_tokens = _t.get('special_tokens', None)
        self.oov_token = _t.get('oov_token', None)
        self.whitespace_token = _t.get('whitespace_token', None)
        self.special_token_strips = [t for t in self.special_tokens if t not in (self.oov_token, self.whitespace_token)]
        self.vocab_limit = _t.get('vocab_limit', None)
        self.max_length = _t.get('max_length', None)
        self.token_index = _t.get('vocabulary', None)
        self.index_token = {v: k for k, v in self.token_index.items()}
        self.vocab_count = _t.get('vocab_count', None)
        assert self.oov_token in self.special_tokens or self.oov_token is None
        assert self.whitespace_token in self.special_tokens or self.whitespace_token is None
        return self


class CharLevelTokenizer(Tokenizer):
    def __init__(self, **kwargs):
        super(CharLevelTokenizer, self).__init__(**kwargs)

    def train_from_files(self, files, encoding='utf-8'):
        vocabs = defaultdict(int)
        print(f"[{self.__class__.__name__}] Training tokenizer from {len(files)} files...")
        for f in files:
            with open(f, encoding=encoding) as file:
                for line in tqdm(file.__iter__(), desc=f"[{self.__class__.__name__}] Processing \"{f}\""):
                    for token in [char for char in list(line.strip()) if char != ' ' and char not in self.filters]:
                        if not self.cased:
                            token = token.lower()
                        vocabs[token] += 1
                    self.documents += 1
        vocabs = sorted(vocabs.items(), key=lambda x: x[1], reverse=True)[:self.vocab_limit - len(self.special_tokens)]
        if self.shave_long_tail:
            vocabs = [v for v in vocabs if v[1] > 1]
        self.vocab_count = {k: v for k, v in vocabs}
        for index, (token, _) in enumerate(sorted(vocabs, key=lambda x: x[0]), start=len(self.special_tokens)):
            self.token_index[token] = index
            self.index_token[index] = token
        assert len(self.token_index) == len(self.index_token)
        self.vocab_size = len(self.token_index)

    def encode(self, inputs, add_special_tokens=True):
        if isinstance(inputs, str):
            truncation_idx = self.max_length - 2 if add_special_tokens and self.max_length else self.max_length
            inputs = inputs[:truncation_idx] if self.max_length else inputs
            return self._encode_text(inputs, add_special_tokens=add_special_tokens)
        elif isinstance(inputs, list) and isinstance(inputs[0], str):
            if self.max_length:
                truncation_idx = self.max_length - 2 if add_special_tokens else self.max_length
                inputs = [text[:truncation_idx] for text in inputs]
            batch_output = [self._encode_text(x, add_special_tokens=add_special_tokens) for x in inputs]
            return batch_output

    def encode_for_transformer(self, inputs, add_special_tokens=True, return_random_mask=False):
        input_ids = None            # token to index ids
        # token_type_ids            # segment ids inverts after '[SEP]'
        # attention_mask            # zeros on '[PAD]'
        # random_masked_input       # randomly generated '[MASK]' offset for masked language model
        if isinstance(inputs, str):
            input_ids = [self._encode_text(inputs, add_special_tokens=add_special_tokens)]
        elif isinstance(inputs, list) and isinstance(inputs[0], str):
            input_ids = [self._encode_text(x, add_special_tokens=add_special_tokens) for x in inputs]
        input_ids, token_type_ids, attention_mask = self._get_special_inputs(input_ids)
        if return_random_mask:
            random_masked_input = self._get_random_mask(rate=0.15, whole_word_mask=True)
        return dict(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

    def _get_special_inputs(self, inputs):
        pad = self.token_index.get('[PAD]')
        sep = self.token_index.get('[SEP]')
        padded_input_ids = []
        token_type_ids = []
        attention_masks = []
        for item in inputs:
            # for pad_to_max_length
            pad_size = (self.max_length - len(item)) if self.max_length else 0
            pads = [pad] * pad_size
            padded_ids = item + pads
            # for type_token_ids
            segment = 0
            segment_ids = []
            for idx in item:
                segment_ids.append(segment)
                if idx == sep:
                    segment = 0 if segment == 1 else 1
                elif idx == pad:
                    break
            if self.max_length:
                segment_ids = segment_ids + [segment] * pad_size
            # for attention_mask
            mask_ids = [1] * len(item) + [0] * pad_size
            if self.max_length:
                padded_ids = padded_ids[:self.max_length]
                segment_ids = segment_ids[:self.max_length]
                mask_ids = mask_ids[:self.max_length]
            padded_input_ids.append(padded_ids)
            token_type_ids.append(segment_ids)
            attention_masks.append(mask_ids)
        return padded_input_ids, token_type_ids, attention_masks

    def _get_random_mask(self, rate, whole_word_mask):
        return []

    def _encode_text(self, text, add_special_tokens=True):
        tokens = None
        text = text if self.cased else text.lower()
        if self.whitespace_token is not None and isinstance(self.whitespace_token, str):
            tokens = [char if char != ' ' else self.whitespace_token for char in list(text)]
        elif self.whitespace_token is None:
            tokens = [char for char in list(text) if char != ' ']
        if add_special_tokens:
            tokens = ['[CLS]'] + tokens + ['[SEP]']
        tokens_to_ids = [self.token_index.get(token, self.token_index[self.oov_token]) for token in tokens]
        return tokens_to_ids

    def decode(self, inputs, strip_special_tokens=False):
        assert isinstance(inputs, list)
        if isinstance(inputs[0], int):
            return self._decode_text(inputs, strip_special_tokens=strip_special_tokens)
        elif isinstance(inputs[0], list) and isinstance(inputs[0][0], int):
            batch_output = [self._decode_text(x, strip_special_tokens=strip_special_tokens) for x in inputs]
            return batch_output

    def _decode_text(self, indices, strip_special_tokens=False):
        ids_to_tokens = [self.index_token.get(index, self.oov_token) for index in indices]
        if strip_special_tokens:
            ids_to_tokens = [item for item in ids_to_tokens if item not in self.special_token_strips]
            ids_to_tokens = [item if item != self.whitespace_token else ' ' for item in ids_to_tokens]
        return ids_to_tokens


if __name__ == '__main__':
    # files = glob.glob('E:/Corpora & Language Resources/모두의 말뭉치/splits/*.txt')

    # tokenizer = CharLevelTokenizer()
    # tokenizer.train_from_files(files)
    # tokenizer.save_tokenizer('your_awesome_tokenizer.json')

    tokenizer = CharLevelTokenizer().load_pretrained_tokenizer('your_awesome_tokenizer.json')
    tokenizer.max_length = 50

    with open('samples.txt', encoding='utf-8') as samples:
        texts = samples.read().splitlines()

    encoded = tokenizer.encode(texts, add_special_tokens=True)
    encoded_tf = tokenizer.encode_for_transformer(texts, add_special_tokens=True)
    decoded = tokenizer.decode(encoded, strip_special_tokens=False)
    for _i, (x, e, d) in enumerate(zip(texts, encoded, decoded)):
        print(f'({_i})', x)
        print(len(e), e)
        print(len(d), d)
        print(''.join(d))
        print('input_ids:', len(encoded_tf['input_ids'][_i]), encoded_tf['input_ids'][_i])
        print('token_type_ids:', len(encoded_tf['token_type_ids'][_i]), encoded_tf['token_type_ids'][_i])
        print('attention_mask:', len(encoded_tf['attention_mask'][_i]),encoded_tf['attention_mask'][_i])
        print()
