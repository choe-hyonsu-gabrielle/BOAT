import json
from collections import defaultdict
from tqdm import tqdm

SPECIAL_TOKENS_PRESET = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4, "[S]": 5, "[BGN]": 6, "[END]": 7}


class Tokenizer:
    def __init__(self, vocab_limit=30000, filters='', cased=True, max_length=None, name=None, oov_token="[UNK]",
                 whitespace_token="[S]", special_tokens_preset=SPECIAL_TOKENS_PRESET):
        assert special_tokens_preset is None or isinstance(special_tokens_preset, dict)
        self.name = name if name else self.__class__.__name__
        self.model = self.__class__.__name__
        self.vocab_size = None
        self.filters = filters
        self.cased = cased
        self.documents = 0
        self.special_tokens = tuple(special_tokens_preset)
        assert oov_token in self.special_tokens or oov_token is None
        self.oov_token = oov_token
        assert whitespace_token in self.special_tokens or whitespace_token is None
        self.whitespace_token = whitespace_token
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
                          special_tokens=self.special_tokens,
                          oov_token=self.oov_token,
                          whitespace_token=self.whitespace_token,
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
        self.special_tokens = _t.get('special_tokens', None)
        self.oov_token = _t.get('oov_token', None)
        self.whitespace_token = _t.get('whitespace_token', None)
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
        self.vocab_count = {k: v for k, v in vocabs}
        for index, (token, _) in enumerate(sorted(vocabs, key=lambda x: x[0]), start=len(self.special_tokens)):
            self.token_index[token] = index
            self.index_token[index] = token
        assert len(self.token_index) == len(self.index_token)
        self.vocab_size = len(self.token_index)

    def encode(self, inputs):
        if isinstance(inputs, str):
            return self._encode_text(inputs)
        if isinstance(inputs, list):
            outputs = [self._encode_text(x) for x in inputs]
            return outputs

    def _encode_text(self, text):
        tokens = None
        text = text if self.cased else text.lower()
        if self.whitespace_token is not None and isinstance(self.whitespace_token, str):
            tokens = [char if char != ' ' else self.whitespace_token for char in list(text)]
        elif self.whitespace_token is None:
            tokens = [char for char in list(text) if char != ' ']
        tokens_to_ids = [self.token_index.get(token, self.token_index[self.oov_token]) for token in tokens]
        return tokens_to_ids

    def decode(self, inputs):
        assert isinstance(inputs, list)
        if isinstance(inputs[0], int):
            return self._decode_text(inputs)
        if isinstance(inputs, list):
            outputs = [self._decode_text(x) for x in inputs]
            return outputs

    def _decode_text(self, indices):
        ids_to_tokens = [self.index_token.get(index, self.oov_token) for index in indices]
        return ids_to_tokens


if __name__ == '__main__':
    t = CharLevelTokenizer()
    t.train_from_files(["train-samples.txt"])
    t.save_tokenizer('your_awesome_tokenizer.json')

    tokenizer = CharLevelTokenizer().load_pretrained_tokenizer('your_awesome_tokenizer.json')

    with open('inference-samples.txt', encoding='utf-8') as samples:
        texts = samples.read().splitlines()

    encoded = tokenizer.encode(texts)
    decoded = tokenizer.decode(encoded)
    for _i, (x, e, d) in enumerate(zip(texts, encoded, decoded)):
        print(f'({_i})', x)
        print(len(e), e)
        print(len(d), ''.join(d))
        print()
