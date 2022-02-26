import tensorflow as tf
from gabrielle.tokenizer import CharLevelTokenizer


class TextDataStreamer:
    def __init__(self, corpus, tokenizer, name=None):
        self.corpus = corpus
        self.tokenizer = tokenizer
        self.name = name if name else self.__class__.__name__
        self.file = open(self.corpus, encoding='utf-8')
        self.stream = self.file.__iter__()
        self.opened = True

    def __call__(self, *args, **kwargs):
        try:
            while True:
                text = self.stream.__next__().strip()
                tokenized = self.tokenizer.encode_for_transformer(text, add_special_tokens=True, random_mask=True)
                x_masked_token_ids = tokenized['masked_input_ids'][0]
                x_token_type_ids = tokenized['token_type_ids'][0]
                y_token_ids = tokenized['input_ids'][0]
                yield (x_masked_token_ids, x_token_type_ids), y_token_ids
        except StopIteration:
            self.file.close()
            self.file = open(self.corpus, encoding='utf-8')
            self.stream = self.file.__iter__()


if __name__ == '__main__':

    tokenizer = CharLevelTokenizer().load_pretrained_tokenizer('../tokenizer/your_awesome_tokenizer.json')

    text_generator = TextDataStreamer(corpus='../tokenizer/samples.txt', tokenizer=tokenizer)

    data = tf.data.Dataset.from_generator(text_generator,
                                          output_signature=(
                                              tf.TensorSpec(shape=(2, tokenizer.max_length,), dtype=tf.int32),
                                              tf.TensorSpec(shape=(tokenizer.max_length,), dtype=tf.int32)
                                          )).batch(5)

    cnt = 1
    for i, d in enumerate(data.as_numpy_iterator()):
        print('batch', i+1)
        for x, y in zip(d[0], d[1]):
            print('\t',
                  cnt,
                  ''.join(tokenizer.decode(x[0], strip_special_tokens=True)),
                  (tf.shape(x), tf.shape(y)),
                  ''.join(tokenizer.decode(y, strip_special_tokens=True)),)
            cnt += 1
