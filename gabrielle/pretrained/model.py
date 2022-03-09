import math
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from gabrielle.tokenizer import CharLevelTokenizer
from gabrielle.pretrained import BOATConfig


class FactorizedEmbeddingLayer(layers.Layer):
    def __init__(self, max_length, vocab_size, embedding_dim, factorized_dim=64, batch_first=False):
        """
        :param max_length: tokenizer.max_length
        :param vocab_size: tokenizer.vocab_size for input_dim of Embedding
        :param embedding_dim: final embedding output dim
        :param factorized_dim: intermediate dim for factorized embedding parameterization
        note that attention_mask will be computed from embedding.compute_mask(token_ids).
        """
        super(FactorizedEmbeddingLayer, self).__init__(name=self.__class__.__name__)
        self.max_length = max_length
        self.token_embedding = layers.Embedding(input_dim=vocab_size, output_dim=factorized_dim, mask_zero=True)
        self.factorized_embedding = layers.Dense(embedding_dim)
        self.segment_embedding = layers.Embedding(input_dim=2, output_dim=embedding_dim)
        self.positional_embedding = layers.Embedding(input_dim=max_length, output_dim=embedding_dim)
        self.batch_first = batch_first

    def call(self, inputs, *args, **kwargs):
        # inputs = (token_ids OR masked_input_ids, token_type_ids) : no need to feed attention_mask manually.
        # |inputs| = 2:(token_ids, token_type_ids) * batch_size * max_length
        if self.batch_first:
            token_ids = inputs[:, 0, :]
            token_type_ids = inputs[:, 1, :]
        else:
            token_ids = inputs[0]
            token_type_ids = inputs[1]
        # Factorized token embedding for token_ids
        token_vectors = self.token_embedding(token_ids)
        factorized_vectors = self.factorized_embedding(token_vectors)
        # get attention_mask from embedding.compute_mask(token_ids)
        attention_mask = self.token_embedding.compute_mask(token_ids)
        attention_mask = attention_mask[:, tf.newaxis, tf.newaxis, :]
        # Positional embedding
        positions = tf.range(start=0, limit=self.max_length, delta=1)
        position_vectors = self.positional_embedding(positions)
        # Segment embedding
        segment_vector = self.segment_embedding(token_type_ids)
        outputs = factorized_vectors + position_vectors + segment_vector
        return outputs, attention_mask


class FactorizedOrderedEmbeddingLayer(layers.Layer):
    def __init__(self, max_length, vocab_size, embedding_dim, factorized_dim=64, batch_first=False):
        """
        :param max_length: tokenizer.max_length
        :param vocab_size: tokenizer.vocab_size for input_dim of Embedding
        :param embedding_dim: final embedding output dim
        :param factorized_dim: intermediate dim for factorized embedding parameterization
        note that attention_mask will be computed from embedding.compute_mask(token_ids).
        """
        super(FactorizedOrderedEmbeddingLayer, self).__init__(name=self.__class__.__name__)
        self.max_length = max_length
        self.token_embedding = layers.Embedding(input_dim=vocab_size, output_dim=factorized_dim, mask_zero=True)
        self.factorized_embedding = layers.Dense(embedding_dim)
        self.segment_embedding = layers.Embedding(input_dim=2, output_dim=embedding_dim)
        self.word_order_embedding = layers.Embedding(input_dim=math.ceil(max_length/2)+1, output_dim=embedding_dim)
        self.char_order_embedding = layers.Embedding(input_dim=max_length+1, output_dim=embedding_dim)
        self.batch_first = batch_first

    def call(self, inputs, *args, **kwargs):
        """
        :param inputs: (token_ids OR masked_input_ids, token_type_ids, word_order_ids, char_order_ids)
                       - no need to feed attention_mask manually.
                       |inputs| = 2:(token_ids, token_type_ids) * batch_size * max_length
        :return: outputs, attention_mask
        """
        if self.batch_first:
            token_ids = inputs[:, 0, :]
            token_type_ids = inputs[:, 1, :]
            word_order_ids = inputs[:, 2, :]
            char_order_ids = inputs[:, 3, :]
        else:
            token_ids = inputs[0]
            token_type_ids = inputs[1]
            word_order_ids = inputs[2]
            char_order_ids = inputs[3]
        # Factorized token embedding for token_ids
        token_vectors = self.token_embedding(token_ids)
        factorized_vectors = self.factorized_embedding(token_vectors)
        # get attention_mask from embedding.compute_mask(token_ids)
        attention_mask = self.token_embedding.compute_mask(token_ids)
        attention_mask = attention_mask[:, tf.newaxis, tf.newaxis, :]
        # word-order & char-order embedding
        word_order_vectors = self.word_order_embedding(word_order_ids)
        char_order_vectors = self.char_order_embedding(char_order_ids)
        # Segment embedding
        segment_vector = self.segment_embedding(token_type_ids)
        outputs = factorized_vectors + word_order_vectors + char_order_vectors + segment_vector
        return outputs, attention_mask


class MaskedLanguageModelEmbeddingLayer(layers.Layer):
    def __init__(self, max_length, vocab_size, embedding_dim, batch_first=False):
        """
        :param max_length: tokenizer.max_length
        :param vocab_size: tokenizer.vocab_size for input_dim of Embedding
        :param embedding_dim: final embedding output dim
        note that attention_mask will be computed from embedding.compute_mask(token_ids).
        """
        super(MaskedLanguageModelEmbeddingLayer, self).__init__(name=self.__class__.__name__)
        self.max_length = max_length
        self.token_embedding = layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True)
        self.segment_embedding = layers.Embedding(input_dim=2, output_dim=embedding_dim)
        self.positional_embedding = layers.Embedding(input_dim=max_length, output_dim=embedding_dim)
        self.batch_first = batch_first

    def call(self, inputs, *args, **kwargs):
        # inputs = (token_ids OR masked_input_ids, token_type_ids) : no need to feed attention_mask manually.
        # |inputs| = 2:(token_ids, token_type_ids) * batch_size * max_length
        if self.batch_first:
            token_ids = inputs[:, 0, :]
            token_type_ids = inputs[:, 1, :]
        else:
            token_ids = inputs[0]
            token_type_ids = inputs[1]
        # token embedding for token_ids
        token_vectors = self.token_embedding(token_ids)
        # get attention_mask from embedding.compute_mask(token_ids)
        attention_mask = self.token_embedding.compute_mask(token_ids)
        attention_mask = attention_mask[:, tf.newaxis, tf.newaxis, :]
        # Positional embedding
        positions = tf.range(start=0, limit=self.max_length, delta=1)
        position_vectors = self.positional_embedding(positions)
        # Segment embedding
        segment_vector = self.segment_embedding(token_type_ids)
        outputs = token_vectors + position_vectors + segment_vector
        return outputs, attention_mask


class TransformerEncoderBlock(layers.Layer):
    def __init__(self, embedding_dim, num_heads, feedforward_dim, dropout_rate, block_id):
        """
        :param embedding_dim: final embedding output dim
        :param num_heads: number of heads for MultiHeadAttention()
        :param feedforward_dim: dense layer output dim
        :param dropout_rate: dropout_rate
        :param block_id: number id for each Transformer block
        """
        super(TransformerEncoderBlock, self).__init__(name=self.__class__.__name__ + str(block_id))
        self.block_id = block_id
        # Multi-head attention layer
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim // num_heads)
        self.layer_normalization1 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        # Feed-forward layer
        self.feedforward = keras.Sequential([layers.Dense(feedforward_dim, activation='relu'), layers.Dense(embedding_dim)])
        self.layer_normalization2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, inputs, *args, **kwargs):
        # |input[0]| = batch_size * max_length * embedding_dim
        # inputs = (token_vectors+position_vectors+segment_vector, attention_mask)
        query, value, key = inputs[0], inputs[0], inputs[0]
        attention_mask = inputs[1]
        attention_output = self.attention(query=query, value=value, key=key, attention_mask=attention_mask)
        attention_output = self.dropout1(attention_output)
        attention_output = self.layer_normalization1(query + attention_output)
        dense_output = self.feedforward(attention_output)
        dense_output = self.dropout2(dense_output)
        sequence_outputs = self.layer_normalization2(attention_output + dense_output)
        return sequence_outputs, attention_mask


class TransformerStackedEncoderLayers(layers.Layer):
    def __init__(self, num_layers, cross_layer_sharing, embedding_dim, num_heads, feedforward_dim, dropout_rate):
        """
        :param num_layers: height or size of stacked Transformer blocks
        :param cross_layer_sharing: If True, The only one block will be initiated and be reused N(=num_layers) times
        :param embedding_dim: final embedding output dim
        :param num_heads: number of heads for MultiHeadAttention()
        :param feedforward_dim: dense layer output dim
        :param dropout_rate: dropout_rate
        """
        super(TransformerStackedEncoderLayers, self).__init__(name=self.__class__.__name__)
        self.num_layers = num_layers
        self.cross_layer_sharing = cross_layer_sharing
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.feedforward_dim = feedforward_dim
        self.dropout_rate = dropout_rate
        self.layers = []
        if cross_layer_sharing:
            common_layer = TransformerEncoderBlock(embedding_dim=self.embedding_dim,
                                                   num_heads=self.num_heads,
                                                   feedforward_dim=self.feedforward_dim,
                                                   dropout_rate=self.dropout_rate,
                                                   block_id=0)
            self.layers = [common_layer] * self.num_layers
        else:
            for i in range(self.num_layers):
                self.layers.append(
                    TransformerEncoderBlock(embedding_dim=self.embedding_dim,
                                            num_heads=self.num_heads,
                                            feedforward_dim=self.feedforward_dim,
                                            dropout_rate=self.dropout_rate,
                                            block_id=i+1)
                )

    def call(self, inputs, *args, **kwargs):
        _outputs = inputs
        for layer in self.layers:
            _outputs = layer(_outputs)
        return _outputs


class TransformerPostEncoderDenseLayer(layers.Layer):
    def __init__(self, embedding_dim, vocab_size, return_argmax=False):
        """
        :param embedding_dim: embedding dim
        :param vocab_size: tokenizer.vocab_size for output_dim
        """
        super(TransformerPostEncoderDenseLayer, self).__init__(name=self.__class__.__name__)
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.classifier = keras.Sequential(
            [layers.Dense(embedding_dim, activation='relu'),
             layers.Dense(vocab_size, activation='softmax')]
        )
        self.return_argmax = return_argmax

    def call(self, inputs, *args, **kwargs):
        # |input[0]| = batch_size * max_length * embedding_dim
        _outputs = self.classifier(inputs[0])
        # no need to argmax on training masked language model.
        if self.return_argmax:
            return keras.backend.argmax(_outputs, axis=-1)
        return _outputs


if __name__ == '__main__':
    cfg = BOATConfig

    tokenizer = CharLevelTokenizer().load_pretrained_tokenizer('../tokenizer/your_awesome_tokenizer.json')
    batch_size = 5

    with open('../tokenizer/samples.txt', encoding='utf-8') as samples:
        texts = samples.read().splitlines()[:batch_size]

    encoded_tf = tokenizer.encode_for_transformer(texts, add_special_tokens=True, random_mask=True)

    # Test feedforward overview.
    # input shape is (2, batch_size, max_length): * If you want batch_size comes first, use batch_first=True on Embedder
    #     ==> tuple(input_ids(batch_size, max_length), token_type_ids(batch_size, max_length))
    # output shape is (batch_size, max_length, vocab_size):
    #     ==> (batch_size, max_length, vocab_size)

    model_inputs = tf.convert_to_tensor([encoded_tf['masked_input_ids'], encoded_tf['token_type_ids']])
    print(model_inputs, ': model_inputs')

    embedding_layer = FactorizedEmbeddingLayer(max_length=cfg.MAX_LENGTH,
                                               vocab_size=tokenizer.vocab_size,
                                               embedding_dim=cfg.EMBEDDING_DIM,
                                               factorized_dim=cfg.FACTORIZED_DIM,
                                               batch_first=False)
    embedding = embedding_layer(model_inputs)
    transformer_encoder_layer = TransformerStackedEncoderLayers(num_layers=cfg.NUM_LAYERS,
                                                                cross_layer_sharing=cfg.CROSS_LAYER_SHARING,
                                                                embedding_dim=cfg.EMBEDDING_DIM,
                                                                num_heads=cfg.NUM_HEADS,
                                                                feedforward_dim=cfg.FEEDFORWARD_DIM,
                                                                dropout_rate=cfg.DROPOUT_RATE)
    sequence_output = transformer_encoder_layer(embedding)
    post_encoder_layer = TransformerPostEncoderDenseLayer(embedding_dim=cfg.EMBEDDING_DIM,
                                                          vocab_size=tokenizer.vocab_size)

    model_outputs = post_encoder_layer(sequence_output)
    print(model_outputs, ': model_outputs')
