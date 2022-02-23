import dataclasses
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from gabrielle.tokenizer import CharLevelTokenizer


@dataclasses.dataclass
class MLMConfig:
    MAX_LENGTH = 512
    EMBEDDING_DIM = 768
    FACTORIZED_DIM = 64
    CROSS_LAYER_SHARING = False
    NUM_HEADS = 8
    FEEDFORWARD_DIM = 768
    NUM_LAYERS = 8
    DROPOUT_RATE = 0.1
    BATCH_SIZE = 256
    LEARNING_RATE = 0.0001


class FactorizedTokSegPositEmbeddingLayer(layers.Layer):
    def __init__(self, max_length, vocab_size, embedding_dim, factorized_dim=64):
        super(FactorizedTokSegPositEmbeddingLayer, self).__init__(name=self.__class__.__name__)
        self.max_length = max_length
        self.token_embedding = layers.Embedding(input_dim=vocab_size, output_dim=factorized_dim)
        self.factorized_embedding = layers.Dense(embedding_dim)
        self.segment_embedding = layers.Embedding(input_dim=2, output_dim=embedding_dim)
        self.positional_embedding = layers.Embedding(input_dim=max_length, output_dim=embedding_dim)

    def call(self, inputs, *args, **kwargs):
        # inputs = (token_ids OR random_masked_input_ids, type_token_ids, attention_mask)
        # |input[i]| = batch_size * max_length
        token_ids = inputs[0]
        type_token_ids = inputs[1]
        attention_mask = inputs[2]
        # Factorized token embedding for token_ids
        token_vectors = self.token_embedding(token_ids, **kwargs)
        factorized_vectors = self.factorized_embedding(token_vectors, **kwargs)
        # Positional embedding
        positions = tf.range(start=0, limit=self.max_length, delta=1)
        position_vectors = self.positional_embedding(positions, **kwargs)
        # Segment embedding
        segment_vector = self.segment_embedding(type_token_ids, **kwargs)
        outputs = factorized_vectors + position_vectors + segment_vector
        print('embedding outputs', tf.shape(outputs))
        print('embedding attention_mask', tf.shape(attention_mask))
        return outputs, attention_mask


class TransformerEncoderBlock(layers.Layer):
    def __init__(self, embedding_dim, num_heads, feedforward_dim, dropout_rate, block_id):
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
        # inputs = (token_vectors+position_vectors+segment_vector, tf.convert_to_tensor(attention_mask))
        query, value, key = inputs[0], inputs[0], inputs[0]
        attention_mask = inputs[1]
        attention_output = self.attention(query=query, value=value, key=key, **kwargs)
        attention_output = self.dropout1(attention_output, **kwargs)
        attention_output = self.layer_normalization1(query + attention_output, **kwargs)
        dense_output = self.feedforward(attention_output, **kwargs)
        dense_output = self.dropout2(dense_output, **kwargs)
        sequence_output = self.layer_normalization2(attention_output + dense_output, **kwargs)
        return sequence_output, attention_mask


class TransformerStackedEncoderLayers(layers.Layer):
    def __init__(self, num_layers, cross_layer_sharing, embedding_dim, num_heads, feedforward_dim, dropout_rate):
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
            self.layers = [common_layer] * self.stack_height
        else:
            for i in range(self.stack_height):
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
    def __init__(self, embedding_dim, vocab_size):
        super(TransformerPostEncoderDenseLayer, self).__init__(name=self.__class__.__name__)
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.classifier = keras.Sequential(
            [layers.Dense(embedding_dim, activation='relu'),
             layers.Dense(vocab_size, activation='softmax')]
        )

    def call(self, inputs, *args, **kwargs):
        # |input[0]| = batch_size * max_length * embedding_dim
        outputs = self.classifier(inputs[0])
        return outputs


if __name__ == '__main__':
    cfg = MLMConfig()

    tokenizer = CharLevelTokenizer().load_pretrained_tokenizer('../tokenizer/your_awesome_tokenizer.json')

    with open('../tokenizer/samples.txt', encoding='utf-8') as samples:
        texts = samples.read().splitlines()[:5]

    encoded_tf = tokenizer.encode_for_transformer(texts, add_special_tokens=True, random_mask=True)
    # inputs = tf.convert_to_tensor([encoded_tf['input_ids'], encoded_tf['token_type_ids'], encoded_tf['attention_mask']])

    inputs = layers.Input(shape=(3, cfg.MAX_LENGTH,), dtype=tf.int64)
    embedding_layer = FactorizedTokSegPositEmbeddingLayer(max_length=cfg.MAX_LENGTH,
                                                          vocab_size=tokenizer.vocab_size,
                                                          embedding_dim=cfg.EMBEDDING_DIM,
                                                          factorized_dim=cfg.FACTORIZED_DIM)
    embedding = embedding_layer(inputs)
    """
    transformer_encoder_layer = TransformerStackedEncoderLayers(num_layers=cfg.NUM_LAYERS,
                                                                cross_layer_sharing=cfg.CROSS_LAYER_SHARING,
                                                                embedding_dim=cfg.EMBEDDING_DIM,
                                                                num_heads=cfg.NUM_HEADS,
                                                                feedforward_dim=cfg.FEEDFORWARD_DIM,
                                                                dropout_rate=cfg.DROPOUT_RATE)
    """

    transformer_encoder_layers = [TransformerEncoderBlock(embedding_dim=cfg.EMBEDDING_DIM,
                                                          num_heads=cfg.NUM_HEADS,
                                                          feedforward_dim=cfg.FEEDFORWARD_DIM,
                                                          dropout_rate=cfg.DROPOUT_RATE,
                                                          block_id=i+1) for i in range(cfg.NUM_LAYERS)]
    z = embedding
    for layer in transformer_encoder_layers:
        z = layer(z)
    sequence_output = z

    # sequence_output = transformer_encoder_layer(embedding)
    post_encoder_layer = TransformerPostEncoderDenseLayer(embedding_dim=cfg.EMBEDDING_DIM,
                                                          vocab_size=tokenizer.vocab_size)
    outputs = post_encoder_layer(sequence_output)

    model = keras.Model(inputs, outputs)

    optimizer = keras.optimizers.Adam(learning_rate=cfg.LEARNING_RATE)
    loss = keras.losses.SparseCategoricalCrossentropy()
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

    model.summary(line_length=200)
