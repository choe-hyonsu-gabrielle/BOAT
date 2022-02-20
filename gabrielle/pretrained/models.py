import dataclasses
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers


@dataclasses.dataclass
class MLMConfig:
    MAX_LENGTH = 512
    BATCH_SIZE = 64
    LEARNING_RATE = 0.0001
    EMBEDDING_DIM = 128
    NUM_HEADS = 6
    FEEDFORWARD_DIM = 128
    NUM_LAYERS = 4


class TokenAndPositionalEmbedding(layers.Layer):
    def __init__(self, max_length, vocab_size, embedding_dim):
        super(TokenAndPositionalEmbedding, self).__init__()
        self.token_embedding = layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim)
        self.positional_embedding = layers.Embedding(input_dim=max_length, output_dim=embedding_dim)

    def call(self, inputs, *args, **kwargs):
        # |inputs| = batch_size * max_length
        positions = tf.range(start=0, limit=tf.shape(inputs)[-1], delta=1)
        position_vectors = self.positional_embedding(positions)
        token_vectors = self.token_embedding(inputs)
        return token_vectors + position_vectors


class TransformerEncoderBlock(layers.Layer):
    def __init__(self, embedding_dim, num_heads, feedforward_dim, dropout_rate, block_id):
        super(TransformerEncoderBlock, self).__init__(name=self.__class__.__name__ + str(block_id))
        self.block_id = block_id
        # Multi-head attention layer
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim / num_heads)
        self.layer_normalization1 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        # Feed-forward layer
        self.feedforward = keras.Sequential([layers.Dense(feedforward_dim, activation='relu'), layers.Dense(embedding_dim)])
        self.layer_normalization2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, inputs, *args, **kwargs):
        query, value, key = inputs, inputs, inputs
        attention_output = self.attention(query=query, value=value, key=key)
        attention_output = self.dropout1(attention_output)
        attention_output = self.layer_normalization1(query + attention_output)
        dense_output = self.feedforward(attention_output)
        dense_output = self.dropout2(dense_output)
        sequence_output = self.layer_normalization2(attention_output + dense_output)
        return sequence_output



