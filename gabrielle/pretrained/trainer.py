import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from gabrielle.pretrained import TextDataStreamer
from gabrielle.pretrained import BOATConfig
from gabrielle.pretrained import FactorizedEmbeddingLayer, FactorizedOrderedEmbeddingLayer, MaskedLanguageModelEmbeddingLayer
from gabrielle.pretrained import TransformerStackedEncoderLayers, TransformerPostEncoderDenseLayer
from gabrielle.tokenizer import CharLevelTokenizer


def get_model(config, name=None, plot=False):
    # Input layer =  batch_size * (token_ids, token_type_ids, **else) * max_length
    num_inputs_ = 4 if config.ORDER_ENCODING else 2
    inputs = keras.layers.Input(shape=(num_inputs_, config.MAX_LENGTH,), dtype=tf.int32, name='CharLevelInputLayer')

    # Embedding layer
    if config.FACTORIZED_DIM:
        if config.ORDER_ENCODING:
            embedding_layer = FactorizedOrderedEmbeddingLayer(max_length=config.MAX_LENGTH,
                                                              vocab_size=config.VOCAB_SIZE,
                                                              embedding_dim=config.EMBEDDING_DIM,
                                                              factorized_dim=config.FACTORIZED_DIM,
                                                              batch_first=True)
        else:
            embedding_layer = FactorizedEmbeddingLayer(max_length=config.MAX_LENGTH,
                                                       vocab_size=config.VOCAB_SIZE,
                                                       embedding_dim=config.EMBEDDING_DIM,
                                                       factorized_dim=config.FACTORIZED_DIM,
                                                       batch_first=True)
    else:
        embedding_layer = MaskedLanguageModelEmbeddingLayer(max_length=cfg.MAX_LENGTH,
                                                            vocab_size=config.VOCAB_SIZE,
                                                            embedding_dim=cfg.EMBEDDING_DIM,
                                                            batch_first=True)
    embedding = embedding_layer(inputs)

    # Stacked Transformer layer
    transformer_encoder_layer = TransformerStackedEncoderLayers(num_layers=config.NUM_LAYERS,
                                                                cross_layer_sharing=config.CROSS_LAYER_SHARING,
                                                                embedding_dim=config.EMBEDDING_DIM,
                                                                num_heads=config.NUM_HEADS,
                                                                feedforward_dim=config.FEEDFORWARD_DIM,
                                                                dropout_rate=config.DROPOUT_RATE)
    sequence_output = transformer_encoder_layer(embedding)

    # Post Encoder layer
    post_encoder_layer = TransformerPostEncoderDenseLayer(embedding_dim=config.EMBEDDING_DIM,
                                                          vocab_size=config.VOCAB_SIZE)
    outputs = post_encoder_layer(sequence_output)

    # Model
    model = keras.Model(inputs, outputs, name=name)

    loss = keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    optimizer = keras.optimizers.Adam(learning_rate=config.LEARNING_RATE, beta_1=0.9, beta_2=0.999, clipnorm=1.)
    model.compile(loss=loss, optimizer=optimizer, metrics=["acc"])

    if plot:
        keras.utils.plot_model(model, 'your_awesome_model.png', show_shapes=True)

    model.summary(line_length=150)

    for k, v in config.__dict__.items():
        if not k.startswith('__'):
            print(k, ':', v)

    return model


class StepHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs):
        self.loss = []
        self.acc = []

    def on_batch_end(self, batch, logs):
        if batch % 1000 == 0:
            self.loss.append(logs.get('loss'))
            self.acc.append(logs.get('acc'))

            # acc
            plt.subplot(2, 1, 1)
            plt.plot(self.acc)
            plt.title('Model accuracy')
            plt.ylabel('Accuracy')

            # loss
            plt.subplot(2, 1, 2)
            plt.plot(self.loss)
            plt.xlabel('Steps (1K)')
            plt.ylabel('Loss')

            plt.savefig('BOAT_learning_curve.png')


def train(model, train_data, validation_data, config):

    callbacks = [
        keras.callbacks.ModelCheckpoint(config.SAVED_MODEL_PATH, monitor='loss', mode='min', verbose=1,
                                        save_freq=config.SAVE_STEP_FREQ, save_best_only=True),
        keras.callbacks.TerminateOnNaN()
    ]

    history = model.fit(train_data, validation_data=validation_data, batch_size=cfg.BATCH_SIZE, epochs=cfg.EPOCHS,
                        callbacks=callbacks + [StepHistory()])


if __name__ == '__main__':
    # load pretrained tokenizer: dynamic masking with random_seed=None
    tokenizer = CharLevelTokenizer().load_pretrained_tokenizer('../tokenizer/your_awesome_tokenizer.json')

    # load config
    cfg = BOATConfig
    cfg.VOCAB_SIZE = tokenizer.vocab_size
    cfg.MAX_LENGTH = tokenizer.max_length
    num_inputs = 4 if cfg.ORDER_ENCODING else 2

    # dataset from generator: train
    train_generator = TextDataStreamer(corpus='D:/Corpora & Language Resources/모두의 말뭉치/splits/modu-paragraphs-train.txt',
                                       tokenizer=tokenizer,
                                       dynamic_strip=cfg.DYNAMIC_STRIP, order_encoding=cfg.ORDER_ENCODING)
    train_set = tf.data.Dataset.from_generator(train_generator,
                                               output_signature=(
                                                   tf.TensorSpec(shape=[num_inputs, cfg.MAX_LENGTH], dtype=tf.int32),
                                                   tf.TensorSpec(shape=[cfg.MAX_LENGTH], dtype=tf.int32)
                                               )).batch(cfg.BATCH_SIZE)
    # dataset from generator: valid
    valid_generator = TextDataStreamer(corpus='D:/Corpora & Language Resources/모두의 말뭉치/splits/modu-paragraphs-dev.txt',
                                       tokenizer=tokenizer,
                                       dynamic_strip=cfg.DYNAMIC_STRIP, order_encoding=cfg.ORDER_ENCODING)
    valid_set = tf.data.Dataset.from_generator(valid_generator,
                                               output_signature=(
                                                   tf.TensorSpec(shape=[num_inputs, cfg.MAX_LENGTH], dtype=tf.int32),
                                                   tf.TensorSpec(shape=[cfg.MAX_LENGTH], dtype=tf.int32)
                                               )).batch(cfg.BATCH_SIZE)

    # define model
    boat = get_model(config=cfg, name="BOAT-FACTORIZED-CROSS-ORDER", plot=True)

    # run trainer
    train(model=boat, train_data=train_set, validation_data=valid_set, config=cfg)

    loaded_model = keras.models.load_model(cfg.SAVED_MODEL_PATH)
    loaded_model.summary()

    print("\n테스트 결과:", loaded_model.evaluate(valid_set))
