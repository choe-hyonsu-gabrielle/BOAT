import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from gabrielle.pretrained import TextDataStreamer
from gabrielle.pretrained import BOATConfig
from gabrielle.pretrained import FactorizedEmbeddingLayer, MaskedLanguageModelEmbeddingLayer
from gabrielle.pretrained import TransformerStackedEncoderLayers, TransformerPostEncoderDenseLayer
from gabrielle.tokenizer import CharLevelTokenizer


def get_model(config, name=None, plot=False):
    # Input layer =  batch_size * (token_ids, token_type_ids) * max_length
    inputs = keras.layers.Input(shape=(2, config.MAX_LENGTH,), dtype=tf.int32, name='CharLevelInputLayer')

    # Embedding layer
    if config.FACTORIZED_DIM:
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
    model.summary(line_length=200)

    return model


def train(model, train_data, validation_data, config):

    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_acc', mode='max', verbose=1, patience=2),
        keras.callbacks.ModelCheckpoint(config.SAVEDMODEL_PATH, monitor='val_loss', mode='min', verbose=1, save_best_only=True),
        keras.callbacks.TerminateOnNaN()
    ]

    history = model.fit(train_data, validation_data=validation_data, batch_size=cfg.BATCH_SIZE, epochs=cfg.EPOCHS, callbacks=callbacks)

    # acc
    plt.subplot(2, 1, 1)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Valid'], loc='upper left')

    # loss
    plt.subplot(2, 1, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Valid'], loc='lower left')
    plt.savefig(model.name + '_learning_curve.png')


if __name__ == '__main__':
    # tokenizer
    tokenizer = CharLevelTokenizer().load_pretrained_tokenizer('../tokenizer/your_awesome_tokenizer.json')

    # config
    cfg = BOATConfig
    cfg.VOCAB_SIZE = tokenizer.vocab_size
    cfg.MAX_LENGTH = tokenizer.max_length

    # dataset
    train_generator = TextDataStreamer(corpus='E:/Corpora & Language Resources/모두의 말뭉치/splits/modu-plm-test.txt',
                                       tokenizer=tokenizer)
    train_set = tf.data.Dataset.from_generator(train_generator,
                                               output_signature=(
                                                   tf.TensorSpec(shape=[2, cfg.MAX_LENGTH], dtype=tf.int32),
                                                   tf.TensorSpec(shape=[cfg.MAX_LENGTH], dtype=tf.int32)
                                               )).batch(cfg.BATCH_SIZE)

    valid_generator = TextDataStreamer(corpus='../tokenizer/samples.txt',
                                       tokenizer=tokenizer)
    valid_set = tf.data.Dataset.from_generator(valid_generator,
                                               output_signature=(
                                                   tf.TensorSpec(shape=[2, cfg.MAX_LENGTH], dtype=tf.int32),
                                                   tf.TensorSpec(shape=[cfg.MAX_LENGTH], dtype=tf.int32)
                                               )).batch(cfg.BATCH_SIZE)

    # model
    boat = get_model(config=cfg, name="BOAT", plot=True)

    # trainer
    train(model=boat, train_data=train_set, validation_data=valid_set, config=cfg)
