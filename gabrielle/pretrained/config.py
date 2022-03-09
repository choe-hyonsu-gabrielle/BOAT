import dataclasses


@dataclasses.dataclass
class BOATConfig:
    ORDER_ENCODING = True
    DYNAMIC_STRIP = True
    MAX_LENGTH = 512
    VOCAB_SIZE = 30000
    EMBEDDING_DIM = 128
    FACTORIZED_DIM = 64
    CROSS_LAYER_SHARING = True
    NUM_HEADS = 4
    FEEDFORWARD_DIM = EMBEDDING_DIM
    NUM_LAYERS = 4
    DROPOUT_RATE = 0.1
    BATCH_SIZE = 16
    LEARNING_RATE = 0.00001
    EPOCHS = 1000
    SAVED_MODEL_PATH = 'saved_model'
