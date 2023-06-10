import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def create_transformer_model(input_vocab_size, target_vocab_size, num_layers, d_model, num_heads, dff, dropout_rate):
    # Input layers
    input = layers.Input(shape=(None,))
    target = layers.Input(shape=(None,))

    # Embedding layers
    input_embedding = layers.Embedding(input_vocab_size, d_model)(input)
    target_embedding = layers.Embedding(target_vocab_size, d_model)(target)

    # Encoder layers
    encoder_output = input_embedding
    for _ in range(num_layers):
        encoder_output = layers.MultiHeadAttention(num_heads, d_model)([encoder_output, encoder_output])
        encoder_output = layers.Dropout(dropout_rate)(encoder_output)
        encoder_output = layers.LayerNormalization(epsilon=1e-6)(encoder_output + input_embedding)

    # Decoder layers
    decoder_output = target_embedding
    for _ in range(num_layers):
        decoder_output = layers.MultiHeadAttention(num_heads, d_model)([decoder_output, decoder_output])
        decoder_output = layers.Dropout(dropout_rate)(decoder_output)
        decoder_output = layers.LayerNormalization(epsilon=1e-6)(decoder_output + target_embedding)

        decoder_output = layers.MultiHeadAttention(num_heads, d_model)([decoder_output, encoder_output])
        decoder_output = layers.Dropout(dropout_rate)(decoder_output)
        decoder_output = layers.LayerNormalization(epsilon=1e-6)(decoder_output + encoder_output)

    # Output layer
    output = layers.Dense(target_vocab_size, activation='softmax')(decoder_output)

    # Model
    model = keras.Model(inputs=[input, target], outputs=output)
    return model
