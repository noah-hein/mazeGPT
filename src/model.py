import math

import torch
import torch.nn as nn

def new_gelu(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))









# class Transformer(tf.keras.Model):
#     def __init__(self, input_vocab_size, output_vocab_size, hidden_size, num_layers, num_heads, max_sequence_length):
#         super(Transformer, self).__init__()
#         self.hidden_size = hidden_size
#         self.embedding = Embedding(input_vocab_size, hidden_size)
#         self.positional_encoding = self.create_positional_encoding(hidden_size, max_sequence_length)
#         self.encoder_layer = tf.keras.layers.TransformerEncoderLayer(hidden_size, num_heads)
#         self.encoder = tf.keras.layers.TransformerEncoder(self.encoder_layer, num_layers)
#         self.decoder = Dense(output_vocab_size)
#
#     def create_positional_encoding(self, hidden_size, max_sequence_length):
#         position = tf.range(max_sequence_length, dtype=tf.float32)[:, tf.newaxis]
#         div_term = tf.exp(tf.range(0, hidden_size, 2, dtype=tf.float32) * -(tf.math.log(10000.0) / hidden_size))
#         pe = tf.zeros([max_sequence_length, hidden_size], dtype=tf.float32)
#         pe[:, 0::2] = tf.sin(position * div_term)
#         pe[:, 1::2] = tf.cos(position * div_term)
#         return pe[tf.newaxis, ...]
#
#     def call(self, src, training=True):
#         embedded = self.embedding(src) * tf.math.sqrt(tf.cast(self.hidden_size, tf.float32))
#         src_with_pe = embedded + self.positional_encoding[:, :src.shape[1], :]
#         encoded = self.encoder(src_with_pe, training=training)
#         output = self.decoder(encoded)
#         return output

