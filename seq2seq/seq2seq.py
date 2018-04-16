from __future__ import print_function

import os
import random
import sys

import numpy as np
import tensorflow as tf

from tensorflow.python.layers import core as layers_core

def build_encoder(num_layers, num_units, forget_bias, dropout):
    cell_list = _cell_list(num_layers, num_units, forget_bias, dropout)
    if len(cell_list) == 1:
        return cell_list[0]
    else:
        return tf.contrib.rnn.MultiRNNCell(cell_list)
    
def _cell_list(num_layers, num_units, forget_bias, dropout):
    cell_list = []
    for i in range(num_layers):
        single_cell = tf.contrib.rnn.BasicLSTMCell(num_units, forget_bias=forget_bias)
        if dropout > 0.0:
            single_cell = tf.contrib.rnn.DropoutWrapper(cell=single_cell, input_keep_prob=(1.0 - dropout))
        cell_list.append(single_cell)
    return cell_list

def build_decoder(encoder_outputs, encoder_state):
    tgt_sos_id = ;
    tgt_eos_id = ;

    # maximum_iteration: The maximum decoding steps
    maximum_iterations = ;

    ## Decoder
    with tf.variable_scope("decoder") as decoder_scope:
        cell, decoder_initial_state = self._build_decoder_cell();

def train():
    print("TRAIN")

    # Initializer
    initializer = tf.random_uniform_initializer(-0.1, 0.1, seed=None)
    tf.get_variable_scope().set_initializer(initializer)

    src_vocab_size = 1000;
    src_embed_size = 1000;
    tgt_vocab_size = 1000;
    tgt_embed_size = 1000;
    num_layers = 2
    num_units = 128
    forget_bias = 1
    dropout = 0.2

    # Embeddings
    with tf.variable_scope("embeddings", dtype=tf.float32, partitioner=None) as scope:
        with tf.variable_scope("encoder", partitioner=None):
            embeddings_encoder = tf.get_variable("embedding_encoder", [src_vocab_size, src_embed_size], tf.float32)
        with tf.variable_scope("decoder", partitioner=None):
            embeddings_decoder = tf.get_variable("embedding_decoder", [tgt_vocab_size, tgt_embed_size], tf.float32)

    batch_size = tf.size(128)

    # Projection
    with tf.variable_scope("build_netword"):
        with tf.variable_scope("decoder/output_projection"):
            output_layer = layers_core.Dense(tgt_vocab_size, use_bias=False, name="output_projection")

    ## Train graph
    with tf.variable_scope("dynamic_seq2seq", dtype=tf.float32):
        # Encoder
        encoder_outputs, encoder_state = build_encoder(num_layers, num_units, forget_bias, dropout)
        # Decoder
        # logits, sample_id, final_context_state = ;

if __name__ == "__main__":
    train()

### EOF
