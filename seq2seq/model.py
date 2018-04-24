# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals, absolute_import
import sys
import os
import codecs
import argparse
import numpy as np
import tensorflow as tf

class Model:
    def __init__(self):
        self.vocab_size = 0
        self.source_sequence_length = None
        self.src_embed_size = 100
        self.tgt_embed_size = 100
        self.max_time = 5
        self.batch_size = 128
        self.num_units = 128
        self.num_layers = 2
        self.forget_bias = 0
        self.dropout = 0.2
        print("INIT")

    def train(self):

        ### Embedding
        embed_encoder = tf.get_variable("embed_encoder", [self.vocab_size, self.src_embed_size], tf.float32)
        embed_decoder = tf.get_variable("embed_decoder", [self.vocab_size, self.tgt_embed_size], tf.float32)

        ### Projection
        with tf.variable_scope("build_netword"):
            with tf.variable_scope("decoder/output_projection"):
                self.output_layer = layers_core.Dense(self.vocab_size, use_bias=False, name="output_projection")

        with tf.variable_scope("dynamic_seq2seq", dtype=tf.float32):
            ### Encoder
            with tf.variable_scope("encoder", dtype=tf.float32) as scope:
                encoder_emb_imp = tf.nn.embedding_lookup(embed_encoder, [max_time, batch_size, self.num_units])

                cell_list = []
                for i range(self.num_layers):
                    cell = tf.contrib.rnn.BasicLSTMCell(self.num_units, forget_bias=self.forget_bias)
                    if self.dropout > 0.0:
                        cell = tf.contrib.rnn.DropoutWrapper(cell=cell, input_keep_prob=(1.0 - dropout))
                    cell_list.append(cell)
                multi_cell = tf.contrib.rnn.MultiRNNCell(cell_list)

                encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
                    multi_cell,
                    encoder_emb_imp,
                    dtype=tf.float32,
                    sequence_length=self.source_sequence_length,
                    swap_memory=True)

            ### Decoder
            with tf.variable_scope("decoder", dtype=tf.float32) as scope:
                cell_list = []
                for i range(self.num_layers):
                    cell = tf.contrib.rnn.BasicLSTMCell(self.num_units, forget_bias=self.forget_bias)
                    if self.dropout > 0.0:
                        cell = tf.contrib.rnn.DropoutWrapper(cell=cell, input_keep_prob=(1.0 - dropout))
                    cell_list.append(cell)
                multi_cell = tf.contrib.rnn.MultiRNNCell(cell_list)

                decoder_initial_state = encoder_state

                decoder_emb_imp = tf.nn.embedding_lookup(embed_decoder, [max_time, batch_size, self.num_units])

                helper = tf.contrib.seq2seq.TrainingHelper(decoder_emb_imp, target_sequence_length)

                my_decoder = tf.contrib.seq2seq.BasicDecoder(multi_cell, helpder, decoder_initial_state,)

                outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
                    my_decoder,
                    swap_memory=True,
                    scope=scope)

                sample_id = outputs.sample_id

                logits = self.output_layer(outputs.rnn_output)

            ### Loss
            target_output = self.iterator.target_output
            max_time = ;
            crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_output, logits=logits)
            target_weight = tf.sequence_mask(self.iterator.target_sequence_length, max_time, dtype=logits.dtype)
            loss = tf.reduce_sum(crossent * target_weights) / tf.to_float(self.batch_size)

        ### Optimizer
        self.train_loss = loss
        self.word_count = tf.reduce_sum(self.iterator.source_sequence_length) + tf.reduce_sum(self.iterator.target_sequence_length)
        ##### Count the number of predicted words for compute ppl.
        self.predict_count = tf.reduce_sum(self.iterator.target_sequence_length)
        self.global_step = tf.Variable(0, trainable=False)
        params = tf.trainable_variables()

        self.learning_rate = tf.constant(self.learning_rate)

        optimizer = "sgd"
        if optimizer == "sgd":
            opt = tf.train.GradientDescentOptimizer(self.learning_rate)
            tf.summary.scalar("lr", self.learning_rate)
        elif optimizer == "adam":
            opt = tf.train.AdamOptimizer(self.learning_rate)

        # Gradients
        gradients = tf.gradients(
            self.train_loss,
            params,
            colocate_gradients_with_ops=self.colocate_gradients_with_ops)

        clipped_grads, grad_norm = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
        grad_norm_summary = [tf.summary.scalar("grad_norm", grad_norm)]
        grad_norm_summary.append(tf.summary.scalar("clipped_gradient", tf.global_norm(clipped_grads)))
        self.grad_norm = grad_norm

        self.update = opt.apply_gradients(
            zip(clipped_grads, params), global_step=self.global_step)

        self.train_summary = tf.summary.merge([
            tf.summary.scalar("lr", self.learning_rate),
            tf.summary.scalar("train_loss", self.train_loss),
        ] + grad_norm_summary)

        # Saver
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=self.num_keep_ckpts)

        with tf.Session() as sess:
            step_results = sess.run([self.update,
                                     self.train_loss,
                                     self.predict_count,
                                     self.train_summary,
                                     self.global_step,
                                     self.word_count,
                                     self.batch_size,
                                     self.grad_norm,
                                     self.learning_rate])

if __name__ == "__main__":
    print("MAIN")
### EOF
