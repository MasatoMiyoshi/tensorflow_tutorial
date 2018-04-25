# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals, absolute_import
import sys
import os
import codecs
import argparse

import numpy as np
import tensorflow as tf

from . import model_helper
from .utils import iterator_utils

class BaseModel(object):
    def __init__(self,
                 hparams,
                 mode,
                 iterator,
                 source_vocab_table,
                 target_vocab_table,
                 reverse_target_vocab_table=None):

        assert isinstance(iterator, iterator_utils.BatchedInput)
        self.iterator = iterator
        self.mode = mode
        self.src_vocab_table = source_vocab_table
        self.tgt_vocab_table = target_vocab_table

        self.src_vocab_size = hparams.src_vocab_size
        self.tgt_vocab_size = hparams.tgt_vocab_size

        self.single_cell_fn = None

        # Set num layers
        self.num_encoder_layers = hparams.num_encoder_layers
        self.num_decoder_layers = hparams.num_decoder_layers
        assert self.num_encoder_layers
        assert self.num_decoder_layers

        # Initializer
        initializer = model_helpder.get_initializer(
            hparams.init_op, hparams.random_seed, hparams.init_weight)
        tf.get_variable_scope().set_initializer(initializer)

        # Embeddings
        self.init_embeddings(hparams)
        self.batch_size = tf.size(self.iterator.source_sequence_length)

        # Projection
        with tf.variable_scope("build_netword"):
            with tf.variable_scope("decoder/output_projection"):
                self.output_layer = layers_core.Dense(hparams.tgt_vocab_size, use_bias=False, name="output_projection")


        ## Train graph
        res = self.build_graph(hparams)

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

    def init_embeddings(self, hparams):
        self.embedding_encoder, self.embedding_decoder = (
            model_helper.create_emb_for_encoder_and_decoder(
                share_vocab=hparams.share_vocab,
                src_vocab_size=self.src_vocab_size,
                tgt_vocab_size=self.tgt_vocab_size,
                src_embed_size=hparams.num_units,
                tgt_embed_size=hparams.num_units,
                src_vocab_file=hparams.srv_vocab_file,
                tgt_vocab_file=hparams.tgt_vocab_file))

    def build_graph(self, hparams):
        print("# createing %s graph ..." % self.mode)
        dtype = tf.float32

        with tf.variable_scope("dynamic_seq2seq", dtype=dtype):
            ### Encoder
            encoder_outputs, encoder_state = self._build_encoder(hparams)

            ### Decoder
            logits, sample_id, final_context_state = self._build_decoder(encoder_outputs, encoder_state, hparams)

            ### Loss
            target_output = self.iterator.target_output
            max_time = ;
            crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_output, logits=logits)
            target_weight = tf.sequence_mask(self.iterator.target_sequence_length, max_time, dtype=logits.dtype)
            loss = tf.reduce_sum(crossent * target_weights) / tf.to_float(self.batch_size)

    def _build_encoder(self, hparams):
        pass

    def _build_encoder_cell(self, hparams, num_layers):
        return model_helper.create_rnn_cell(
            unit_type=hparams.unit_type,
            num_units=hparams.num_units,
            num_layers=num_layers,
            forget_bias=hparams.forget_bias,
            dropout=hparams.dropout,
            mode=self.mode,
            single_cell_fn=self.single_cell_fn)

    def _get_infer_maximum_iterations(self, hparams, source_sequence_length):
        return tf.constant(1000, dtype=tf.int32)

    def _build_decoder(self, encoder_outputs, encoder_state, hparams):
        tgt_sos_id = tf.cast(self.tgt_vocab_table.lookup(tf.constant(hparams.sos)), tf.int32)
        tgt_eos_id = tf.cast(self.tgt_vocab_table.lookup(tf.constant(hparams.eos)), tf.int32)
        iterator = self.iterator

        # maximum_iteration: The maximum decoding steps.
        maximum_iterations = self._get_infer_maximum_iterations(hparams, iterator.source_sequence_length)

        ## Decoder.
        with tf.variable_scope("decoder") as decoder_scope:
            cell, decoder_initial_state = self._build_decoder_cell(hparams, encoder_outputs, encoder_state, iterator.source_sequence_length)

            ## Train or eval
            if self.mode != tf.estimator.ModeKeys.PREDICT:
                # decoder_emp_inp: [max_time, batch_size, num_units]
                target_input = iterator.target_input
                decoder_emp_inp = tf.nn.embedding_lookup(self.embedding_decoder, target_input)
                # Helper
                helper = tf.contrib.seq2seq.TrainingHelper(decoder_emp_inp, iterator.target_sequence_length)
                # Decoder
                my_decoder = tf.contrib.seq2seq.BasicDecoder(cell, helpder, decoder_initial_state)
                # Dynamic decoding
                outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
                    my_decoder,
                    swap_memory=True,
                    scope=decoder_scope)

                sample_id = outputs.sample_id
                logits = self.output_layer(outputs.rnn_output)
            ## Inference
            else:
                beam_width = hparams.beam_width
                length_penalty_weight = hparams.length_penalty_weight
                start_tokens = tf.fill([self.batch_size], tgt_sos_id)
                end_token = tgt_eos_id

                if beam_width > 0:
                    my_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                        cell=cell,
                        embedding=self.embedding_decoder,
                        start_tokens=start_tokens,
                        end_token=end_token,
                        initial_state=decoder_initial_state,
                        beam_width=beam_width,
                        output_layer=self.output_layer,
                        length_penalty_weight=length_penalty_weight)
                else:
                    # Helper
                    sampling_temperature = hparams.sampling_temperature
                    if sampling_temperature > 0.0:
                        helper = tf.contrib.seq2seq.SampleEmbeddingHelper(
                            self.embedding_decoder, start_tokens, end_token,
                            softmax_temperature=sampling_temperature,
                            seed=hparams.random_seed)
                    else:
                        helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                            self.embedding_decoder, start_tokens, end_token)

                    # Decoder
                    my_decoder = tf.contrib.seq2seq.BasicDecoder(
                        cell,
                        helper,
                        decoder_initial_state,
                        output_layer=self.output_layer  # applied per timestep
                    )

                # Dynamic decoding
                outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
                    my_decoder,
                    maximum_iterations=maximum_iterations,
                    swap_memory=True)

                if beam_width > 0:
                    logits = tf.np_op()
                    sample_id = outputs.predicted_ids
                else:
                    logits = outputs.rnn_output
                    sample_id = outputs.sample_id

        return logits, sample_id, final_context_state

    @abc.abstractmethod
    def _build_decoder_cell(self, hparams, encoder_outputs, encoder_state, source_sequence_length):
        pass

    def train(self):
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

class Model(BaseModel):
    def _build_encoder(self, hparams):
        num_layers = self.num_encoder_layers
        iterator = self.iterator

        source = iterator.source

        with tf.variable_scope("encoder") as scope:
            dtype=scope.dtype
            # Look up embedding, emp_inp: [max_time, batch_size, num_units]
            encoder_emb_imp = tf.nn.embedding_lookup(embed_encoder, source)

            print("  num_layers = %d" % (num_layers))
            cell = _build_encoder_cell(hparams, num_layers)
            encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
                cell,
                encoder_emb_imp,
                dtype=dtype,
                sequence_length=iterator.source_sequence_length,
                swap_memory=True)

        return encoder_outputs, encoder_state

    def _build_decoder_cell(self, hparams, encoder_outputs, encoder_state, source_sequence_length):
        cell = model_helper.create_rnn_cell(
            unit_type=hparams.unit_type,
            num_units=hparams.num_units,
            num_layers=num_layers,
            forget_bias=hparams.forget_bias,
            dropout=hparams.dropout,
            mode=self.mode,
            single_cell_fn=self.single_cell_fn)

        # For beam search, we need to replicate encoder infos beam_width times
        if self.mode == tf.estimator.ModeKeys.PREDICT and hparams.beam_width > 0:
            decoder_initial_state = tf.contrib.seq2seq.tile_batch(encoder_state, multiplier=hparams.beam_width)
        else:
            decoder_initial_state = encoder_state

        return cell, decoder_initial_state

if __name__ == "__main__":
    print("MAIN")
### EOF
