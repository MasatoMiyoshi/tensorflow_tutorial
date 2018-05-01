# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals, absolute_import
import sys
import os
import codecs
import abc

import numpy as np
import tensorflow as tf
from tensorflow.python.layers import core as layers_core

import model_helper
from utils import iterator_utils
from utils import misc_utils as utils

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
        self.time_major = hparams.time_major

        self.single_cell_fn = None

        # Set num layers
        self.num_encoder_layers = hparams.num_encoder_layers
        self.num_decoder_layers = hparams.num_decoder_layers
        assert self.num_encoder_layers
        assert self.num_decoder_layers

        # Initializer
        initializer = model_helper.get_initializer(
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

        if self.mode == tf.estimator.ModeKeys.TRAIN:
            self.train_loss = res[1]
            self.word_count = tf.reduce_sum(self.iterator.source_sequence_length) + tf.reduce_sum(self.iterator.target_sequence_length)
        elif self.mode == tf.estimator.ModeKeys.EVAL:
            self.eval_loss = res[1]
        elif self.mode == tf.estimator.ModeKeys.PREDICT:
            self.infer_logits, _, self.final_context_state, self.sample_id = res
            self.sample_words = reverse_target_vocab_table.lookup(tf.to_int64(self.sample_id))

        if self.mode != tf.estimator.ModeKeys.PREDICT:
            ## Count the number of predicted words for compute ppl.
            self.predict_count = tf.reduce_sum(self.iterator.target_sequence_length)    
            
        self.global_step = tf.Variable(0, trainable=False)
        params = tf.trainable_variables()

        # Gradients and SGD update operation for training the model.
        # Arrange for the embedding vars to appear at the beginning.
        if self.mode == tf.estimator.ModeKeys.TRAIN:
            self.learning_rate = tf.constant(hparams.learning_rate)
            # warm-up
            self.learning_rate = self._get_learning_rate_warmup(hparams)
            # decay
            self.learning_rate = self._get_learning_rate_decay(hparams)

            # Optimizer
            if hparams.optimizer == "sgd":
                opt = tf.train.GradientDescentOptimizer(self.learning_rate)
                tf.summary.scalar("lr", self.learning_rate)
            elif optimizer == "adam":
                opt = tf.train.AdamOptimizer(self.learning_rate)

            # Gradients
            gradients = tf.gradients(
                self.train_loss,
                params,
                colocate_gradients_with_ops=hparams.colocate_gradients_with_ops)

            clipped_grads, grad_norm_summary, grad_norm = model_helper.gradient_clip(gradients, max_gradient_norm=hparams.max_gradient_norm)
            self.grad_norm = grad_norm

            self.update = opt.apply_gradients(zip(clipped_grads, params), global_step=self.global_step)

            # Summary
            self.train_summary = tf.summary.merge([
                tf.summary.scalar("lr", self.learning_rate),
                tf.summary.scalar("train_loss", self.train_loss),
            ] + grad_norm_summary)

        if self.mode == tf.estimator.ModeKeys.PREDICT:
            self.infer_summary = self._get_infer_summary(hparams)

        # Saver
        self.saver = tf.train.Saver(tf.global_variables())

        # Print trainable variables
        utils.print_out("# Trainable variables")
        for param in params:
            utils.print_out("  %s, %s, %s" % (param.name, str(param.get_shape()), param.op.device))

    def init_embeddings(self, hparams):
        self.embedding_encoder, self.embedding_decoder = (
            model_helper.create_emb_for_encoder_and_decoder(
                share_vocab=hparams.share_vocab,
                src_vocab_size=self.src_vocab_size,
                tgt_vocab_size=self.tgt_vocab_size,
                src_embed_size=hparams.num_units,
                tgt_embed_size=hparams.num_units,
                src_vocab_file=hparams.src_vocab_file,
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
            if self.mode != tf.estimator.ModeKeys.PREDICT:
                with tf.device("/cpu:0"):
                    loss = self._compute_loss(logits)
            else:
                loss = None

            return logits, loss, final_context_state, sample_id

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
                if self.time_major:
                    target_input = tf.transpose(target_input)
                decoder_emp_inp = tf.nn.embedding_lookup(self.embedding_decoder, target_input)
                # Helper
                helper = tf.contrib.seq2seq.TrainingHelper(decoder_emp_inp, iterator.target_sequence_length,
                                                           time_major=self.time_major)
                # Decoder
                my_decoder = tf.contrib.seq2seq.BasicDecoder(cell, helper, decoder_initial_state)
                # Dynamic decoding
                outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
                    my_decoder,
                    output_time_major=self.time_major,
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
                    output_time_major=self.time_major,
                    swap_memory=True,
                    scope=decoder_scope)

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

    def _compute_loss(self, logits):
        target_output = self.iterator.target_output
        if self.time_major:
            target_output = tf.transpose(target_output)
        max_time = self.get_max_time(target_output)
        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_output, logits=logits)
        target_weights = tf.sequence_mask(self.iterator.target_sequence_length, max_time, dtype=logits.dtype)
        if self.time_major:
            target_weights = tf.transpose(target_weights)

        loss = tf.reduce_sum(crossent * target_weights) / tf.to_float(self.batch_size)
        return loss

    def get_max_time(self, tensor):
        time_axis = 0 if self.time_major else 1
        return tensor.shape[time_axis].value or tf.shape(tensor)[time_axis]

    def _get_learning_rate_warmup(self, hparams):
        warmup_steps = hparams.warmup_steps
        warmup_scheme = hparams.warmup_scheme
        utils.print_out("  learning_rate=%g, warmup_steps=%d, warmup_scheme=%s" %
                        (hparams.learning_rate, warmup_steps, warmup_scheme))

        # Apply inverse decay if global steps less than warmup steps.
        # Inspired by https://arxiv.org/pdf/1706.03762.pdf (Section 5.3)
        # When step < warmup_steps,
        #   learning_rate *= warmup_factor ** (warmup_steps - step)
        if warmup_scheme == "t2t":
            # 0.01^(1/warmup_steps): we start with a lr, 100 times smaller
            warmup_factor = tf.exp(tf.log(0.01) / warmup_steps)
            inv_decay = warmup_factor**(tf.to_float(warmup_steps - self.global_step))
        else:
            raise ValueError("Unknown warmup scheme %s" % warmup_scheme)

        return tf.cond(self.global_step < hparams.warmup_steps,
                       lambda: inv_decay * self.learning_rate,
                       lambda: self.learning_rate,
                       name="learning_rate_warmup_cond")

    def _get_learning_rate_decay(self, hparams):
        """Get learning rate decay."""
        if hparams.decay_scheme in ["luong5", "luong10", "luong234"]:
            decay_factor = 0.5
            if hparams.decay_scheme == "luong5":
                start_decay_step = int(hparams.num_train_steps / 2)
                decay_times = 5
            elif hparams.decay_scheme == "luong10":
                start_decay_step = int(hparams.num_train_steps / 2)
                decay_times = 10
            elif hparams.decay_scheme == "luong234":
                start_decay_step = int(hparams.num_train_steps * 2 / 3)
                decay_times = 4
            remain_steps = hparams.num_train_steps - start_decay_step
            decay_steps = int(remain_steps / decay_times)
        elif not hparams.decay_scheme:  # no decay
            start_decay_step = hparams.num_train_steps
            decay_steps = 0
            decay_factor = 1.0
        elif hparams.decay_scheme:
            raise ValueError("Unknown decay scheme %s" % hparams.decay_scheme)
        utils.print_out("  decay_scheme=%s, start_decay_step=%d, decay_steps %d, "
                        "decay_factor %g" % (hparams.decay_scheme,
                                             start_decay_step,
                                             decay_steps,
                                             decay_factor))

        return tf.cond(
            self.global_step < start_decay_step,
            lambda: self.learning_rate,
            lambda: tf.train.exponential_decay(
                self.learning_rate,
                (self.global_step - start_decay_step),
                decay_steps, decay_factor, staircase=True),
            name="learning_rate_decay_cond")

    def _get_infer_summary(self, hparams):
        return tf.no_op()

    def train(self, sess):
        assert self.mode == tf.estimator.ModeKeys.TRAIN
        return sess.run([self.update,
                         self.train_loss,
                         self.predict_count,
                         self.train_summary,
                         self.global_step,
                         self.word_count,
                         self.batch_size,
                         self.grad_norm,
                         self.learning_rate])

    def infer(self, sess):
        assert self.mode == tf.estimator.ModeKeys.PREDICT
        return sess.run([
            self.infer_logits, self.infer_summary, self.sample_id, self.sample_words
        ])

    def decode(self, sess):
        _, infer_summary, _, sample_words = self.infer(sess)

        # make sure outputs is of shape [batch_size, time] or [beam_width,
        # batch_size, time] when using beam search.
        if self.time_major:
            sample_words = sample_words.transpose()
        elif sample_words.ndim == 3:  # beam search output in [batch_size,
                                      # time, beam_width] shape.
            sample_words = sample_words.transpose([2, 0, 1])
        return sample_words, infer_summary

class Model(BaseModel):
    def _build_encoder(self, hparams):
        num_layers = self.num_encoder_layers
        iterator = self.iterator

        source = iterator.source
        if self.time_major:
            source = tf.transpose(source)

        with tf.variable_scope("encoder") as scope:
            dtype=scope.dtype
            # Look up embedding, emp_inp: [max_time, batch_size, num_units]
            encoder_emp_inp = tf.nn.embedding_lookup(self.embedding_encoder, source)

            print("  num_layers = %d" % (num_layers))
            cell = self._build_encoder_cell(hparams, num_layers)
            encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
                cell,
                encoder_emp_inp,
                dtype=dtype,
                sequence_length=iterator.source_sequence_length,
                time_major=self.time_major,
                swap_memory=True)

        return encoder_outputs, encoder_state

    def _build_decoder_cell(self, hparams, encoder_outputs, encoder_state, source_sequence_length):
        cell = model_helper.create_rnn_cell(
            unit_type=hparams.unit_type,
            num_units=hparams.num_units,
            num_layers=hparams.num_decoder_layers,
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
### EOF
