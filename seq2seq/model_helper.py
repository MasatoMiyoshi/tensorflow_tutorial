# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals, absolute_import
import sys
import os
import codecs
import time
import collections

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import lookup_ops

from utils import iterator_utils
from utils import vocab_utils
from utils import misc_utils as utils

def get_initializer(init_op, seed=None, init_weight=None):
    if init_op == "uniform":
        assert init_weight
        return tf.random_uniform_initializer(
            -init_weight, init_weight, seed=seed)
    elif init_op == "glorot_normal":
        return tf.keras.initializers.glorot_normal(seed=seed)
    elif init_op == "glorot_uniform":
        return tf.keras.initializers.glorot_uniform(seed=seed)
    else:
        raise ValueError("Unknown init_op %s" % init_op)

def create_emb_for_encoder_and_decoder(share_vocab,
                                       src_vocab_size,
                                       tgt_vocab_size,
                                       src_embed_size,
                                       tgt_embed_size,
                                       dtype=tf.float32,
                                       src_vocab_file=None,
                                       tgt_vocab_file=None):
    partitioner = None

    with tf.variable_scope("embeddings", dtype=dtype, partitioner=partitioner) as scope:
        # Share embedding
        if share_vocab:
            if src_vocab_size != tgt_vocab_size:
                raise ValueError("Share embedding but different src/tgt vocab size"
                                 " %d vs. %d" % (src_vocab_size, tgt_vocab_size))
            assert src_embed_size == tgt_embed_size
            utils.print_out("# Use the same embedding for source and target")
            vocab_file = src_vocab_file or tgt_vocab_file

            embedding_encoder = _create_or_load_embed("embedding_share", vocab_file, src_vocab_size, src_embed_size, dtype)
            embedding_decoder = embedding_encoder
        else:
            with tf.variable_scope("encoder", partitioner=partitioner):
                embedding_encoder = _create_or_load_embed("embedding_encoder", src_vocab_file, src_vocab_size, src_embed_size, dtype)
            with tf.variable_scope("decoder", partitioner=partitioner):
                embedding_decoder = _create_or_load_embed("embedding_decoder", tgt_vocab_file, tgt_vocab_size, tgt_embed_size, dtype)

    return embedding_encoder, embedding_decoder

def _create_or_load_embed(embed_name, vocab_file, vocab_size, embed_size, dtype):
    with tf.device("/cpu:0"):
        embedding = tf.get_variable(embed_name, [vocab_size, embed_size], dtype)
    return embedding

def create_rnn_cell(unit_type, num_units, num_layers, forget_bias, dropout, mode, single_cell_fn=None):
    cell_list = _cell_list(unit_type=unit_type,
                           num_units=num_units,
                           num_layers=num_layers,
                           forget_bias=forget_bias,
                           dropout=dropout,
                           mode=mode,
                           single_cell_fn=single_cell_fn)

    if len(cell_list) == 1:  # Single layer.
        return cell_list[0]
    else:  # Multi layers
        return tf.contrib.rnn.MultiRNNCell(cell_list)

def _cell_list(unit_type, num_units, num_layers, forget_bias, dropout, mode, single_cell_fn=None):
    if not single_cell_fn:
        single_cell_fn = _single_cell

    cell_list = []
    for i in range(num_layers):
        utils.print_out("  cell %d" % i, new_line=False)
        single_cell = single_cell_fn(
            unit_type=unit_type,
            num_units=num_units,
            forget_bias=forget_bias,
            dropout=dropout,
            mode=mode
        )
        utils.print_out("")
        cell_list.append(single_cell)

    return cell_list

def _single_cell(unit_type, num_units, forget_bias, dropout, mode):
    dropout = dropout if mode == tf.estimator.ModeKeys.TRAIN else 0.0

    # Cell Type
    if unit_type == "lstm":
        utils.print_out("  LSTM, forget_bias=%g" % forget_bias, new_line=False)
        single_cell = tf.contrib.rnn.BasicLSTMCell(
            num_units,
            forget_bias=forget_bias)
    elif unit_type == "gru":
        utils.print_out("  GRU", new_line=False)
        single_cell = tf.contrib.rnn.GRUCell(num_units)
    else:
        raise ValueError("Unknown unit type %s!" % unit_type)

    # Dropout (= 1 - keep_prob)
    if dropout > 0.0:
        single_cell = tf.contrib.rnn.DropoutWrapper(cell=single_cell, input_keep_prob=(1.0 - dropout))
        utils.print_out("  %s, dropout=%g " % (type(single_cell).__name__, dropout), new_line=False)

    return single_cell

def gradient_clip(gradients, max_gradient_norm):
    """Clipping gradients of a model."""
    clipped_gradients, gradient_norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
    gradient_norm_summary = [tf.summary.scalar("grad_norm", gradient_norm)]
    gradient_norm_summary.append(tf.summary.scalar("clipped_gradient", tf.global_norm(clipped_gradients)))

    return clipped_gradients, gradient_norm_summary, gradient_norm

class TrainModel(
        collections.namedtuple("TrainModel", ("graph", "model", "iterator",
                                              "skip_count_placeholder"))):
    pass

def create_train_model(model_creator, hparams):
    """Create train graph, model, and iterator."""
    src_file = hparams.src_file
    tgt_file = hparams.tgt_file
    src_vocab_file = hparams.src_vocab_file
    tgt_vocab_file = hparams.tgt_vocab_file

    graph = tf.Graph()

    with graph.as_default(), tf.container("train"):
        src_vocab_table, tgt_vocab_table = vocab_utils.create_vocab_tables(
            src_vocab_file, tgt_vocab_file, hparams.share_vocab)

        src_dataset = tf.data.TextLineDataset(src_file)
        tgt_dataset = tf.data.TextLineDataset(tgt_file)
        skip_count_placeholder = tf.placeholder(shape=(), dtype=tf.int64)

        iterator = iterator_utils.get_iterator(
            src_dataset,
            tgt_dataset,
            src_vocab_table,
            tgt_vocab_table,
            batch_size=hparams.batch_size,
            sos=hparams.sos,
            eos=hparams.eos,
            random_seed=hparams.random_seed,
            num_buckets=hparams.num_buckets,
            src_max_len=hparams.src_max_len,
            tgt_max_len=hparams.tgt_max_len,
            skip_count=skip_count_placeholder)

        with tf.device("/cpu:0"):
            model = model_creator(
                hparams,
                iterator=iterator,
                mode=tf.estimator.ModeKeys.TRAIN,
                source_vocab_table=src_vocab_table,
                target_vocab_table=tgt_vocab_table)

    return TrainModel(
        graph=graph,
        model=model,
        iterator=iterator,
        skip_count_placeholder=skip_count_placeholder)

class InferModel(
        collections.namedtuple("InferModel",
                               ("graph", "model", "src_placeholder",
                                "batch_size_placeholder", "iterator"))):
    pass

def create_infer_model(model_creator, hparams):
    """Create inference model."""
    graph = tf.Graph()
    src_vocab_file = hparams.src_vocab_file
    tgt_vocab_file = hparams.tgt_vocab_file

    with graph.as_default(), tf.container("infer"):
        src_vocab_table, tgt_vocab_table = vocab_utils.create_vocab_tables(
            src_vocab_file, tgt_vocab_file, hparams.share_vocab)
        reverse_tgt_vocab_table = lookup_ops.index_to_string_table_from_file(
            tgt_vocab_file, default_value=vocab_utils.UNK)

        src_placeholder = tf.placeholder(shape=[None], dtype=tf.string)
        batch_size_placeholder = tf.placeholder(shape=[], dtype=tf.int64)

        src_dataset = tf.data.Dataset.from_tensor_slices(src_placeholder)

        iterator = iterator_utils.get_infer_iterator(
            src_dataset,
            src_vocab_table,
            batch_size=batch_size_placeholder,
            eos=hparams.eos,
            src_max_len=hparams.src_max_len_infer)

        model = model_creator(
            hparams,
            iterator=iterator,
            mode=tf.estimator.ModeKeys.PREDICT,
            source_vocab_table=src_vocab_table,
            target_vocab_table=tgt_vocab_table,
            reverse_target_vocab_table=reverse_tgt_vocab_table)

    return InferModel(
        graph=graph,
        model=model,
        src_placeholder=src_placeholder,
        batch_size_placeholder=batch_size_placeholder,
        iterator=iterator)

def load_model(model, ckpt, session, name):
    start_time = time.time()
    model.saver.restore(session, ckpt)
    session.run(tf.tables_initializer())
    utils.print_out("  loaded %s model parameters from %s, time %.2fs" % (name, ckpt, time.time() - start_time))
    return model

def create_or_load_model(model, model_dir, session, name):
    start_time = time.time()
    session.run(tf.global_variables_initializer())
    session.run(tf.tables_initializer())
    utils.print_out("  created %s model with fresh parameters, time %.2fs" %
                    (name, time.time() - start_time))

    global_step = model.global_step.eval(session=session)
    return model, global_step
### EOS
