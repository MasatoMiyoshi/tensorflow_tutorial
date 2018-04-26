# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals, absolute_import

import tensorflow as tf
from tensorflow.python.ops import lookup_ops

from utils import iterator_utils
from utils import standard_hparams_utils

def create_test_hparams(unit_type="lstm",
                        num_layers=4,
                        num_translations_per_input=1,
                        beam_width=0,
                        init_op="uniform"):
    standard_hparams = standard_hparams_utils.create_standard_hparams()

    # Networks
    standard_hparams.num_units = 5
    standard_hparams.num_encoder_layers = num_layers
    standard_hparams.num_decoder_layers = num_layers
    standard_hparams.dropout = 0.5
    standard_hparams.unit_type = unit_type
    # Train
    standard_hparams.init_op = init_op
    standard_hparams.num_train_steps = 1
    standard_hparams.decay_scheme = ""
    # Infer
    standard_hparams.tgt_max_len_infer = 100
    standard_hparams.beam_width = beam_width
    standard_hparams.num_translations_per_input = num_translations_per_input
    # Misc
    standard_hparams.forget_bias = 0.0
    standard_hparams.random_seed = 3
    # Vocab
    standard_hparams.src_vocab_size = 5
    standard_hparams.tgt_vocab_size = 5
    standard_hparams.eos = "eos"
    standard_hparams.sos = "sos"
    # For inference.py test
    # standard_hparams.subword_option = "bpe"
    # standard_hparams.src = "src"
    # standard_hparams.tgt = "tgt"
    standard_hparams.src_max_len = 400
    standard_hparams.tgt_eos_id = 0
    # standard_hparams.inference_indices = inference_indices
    return standard_hparams

def create_test_iterator(hparams, mode):
    src_vocab_table = lookup_ops.index_table_from_tensor(
        tf.constant([hparams.eos, "a", "b", "c", "d"]))
    tgt_vocab_mapping = tf.constant([hparams.sos, hparams.eos, "a", "b", "c"])
    tgt_vocab_table = lookup_ops.index_table_from_tensor(tgt_vocab_mapping)
    if mode == tf.estimator.ModeKeys.PREDICT:
        reverse_tgt_vocab_table = lookup_ops.index_to_string_table_from_tensor(tgt_vocab_mapping)

    src_dataset = tf.data.Dataset.from_tensor_slices(tf.constant(["a a b b c", "a b b"]))

    if mode != tf.estimator.ModeKeys.PREDICT:
        tgt_dataset = tf.data.Dataset.from_tensor_slices(
            tf.constant(["a b c b c", "a b c b"]))
        return (
            iterator_utils.get_iterator(
                src_dataset=src_dataset,
                tgt_dataset=tgt_dataset,
                src_vocab_table=src_vocab_table,
                tgt_vocab_table=tgt_vocab_table,
                batch_size=hparams.batch_size,
                sos=hparams.sos,
                eos=hparams.eos,
                random_seed=hparams.random_seed,
                num_buckets=hparams.num_buckets),
            src_vocab_table,
            tgt_vocab_table)
    else:
        return (
            iterator_utils.get_infer_iterator(
                src_dataset=src_dataset,
                src_vocab_table=src_vocab_table,
                eos=hparams.eos,
                batch_size=hparams.batch_size),
            src_vocab_table,
            tgt_vocab_table,
            reverse_tgt_vocab_table)
### EOF
