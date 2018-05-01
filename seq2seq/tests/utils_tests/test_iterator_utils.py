# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals, absolute_import
import sys
import os
import tensorflow as tf
from tensorflow.python.ops import lookup_ops
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../..')
from utils import iterator_utils

class TestIteratorUtils(tf.test.TestCase):
    def test_get_iterator(self):
        tf.set_random_seed(1)
        src_vocab_table = tgt_vocab_table = lookup_ops.index_table_from_tensor(
            tf.constant(["a", "b", "c", "eos", "sos"]))
        src_dataset = tf.data.Dataset.from_tensor_slices(
            tf.constant(["f e a g", "c c a", "d", "c a"]))
        tgt_dataset = tf.data.Dataset.from_tensor_slices(
            tf.constant(["c c", "a b", "", "b c"]))
        hparams = tf.contrib.training.HParams(
            random_seed=3,
            num_buckets=1,
            eos="eos",
            sos="sos")
        batch_size = 2
        src_max_len = 3
        iterator = iterator_utils.get_iterator(
            src_dataset=src_dataset,
            tgt_dataset=tgt_dataset,
            src_vocab_table=src_vocab_table,
            tgt_vocab_table=tgt_vocab_table,
            batch_size=batch_size,
            sos=hparams.sos,
            eos=hparams.eos,
            random_seed=hparams.random_seed,
            num_buckets=hparams.num_buckets,
            src_max_len=src_max_len,
            reshuffle_each_iteration=False,
            delimiter=" ")
        table_initializer = tf.tables_initializer()
        source = iterator.source
        src_seq_len = iterator.source_sequence_length
        self.assertEqual([None, None], source.shape.as_list())
        with self.test_session() as sess:
            sess.run(table_initializer)
            sess.run(iterator.initializer)
            (source_v, src_seq_len_v) = sess.run((source, src_seq_len))
            print(source_v)
            print(src_seq_len_v)

            (source_v, src_seq_len_v) = sess.run((source, src_seq_len))
            print(source_v)
            print(src_seq_len_v)

    def test_get_iterator_jp(self):
        tf.set_random_seed(1)
        src_vocab_table = tgt_vocab_table = lookup_ops.index_table_from_tensor(
            tf.constant(["あ", "い", "う", "eos", "sos"]))
        src_dataset = tf.data.Dataset.from_tensor_slices(
            tf.constant(["か\tお\tあ\tき", "う\tう\tあ", "え", "う\tあ"]))
        tgt_dataset = tf.data.Dataset.from_tensor_slices(
            tf.constant(["う\tう", "あ\tい", "", "い\tう"]))
        hparams = tf.contrib.training.HParams(
            random_seed=3,
            num_buckets=1,
            eos="eos",
            sos="sos")
        batch_size = 2
        src_max_len = 3
        iterator = iterator_utils.get_iterator(
            src_dataset=src_dataset,
            tgt_dataset=tgt_dataset,
            src_vocab_table=src_vocab_table,
            tgt_vocab_table=tgt_vocab_table,
            batch_size=batch_size,
            sos=hparams.sos,
            eos=hparams.eos,
            random_seed=hparams.random_seed,
            num_buckets=hparams.num_buckets,
            src_max_len=src_max_len,
            reshuffle_each_iteration=False)
        table_initializer = tf.tables_initializer()
        source = iterator.source
        src_seq_len = iterator.source_sequence_length
        self.assertEqual([None, None], source.shape.as_list())
        with self.test_session() as sess:
            sess.run(table_initializer)
            sess.run(iterator.initializer)
            (source_v, src_seq_len_v) = sess.run((source, src_seq_len))
            (source_v, src_seq_len_v) = sess.run((source, src_seq_len))

            print(source_v)
            print(src_seq_len_v)

if __name__ == "__main__":
    tf.test.main()
### EOF
