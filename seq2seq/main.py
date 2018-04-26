# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals, absolute_import
import sys
import os
import codecs
import argparse
import numpy as np
import tensorflow as tf
import train

FLAGS = None

def add_arguments(parse):
    """Build ArgumentParser."""
    parser.register("type", "bool", lambda v: v.lower() == "true")

    # Data
    parser.add_argument("--src_file", type=str, default=None)
    parser.add_argument("--tgt_file", type=str, default=None)
    parser.add_argument("--src_vocab_file", type=str, default=None)
    parser.add_argument("--tgt_vocab_file", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None, help="Store log/model files.")

    # Vocab
    parser.add_argument("--sos", type=str, default="<s>", help="Start-of-sentence symbol.")
    parser.add_argument("--eos", type=str, default="</s>", help="End-of-sentence symbol.")
    parser.add_argument("--share_vocab", type="bool", nargs="?", const=True, default=False,
                        help="Whether to use the source vocab and embeddings for both source and target.")
    parser.add_argument("--check_special_token", type="bool", default=True,
                        help="Whether check special sos, eos, unk tokens exist in the vocab files.")

    # Sequence lengths
    parser.add_argument("--src_max_len", type=int, default=50,
                        help="Max length of src sequences during training.")
    parser.add_argument("--tgt_max_len", type=int, default=50,
                        help="Max length of tgt sequences during training.")
    parser.add_argument("--src_max_len_infer", type=int, default=None,
                        help="Max length of src sequences during inference.")
    parser.add_argument("--tgt_max_len_infer", type=int, default=None,
                        help="Max length of tgt sequences during inference. Also use to restrict the maximum decoding length.")

    # Network
    parser.add_argument("--num_units", type=int, default=32, help="Network size.")
    parser.add_argument("--num_layers", type=int, default=2, help="Network depth.")
    parser.add_argument("--num_encoder_layers", type=int, default=None,
                        help="Encoder depth, equal to num_layers if None.")
    parser.add_argument("--num_decoder_layers", type=int, default=None,
                        help="Decoder depth, equal to num_layers if None.")
    parser.add_argument("--time_major", type="bool", nargs="?", const=True, default=True,
                        help="Whether to use time-major mode for dynamic RNN.")

    # Optimizer
    parser.add_argument("--optimizer", type=str, default="sgd", help="sgd | adam")
    parser.add_argument("--learning_rate", type=float, default=1.0, help="Learning rate. Adam: 0.001 | 0.0001")
    parser.add_argument("--warmup_steps", type=int, default=0, help="How many steps we inverse-decay learning.")
    parser.add_argument("--warmup_scheme", type=str, default="t2t", help="""\
       How to warmup learning rates. Options include:
         t2t: Tensor2Tensor's way, start with lr 100 times smaller, then
              exponentiate until the specified lr.\
       """)
    parser.add_argument("--decay_scheme", type=str, default="", help="""\
       How we decay learning rate. Options include:
         luong234: after 2/3 num train steps, we start halving the learning rate
           for 4 times before finishing.
         luong5: after 1/2 num train steps, we start halving the learning rate
           for 5 times before finishing.\
         luong10: after 1/2 num train steps, we start halving the learning rate
           for 10 times before finishing.\
       """)
    parser.add_argument("--num_train_steps", type=int, default=12000, help="Num steps to train.")
    parser.add_argument("--colocate_gradients_with_ops", type="bool", nargs="?",
                        const=True, default=True, help="Whether try colocating gradients with corresponding op")

    # Inference
    parser.add_argument("--ckpt", type=str, default="",
                        help="Checkpoint file to load a model for inference.")
    parser.add_argument("--inference_input_file", type=str, default=None,
                        help="Set to the text to decode.")
    parser.add_argument("--inference_list", type=str, default=None,
                        help="A comma-separated list of sentence indices (0-based) to decode.")
    parser.add_argument("--infer_batch_size", type=int, default=32,
                        help="Batch size for inference mode.")
    parser.add_argument("--inference_output_file", type=str, default=None,
                        help="Output file to store decoding results.")
    parser.add_argument("--beam_width", type=int, default=0,
                        help=("""\
       beam width when using beam search decoder. If 0 (default), use standard
       decoder with greedy helper.\
       """))
    parser.add_argument("--length_penalty_weight", type=float, default=0.0,
                        help="Length penalty for beam search.")
    parser.add_argument("--sampling_temperature", type=float, default=0.0,
                        help=("""\
       Softmax sampling temperature for inference decoding, 0.0 means greedy
       decoding. This option is ignored when using beam search.\
       """))
    parser.add_argument("--num_translations_per_input", type=int, default=1,
                        help=("""\
       Number of translations generated for each sentence. This is only used for
       inference.\
       """))

    # Initializer
    parser.add_argument("--init_op", type=str, default="uniform",
                        help="uniform | glorot_normal | glorot_uniform")
    parser.add_argument("--init_weight", type=float, default=0.1,
                        help=("for uniform init_op, initialize weights between [-this, this]."))

    # Default settings works well (rarely need to change)
    parser.add_argument("--unit_type", type=str, default="lstm", help="lstm | gru | layer_norm_lstm | nas")
    parser.add_argument("--forget-bias", type=float, default=1.0, help="Forget bias for BasicLSTMCell.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate (not keep_prob)")
    parser.add_argument("--max_gradient_norm", type=float, default=5.0, help="Clip gradients to this norm.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
    parser.add_argument("--steps_per_stats", type=int, default=100,
                        help=("How many training steps to do per stats logging."
                              "Save checkpoint every 10x steps_per_stats"))
    parser.add_arguemtn("--max_train", type=int, default=0,
                        help="Limit on the size of training data (0: no limit).")
    parser.add_argument("--num_buckets", type=int, default=5,
                        help="Put data into similar-length buckets.")

    # Job info
    parser.add_argument("--num_inter_threads", type=int, default=0,
                        help="number of inter_op_parallelism_threads")
    parser.add_argument("--num_intra_threads", type=int, default=0,
                        help="number of intra_op_parallelism_threads")

    # Misc
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of gpus in each worker.")
    parser.add_argument("--log_device_placement", type="bool", nargs="?",
                        const=True, default=False, help="Debug GPU allocation.")
    parser.add_argument("--random_seed", type=int, default=None,
                        help="Random seed (>0, set a specific seed).")

def create_hparams(flags):
    return tf.contrib.training.HParams(
        # Data
        src_file=flags.src_file,
        tgt_file=flags.tgt_file,
        src_vocab_file=flags.src_vocab_file,
        tgt_vocab_file=flags.tgt_vocab_file,
        out_dir=flags.out_dir,
        num_buckets=flags.num_buckets,
        max_train=flags.max_train,
        src_max_len=flags.src_max_len,
        tgt_max_len=flags.tgt_max_len,

        # Vocab
        sos=flags.sos,
        eos=flags.eos,
        share_vocab=flags.share_vocab,
        check_special_token=flags.check_special_token,

        # Sequence lengths
        src_max_len=flags.src_max_len,
        tgt_max_len=flags.tgt_max_len,
        src_max_len_infer=flags.src_max_len_infer,
        tgt_max_len_infer=flags.tgt_max_len_infer,

        # Networks
        num_units=flags.num_units,
        num_layers=flags.num_layers,
        num_encoder_layers=(flags.num_encoder_layers or flags.num_layers),
        num_decoder_layers=(flags.num_decoder_layers or flags.num_layers),
        dropout=flags.dropout,
        unit_type=flags.unit_type,
        time_major=flags.time_major,

        # Optimizer
        optimizer=flags.optimizer,
        learning_rate=flags.learning_rate,
        warmup_steps=flags.warmup_steps,
        warmup_scheme=flags.warmup_scheme,
        decay_scheme=flags.decay_scheme,
        num_train_steps=flags.num_train_steps,
        batch_size=flags.batch_size,
        forget_bias=flags.forget_bias,
        colocate_gradients_with_ops=flags.colocate_gradients_with_ops,
        max_gradient_norm=flags.max_gradient_norm,

        # Inference
        src_max_len_infer=flags.src_max_len_infer,
        tgt_max_len_infer=flags.tgt_max_len_infer,
        infer_batch_size=flags.infer_batch_size,
        beam_width=flags.beam_width,
        length_penalty_weight=flags.length_penalty_weight,
        sampling_temperature=flags.sampling_temperature,
        num_translations_per_input=flags.num_translations_per_input,

        # Initializer
        init_op=flags.init_op,
        init_weight=flags.init_weight,

        # Misc
        epoch_step=0,
        steps_per_stats=flags.steps_per_stats,
        log_device_placement=flags.log_device_placement,
        random_seed=flags.random_seed,
        num_intra_threads=flags.num_intra_threads,
        num_inter_threads=flags.num_inter_threads,
    )

def run_main(flags, default_hparams, train_fn):
    ## Train / Decode
    out_dir = flags.out_dir
    if not tf.gfile.Exists(out_dir): tf.gfile.MakeDirs(out_dir)
    # Train
    # train_fn(default_hparams)

def main(unused_argv):
    default_hparams = create_hparams(FLAGS)
    train_fn = train.train
    # inference_fn = inference.inference
    run_main(FLAGS, default_hparams, train_fn)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    FLAGS, unparsed = parser.parser_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
### EOF
