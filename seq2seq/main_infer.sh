#!/bin/bash

python main.py \
       --src_vocab_file=./data/vocab_file.txt \
       --tgt_vocab_file=./data/vocab_file.txt \
       --share_vocab=true \
       --out_dir=./model \
       --num_units=128 \
       --inference_input_file=./data/infer.txt \
       --inference_output_file=./model/output_infer
### EOF
