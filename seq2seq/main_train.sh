#!/bin/bash

python main.py \
       --src_file=./data/faq_title_mrphs_fix.txt \
       --tgt_file=./data/faq_body_mrphs_fix.txt \
       --src_vocab_file=./data/vocab_file.txt \
       --tgt_vocab_file=./data/vocab_file.txt \
       --share_vocab=true \
       --out_dir=./model \
       --num_train_steps=100 \
       --steps_per_stats=100 \
       --num_layers=2 \
       --num_units=128 \
       --dropout=0.2
### EOF
