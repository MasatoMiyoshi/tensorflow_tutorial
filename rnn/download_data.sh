#!/bin/bash

curl -fsSLO http://www.anc.org/MASC/download/masc_500k_texts.tgz
tar xvf masc_500k_texts.tgz
mkdir corpus
mkdir data
mkdir log
cp -r masc_500k_texts/written/newspaper\:newswire corpus/newspaper_newswire
cp -r masc_500k_texts/written/non-fiction corpus/non-fiction
### EOF
