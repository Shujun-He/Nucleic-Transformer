#!/bin/bash
for i in {0..4};do
python train.py --fold $i --kmer_aggregation --epochs 50 --nlayers 6 --nmute 15 \
--path bert_enhancer_dataset.csv \
--kmers 7 --ninp 256 --nhid 1024
done
