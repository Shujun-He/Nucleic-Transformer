#!/bin/bash


python train.py --gpu_id 0,1 --kmer_aggregation --nmute 15 --epochs 100 --nlayers 1 \
--batch_size 2048 --kmers 7 --lr_scale 0.5 --path ../../data --workers 8 \
--dropout 0.1 --nclass 202 --ntoken 16 --nhead 8 --ninp 512 --nhid 2048
