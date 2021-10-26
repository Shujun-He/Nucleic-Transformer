#!/bin/bash


python train.py --gpu_id 0 --kmer_aggregation --nmute 40 --epochs 60 --nlayers 6 \
--batch_size 128 --kmers 13 --lr_scale 0.1 --ninp 512 --nhid 2048 --num_workers 8
