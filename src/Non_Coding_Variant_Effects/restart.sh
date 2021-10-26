#!/bin/bash


python restart.py --gpu_id 0,1 --kmer_aggregation --nmute 40 --epochs 60 --nlayers 3 \
--batch_size 256 --kmers 13 --lr_scale 0.1 --ninp 512 --nhid 2048 --num_workers 32 \
--nclass 919 --nhead 8 --restart_epoch 20
