#!/bin/bash


python train.py --gpu_id 0,1 --kmer_aggregation --nmute 40 --epochs 60 --nlayers 3 \
--batch_size 512 --kmers 7 --lr_scale 1 --ninp 1024 --nhid 4096 --num_workers 32 \
--nclass 919 --nhead 16 --weight_decay 1e-6 --dropout 0.2
