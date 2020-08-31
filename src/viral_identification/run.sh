#!/bin/bash
python train.py --gpu_id 0,1 --kmer_aggregation --nmute 60 --epochs 100 --nlayers 1 \
--batch_size 128 --kmers 11 --lr_scale 0.1 --path ../..
