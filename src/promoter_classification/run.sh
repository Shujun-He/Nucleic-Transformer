#!/bin/bash
for i in {0..4};do
python train.py --fold $i --gpu_id 0 --kmer_aggregation --epochs 150 --nlayers 6 --nmute 15 --path v9d3.csv --kmers 7 --ninp 256 --nhid 1024
done
