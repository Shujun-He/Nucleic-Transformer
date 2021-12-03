#!/bin/bash
for i in {0..4};do
python train.py --fold $i --gpu_id 0 --kmer_aggregation --epochs 50 \
--nlayers 6 --nmute 45 --path ../../data/human_non_tata_dataset.csv --kmers 7 --ninp 256 --nhid 1024 \
--batch_size 64
done

python evaluate.py --gpu_id 0 --kmer_aggregation --epochs 150 \
--nlayers 6 --nmute 15 --kmers 7 --ninp 256 --nhid 1024 \
--path ../../data/human_non_tata_dataset.csv
