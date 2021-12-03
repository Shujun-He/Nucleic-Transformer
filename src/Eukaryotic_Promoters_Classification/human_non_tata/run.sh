#!/bin/bash
for i in {0..4};do
python train.py --fold $i --gpu_id 0 --kmer_aggregation --epochs 150 \
--nlayers 6 --nmute 45 --path ../../data/human_non_tata_dataset.csv --kmers 11 --ninp 256 --nhid 1024 \
--batch_size 64 --lr_scale 0.2
done

python evaluate.py --gpu_id 0 --kmer_aggregation --epochs 150 \
--nlayers 6 --nmute 15 --kmers 11 --ninp 256 --nhid 1024 \
--path ../../data/human_non_tata_dataset.csv
