for i in {0..0};do
python pretrain.py --gpu_id 0,1 --kmer_aggregation --nmute 0 --epochs 200 --nlayers 5 \
--batch_size 96 --kmers 5 --lr_scale 0.1 --path ../../../data --workers 2 \
--dropout 0.1 --nclass 5 --ntoken 15 --nhead 32 --ninp 256 --nhid 1024 --warmup_steps 600 \
--fold $i --weight_decay 0.1
done