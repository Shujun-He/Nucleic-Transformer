#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=JobExample5       #Set the job name to "JobExample4"
#SBATCH --time=02:30:00              #Set the wall clock limit to 1hr and 30min
#SBATCH --ntasks=1                   #Request 1 task
#SBATCH --mem=5120M                  #Request 2560MB (2.5GB) per node
#SBATCH --output=out      #Send stdout/err to "Example4Out.[jobID]"
#SBATCH --gres=gpu:rtx:1             #Request 1 "rtx" GPU per node
#SBATCH --partition=gpu              #Request the GPU partition/queue


##OPTIONAL JOB SPECIFICATIONS
#SBATCH --account=132825315633
#SBATCH --mail-type=ALL              #Send email on all job events
#SBATCH --mail-user=shujun@tamu.edu    #Send all emails to email_address

#First Executable Line
#cd $SCRATCH
cd /scratch/user/shujun/Nucleic-Transformer/src/promoter_classification_v9d4
#module load Anaconda3
#source /scratch/user/shujun/.conda/envs/torch/bin/activate
#./run.sh

for i in {0..4};do
  /scratch/user/shujun/.conda/envs/torch/bin/python train.py --fold $i --kmer_aggregation --epochs 150 --nlayers 6 --nmute 15 --path v9d4.csv --kmers 7 --ninp 256 --nhid 1024
done
