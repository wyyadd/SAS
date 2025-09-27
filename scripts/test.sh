#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=0-12:00:00
#SBATCH --mail-user=wyyadd@gmail.com
#SBATCH --mail-type=ALL

cd $project/SAS
module purge
module load python/3.12.4
source ../agents/bin/activate

srun python3 train_sas.py \
--mode="val" \
--root="$project/SAS/data" \
--num_workers=8 \
--accelerator="auto" \
--devices=-1 \
--num_nodes=$SLURM_NNODES \
--train_batch_size=16 \
--val_batch_size=16 \
--test_batch_size=16 \
--submission_dir="./data/pkl_files" \
--simulation_times=32 \
--ckpt_path=""
