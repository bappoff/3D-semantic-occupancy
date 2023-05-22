#!/usr/bin/env bash

#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 20
#SBATCH --begin=now
#SBATCH --mem 40G
#SBATCH --partition gpu
#SBATCH --gres gpu:1

CONFIG=$1
GPUS=$2
PORT=${PORT:-28509}

PYTHONPATH="/home/bpoffet/miniconda3/envs/voxformer_venv/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    /work/scitas-share/voxformer/VoxFormer/scripts/train.py $CONFIG --launcher pytorch ${@:3} --deterministic
