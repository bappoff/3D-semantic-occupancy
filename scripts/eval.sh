#!/bin/bash

#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 20
#SBATCH --begin=now
#SBATCH --mem 100G
#SBATCH --partition gpu
#SBATCH --gres gpu:1

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
PORT=${PORT:-29503}

PYTHONPATH="/home/bpoffet/miniconda3/envs/voxformer_venv/..":$PYTHONPATH \
python -m torch.distributed.run --nproc_per_node=$GPUS --master_port=$PORT \
    /work/scitas-share/voxformer/VoxFormer/3D-semantic-occupancy/scripts/eval.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4} --eval bbox --out test3.pkl
