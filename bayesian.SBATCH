#!/bin/bash
#SBATCH--nodes=1
#SBATCH--ntasks-per-node=1
#SBATCH--cpus-per-task=1
#SBATCH--time=160:00:00
#SBATCH--mem=128GB
#SBATCH--gres=gpu

module purge

SEED=$1

singularity exec --nv\
    --overlay /scratch/jl14310/nanoGPTenv/nanoGPT.ext3:ro \
    /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif\
    /bin/bash -c "source /ext3/env.sh; \
   echo 'hello';\
   cd nanoGPT;\
     python smacc.py --seed=$SEED"
