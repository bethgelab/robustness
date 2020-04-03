#!/bin/sh
#SBATCH --job-name=robustness
#SBATCH --output=logs/logs-%A_%a.out  # Standard output and error log
#SBATCH --partition=gpu-2080ti-preemptable
#SBATCH --gres=gpu:1

scontrol show job "$SLURM_JOB_ID"

# The image georgepachitariu/robustness was created using 
# the Dockerfile from parent folder.
row="2" # This is the row in the table
singularity exec --nv -B /scratch_local \
    -B "$IMAGENET_C_PATH":/ImageNet-C:ro \
    -B "$CHECKPOINT_PATH":/checkpoints:ro \
    -B .:/batchnorm \
    -B ..:/deps \
    docker://georgepachitariu/robustness:latest \
    bash /batchnorm/scripts/paper/table1.sh $row 2>&1
