#!/bin/bash -l
#SBATCH --job-name=imgGen
#SBATCH --output=imgGen.out
#SBATCH --error=imgGen.err
#SBATCH --partition=testing
#SBATCH --account=thesis
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --exclude=vgpu11

### Some useful informative commands
echo -n 'Job started at: '
TZ="Europe/Rome" date
echo -n 'Directory: '
pwd
echo -n 'This job will be executed on th following nodes: '
echo ${SLURM_NODELIST}
echo
SHELL=/bin/bash
conda activate gpuenv
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6
python -u ImgGen.py
echo -n 'Job finished at: '
TZ="Europe/Rome" date