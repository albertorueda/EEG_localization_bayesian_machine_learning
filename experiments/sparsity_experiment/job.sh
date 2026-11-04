#!/bin/bash
#BSUB -J sparsity[1-300]%30
#BSUB -q hpc
#BSUB -n 2
#BSUB -R "span[hosts=1]"
#BSUB -R "select[model == XeonGold6126]"
#BSUB -R "rusage[mem=1GB]"
#BSUB -W 00:30
#BSUB -o out/sparsity_%J_%I.out
#BSUB -e err/sparsity_%J_%I.err

export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYTHONUNBUFFERED=1

source ~/special_course/venv/bin/activate
python3 -u sparsity_experiment.py $LSB_JOBINDEX
