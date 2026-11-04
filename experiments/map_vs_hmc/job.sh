#!/bin/bash
#BSUB -J mapvshmc[1-213]%30
#BSUB -q hpc
#BSUB -n 2
#BSUB -R "span[hosts=1]"
#BSUB -R "select[model == XeonGold6126]"
#BSUB -R "rusage[mem=2GB]"
#BSUB -W 02:00
#BSUB -o out/mapvshmc_%J_%I.out
#BSUB -e err/mapvshmc_%J_%I.err

export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYTHONUNBUFFERED=1

source ~/special_course/venv/bin/activate
python3 -u map_vs_hmc_experiment.py $LSB_JOBINDEX