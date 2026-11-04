#!/bin/bash
#BSUB -J mapvshmc[1-214]%30
#BSUB -q hpc
#BSUB -n 2
#BSUB -R "span[hosts=1]"
#BSUB -R "select[model == XeonGold6126]"
#BSUB -R "rusage[mem=2GB]"
#BSUB -W 00:30
#BSUB -o experiments/map_vs_hmc_inverse/out/mapvshmc_%J_%I.out
#BSUB -e experiments/map_vs_hmc_inverse/err/mapvshmc_%J_%I.err

export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYTHONUNBUFFERED=1

source ~/EEG_localization_bayesian_machine_learning/sc_venv/bin/activate
python3 -u experiments/map_vs_hmc_inverse/map_vs_hmc_experiment.py $LSB_JOBINDEX