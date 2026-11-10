#!/bin/bash
#BSUB -J smc[1-61]%30
#BSUB -q hpc
#BSUB -n 2
#BSUB -R "span[hosts=1]"
#BSUB -R "select[model == XeonGold6126]"
#BSUB -R "rusage[mem=2GB]"
#BSUB -W 00:10
#BSUB -o experiments/smc_experiment/out/smc_%J_%I.out
#BSUB -e experiments/smc_experiment/err/smc_%J_%I.err

export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYTHONUNBUFFERED=1

mkdir -p experiments/smc_experiment/out experiments/smc_experiment/err
source ~/EEG_localization_bayesian_machine_learning/sc_venv/bin/activate
python3 -u experiments/smc_experiment/smc_experiment.py $LSB_JOBINDEX