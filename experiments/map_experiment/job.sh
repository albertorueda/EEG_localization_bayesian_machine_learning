#!/bin/bash
# 61 sizes x 10 seeds (lib/seeds.py: N_SEEDS) = 610 jobs; job i -> size_idx, seed = divmod(i-1, 10)
#BSUB -J map[1-610]%30
#BSUB -q hpc
#BSUB -n 2
#BSUB -R "span[hosts=1]"
#BSUB -R "select[model == XeonGold6126]"
#BSUB -R "rusage[mem=1GB]"
#BSUB -W 00:30
#BSUB -o experiments/map_experiment/out/mapvshmc_%J_%I.out
#BSUB -e experiments/map_experiment/err/mapvshmc_%J_%I.err

export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYTHONUNBUFFERED=1

mkdir -p experiments/map_experiment/out experiments/map_experiment/err
source ~/EEG_localization_bayesian_machine_learning/sc_venv/bin/activate
python3 -u experiments/map_experiment/map_experiment.py $LSB_JOBINDEX