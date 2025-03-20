#!/bin/bash
#SBATCH --job-name=train-job
#SBATCH --output=train-job2.out
#SBATCH --error=train-job2.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --partition=clara
#SBATCH --gres=gpu:1

source /work/hv49coni-data_gen/fame-private/sc_venv_template/activate.sh

python main.py

