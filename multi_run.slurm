#! /bin/sh

#SBATCH --job-name=inst_attack
#SBATCH --output=./inst_attack.out
#SBATCH --error=./inst_attack.err
#SBATCH --partition=killable
#SBATCH --time=1400
#SBATCH --signal=USR1@120 # how to end job when time’s up
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=50000
#SBATCH --gpus=1

export HF_HOME=/home/joberant/home/roi1/hf_cache
export TRANSFORMERS_CACHE=/home/joberant/home/roi1/hf_cache
export HF_DATASETS_CACHE=/home/joberant/home/roi1/datasets_cache

python run_experiments.py