#! /bin/sh

#SBATCH --job-name=multi_inst_base
#SBATCH --output=./multi_inst_base.out
#SBATCH --error=./multi_inst_base.err
#SBATCH --partition=gpu-a100-killable
#SBATCH --time=1400
#SBATCH --signal=USR1@120 # how to end job when time’s up
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=50000
#SBATCH --gpus=1

export HF_HOME=/home/joberant/home/roi1/hf_cache
export TRANSFORMERS_CACHE=/home/joberant/home/roi1/hf_cache
export HF_DATASETS_CACHE=/home/joberant/home/roi1/datasets_cache

chmod +x /home/joberant/home/roi1/instructionAttack/run_experiment_multi_instructions_base.sh
/home/joberant/home/roi1/instructionAttack/run_experiment_multi_instructions_base.sh