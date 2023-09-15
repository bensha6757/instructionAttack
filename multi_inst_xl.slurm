#! /bin/sh

#SBATCH --job-name=multi_inst_xl
#SBATCH --output=./multi_inst_xl.out
#SBATCH --error=./multi_inst_xl.err
#SBATCH --partition=gpu-a100-killable
#SBATCH --account=gpu-research
#SBATCH --time=1400
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=50000
#SBATCH --gpus=1

export HF_HOME=/home/joberant/home/roi1/hf_cache
export TRANSFORMERS_CACHE=/home/joberant/home/roi1/hf_cache
export HF_DATASETS_CACHE=/home/joberant/home/roi1/datasets_cache

chmod +x /home/joberant/home/roi1/instructionAttack/run_experiment_multi_instructions_xl.sh
/home/joberant/home/roi1/instructionAttack/run_experiment_multi_instructions_xl.sh