#!/bin/bash
#SBATCH --job-name=setup
#SBATCH --account=EUHPC_B25_058
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --mem-per-cpu=5000M
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=aleksa.jelaca@student.kuleuven.be
#SBATCH --output=alignedSft.out
#SBATCH --error=alignedSft.err
module load cuda/12.1
module load gcc/12.2.0
source /leonardo_work/EUHPC_B25_058/aleksa_files/downloaded_work/aj/bin/activate
srun /leonardo_work/EUHPC_B25_058/aleksa_files/downloaded_work/thesis/gpt/bash_scripts/gpt_train_aligned.sh

