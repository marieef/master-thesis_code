#!/bin/bash

#SBATCH --account=nn9851k
#SBATCH --mem-per-cpu=3G
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --partition=accel
#SBATCH --mail-type=FAIL
#SBATCH --time=00:45:00

# NB: I used the same template for my slurm files as was done in the IN5550 course at 
# The University of Oslo, spring 2022. 

source ${HOME}/.bashrc

# sanity: exit on all errors and disallow unset environment variables
set -o errexit
set -o nounset

# the important bit: unload all current modules (just in case) and load only the necessary ones
module purge
module use -a /cluster/shared/nlpl/software/eb/etc/all/
module load nlpl-nlptools/2021.01-gomkl-2019b-Python-3.7.4
module load nlpl-pytorch/1.7.1-gomkl-2019b-cuda-10.1.243-Python-3.7.4
module load nlpl-transformers/4.14.1-gomkl-2019b-Python-3.7.4

# print information (optional)
echo "submission directory: ${SUBMITDIR}"
#ulimit -a
#module list

# by default, pass on any remaining command-line options
echo ${@}
python3 train_scopemodel.py ${@}
