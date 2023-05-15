#!/bin/bash 

#SBATCH --job-name=eval_nb_bert_large 
#SBATCH --account=nn9851k
#SBATCH --mail-type=FAIL
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=3G
#
# by default, request two cores (NumPy may just know how to take
# advantage of them; for larger computations, maybe use between
# six and ten; at some point, we will look at how to run on gpus
#
#SBATCH --cpus-per-task=2

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
python3 evaluate.py ${@}
