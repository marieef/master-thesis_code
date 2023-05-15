#!/bin/sh

SEEDS=(1234 2345 3456 2022 5550)
EPOCHS=20
DIR_PATH_PREFIX='NAME_OF_DIR' # must exist
DIR_NAME='EXPERIMENTS_NBBERT-LARGE_BS16' # must exist inside DIR_PATH_PREFIX
BATCH_SIZE=16

for seed in ${SEEDS[@]} ; do
	sbatch train_cue.slurm --dir_path_prefix $DIR_PATH_PREFIX --batch_size $BATCH_SIZE --epochs $EPOCHS --initial_lr 0.00003 --warmup_steps 200 --model_path NbAiLab/nb-bert-large --train_set ../../norec_neg_data/sem/negation_train.sem --dev_set ../../norec_neg_data/sem/negation_dev.sem --dir_name $DIR_NAME --seed $seed

done
