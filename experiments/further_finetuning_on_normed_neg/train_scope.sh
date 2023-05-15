#!/bin/sh

SEEDS=(3456) #(1234 2345 3456 2022 5550) # must be run once per seed 
EPOCHS=20
DIR_PATH_PREFIX='NAME_OF_DIR' # must exist (for output)
DIR_NAME='FINETUNE_nbbert-large_3456_SCOPE' # must exist inside DIR_PATH_PREFIX
BATCH_SIZE=16
TRAIN_SETS=('../../normed_neg_data/datasplit/NorMed_neg_train_10percent.sem' '../../normed_neg_data/datasplit/NorMed_neg_train_40percent.sem' '../../normed_neg_data/datasplit/NorMed_neg_train.sem')
TRAINED_MODEL_PATH='path/to/model/finetuned/on/norec_neg' # full path

for seed in ${SEEDS[@]} ; do
	for train_set in ${TRAIN_SETS[@]} ; do
		sbatch train_scope.slurm --dir_path_prefix $DIR_PATH_PREFIX --batch_size $BATCH_SIZE --epochs $EPOCHS --initial_lr 0.00003 --warmup_steps 200 --model_path NbAiLab/nb-bert-large --train_set $train_set --dev_set ../../normed_neg_data/datasplit/NorMed_neg_dev.sem --dir_name $DIR_NAME --seed $seed --loss_mean auto --trained_model_path $TRAINED_MODEL_PATH
	done
done
