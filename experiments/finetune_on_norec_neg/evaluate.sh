#!/bin/sh

MODELS_DIR='PATH_TO_DIR' # an existing dir in which the models in cmp and smp (below) are located 

# Paths to saved cue- and scope models, replace with your paths.
cmp=('EXPERIMENTS_NBBERT-LARGE_BS16/cuemodel_2022-11-28_warmup200_ep20_linear_lr3e-05_bs16_dl1_nb-bert-large_GeneralNegationModel_SEED_1234.pt' 'EXPERIMENTS_NBBERT-LARGE_BS16/cuemodel_2022-11-28_warmup200_ep20_linear_lr3e-05_bs16_dl1_nb-bert-large_GeneralNegationModel_SEED_2345.pt' 'EXPERIMENTS_NBBERT-LARGE_BS16/cuemodel_2022-11-28_warmup200_ep20_linear_lr3e-05_bs16_dl1_nb-bert-large_GeneralNegationModel_SEED_3456.pt' 'EXPERIMENTS_NBBERT-LARGE_BS16/cuemodel_2022-11-28_warmup200_ep20_linear_lr3e-05_bs16_dl1_nb-bert-large_GeneralNegationModel_SEED_2022.pt' 'EXPERIMENTS_NBBERT-LARGE_BS16/cuemodel_2022-11-28_warmup200_ep20_linear_lr3e-05_bs16_dl1_nb-bert-large_GeneralNegationModel_SEED_5550.pt')
smp=('EXPERIMENTS_NBBERT-LARGE_BS16/scopemodel_2022-11-28_warmup200_ep20_linear_lr3e-05_bs16_dl1_nb-bert-large_GeneralNegationModel_lmean-auto_SEED_1234.pt' 'EXPERIMENTS_NBBERT-LARGE_BS16/scopemodel_2022-11-28_warmup200_ep20_linear_lr3e-05_bs16_dl1_nb-bert-large_GeneralNegationModel_lmean-auto_SEED_2345.pt' 'EXPERIMENTS_NBBERT-LARGE_BS16/scopemodel_2022-11-28_warmup200_ep20_linear_lr3e-05_bs16_dl1_nb-bert-large_GeneralNegationModel_lmean-auto_SEED_3456.pt' 'EXPERIMENTS_NBBERT-LARGE_BS16/scopemodel_2022-11-28_warmup200_ep20_linear_lr3e-05_bs16_dl1_nb-bert-large_GeneralNegationModel_lmean-auto_SEED_2022.pt' 'EXPERIMENTS_NBBERT-LARGE_BS16/scopemodel_2022-11-28_warmup200_ep20_linear_lr3e-05_bs16_dl1_nb-bert-large_GeneralNegationModel_lmean-auto_SEED_5550.pt')
LEN=${#cmp[@]}
OUTPUT_DIR='PATH_TO_OUTPUT_DIR'


# Run evaluation script for each pair of cue- and scope model
for ((i=0;i<LEN;i++)); do
	sbatch evaluate.slurm --models_dir $MODELS_DIR --output_dir $OUTPUT_DIR --cue_model_path ${cmp[$i]} --scope_model_path ${smp[$i]} --test_set ../../norec_neg_data/sem/negation_dev.sem --model_path_cue NbAiLab/nb-bert-large --model_path_scope NbAiLab/nb-bert-large --architecture_cue linear_only --architecture_scope linear_only --recurrent_layers_cue 0 --recurrent_layers_scope 0
done


