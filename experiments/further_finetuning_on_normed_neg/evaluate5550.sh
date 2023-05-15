#!/bin/sh

MODELS_DIR='PATH_TO_DIR' # an existing dir in which the models in cmp and smp (below) are located 

# Paths to saved cue- and scope models, replace with your paths.
# Evaluate models finetuned with seed 5550, finetuned on 10, 40 and 100 % of NorMed_neg training set, respectively
cmp=('FINETUNE_nbbert-large_5550_CUE/cuemodel_2023-04-12_warmup200_ep20_linear_lr3e-05_bs16_dl1_nb-bert-large_GeneralNegationModel_SEED_5550_10.pt' 'FINETUNE_nbbert-large_5550_CUE/cuemodel_2023-04-12_warmup200_ep20_linear_lr3e-05_bs16_dl1_nb-bert-large_GeneralNegationModel_SEED_5550_40.pt' 'FINETUNE_nbbert-large_5550_CUE/cuemodel_2023-04-12_warmup200_ep20_linear_lr3e-05_bs16_dl1_nb-bert-large_GeneralNegationModel_SEED_5550_100.pt')
smp=('FINETUNE_nbbert-large_5550_SCOPE/scopemodel_2023-04-12_warmup200_ep20_linear_lr3e-05_bs16_dl1_nbbert-large_GeneralNegationModel_lmean-auto_SEED_5550_10.pt' 'FINETUNE_nbbert-large_5550_SCOPE/scopemodel_2023-04-12_warmup200_ep20_linear_lr3e-05_bs16_dl1_nbbert-large_GeneralNegationModel_lmean-auto_SEED_5550_40.pt' 'FINETUNE_nbbert-large_5550_SCOPE/scopemodel_2023-04-12_warmup200_ep20_linear_lr3e-05_bs16_dl1_nbbert-large_GeneralNegationModel_lmean-auto_SEED_5550_100.pt')

OUTPUT_DIR='PATH_TO_OUTPUT_DIR' # existing dir 
LEN=${#cmp[@]}


# TESTING ON THE HELD-OUT TEST SET:
for ((i=0;i<LEN;i++)); do
	sbatch evaluate.slurm --models_dir $MODELS_DIR --output_dir $OUTPUT_DIR --cue_model_path ${cmp[$i]} --scope_model_path ${smp[$i]} --test_set ../../normed_neg_data/datasplit/NorMed_neg_test.sem --model_path_cue NbAiLab/nb-bert-large --model_path_scope NbAiLab/nb-bert-large --architecture_cue linear_only --architecture_scope linear_only --recurrent_layers_cue 0 --recurrent_layers_scope 0
done




