# How to run
This repo includes code for negation models finetuned on NoReC_neg (experiments/finetune_on_norec_neg), and code for finetuning these models further on NorMed_neg (experiments/further_finetuning_on_normed_neg). 
The files in these two directories have the same names, and the code is the same except for some small differences in the training and evaluation scripts.

Training and evaluation was conducted on [Saga](https://documentation.sigma2.no/hpc_machines/saga.html). 

## Finetune on NoReC_neg


    bash train_cue.sh       # train cue detection model
    bash train_scope.sh     # train scope resolution model
    bash evaluate.sh        # evaluate negation resolution system

The arguments used are hardcoded into these scripts.
Names of pre-existing output directories will need to be provided in train_cue.sh and train_scope.sh.
In evaluate.sh, paths to the models to test must be provided.


To train the models using the python scripts directly (finetune_on_norec_neg):

    python train_cuemodel.py --batch_size 16 --epochs 20 --initial_lr 0.00003 --warmup_steps 200 --model_path NbAiLab/nb-bert-large --train_set ../../norec_neg_data/sem/negation_train.sem --dev_set ../../norec_neg_data/sem/negation_dev.sem --dir_path_prefix path/to/existing/outputdir --dir_name existing/subdir/in/dir_path/prefix --seed your_seed_here

    python train_scopemodel.py --batch_size 16 --epochs 20 --initial_lr 0.00003 --warmup_steps 200 --model_path NbAiLab/nb-bert-large --train_set ../../norec_neg_data/sem/negation_train.sem --dev_set ../../norec_neg_data/sem/negation_dev.sem --dir_path_prefix path/to/existing/outputdir --dir_name existing/subdir/in/dir_path/prefix --seed your_seed_here --loss_mean auto

    python evaluate.py --models_dir path/to/dir/where/cue/and/scope/models/are/stored --cue_model_path path/to/saved/cuemodel/inside/models_dir --scope_model_path path/to/saved/scopemodel/inside/models_dir --test_set path/to/testset --model_path_cue NbAiLab/nb-bert-large --model_path_scope NbAiLab/nb-bert-large --architecture_cue linear_only --architecture_scope linear_only --recurrent_layers_cue 0 --recurrent_layers_scope 0 --output_dir path/to/existing/output/dir

## Further finetuning on NorMed_neg 
As for the finetuning on NoReC_neg, provide the necessary parameters inside the bash files and run as a batch job: 

    bash train_cue.sh       # train cue detection model
    bash train_scope.sh     # train scope resolution model
    bash evaluate.sh        # evaluate negation resolution system

or run the python scripts: 

    python train_cuemodel.py --batch_size 16 --epochs 20 --initial_lr 0.00003 --warmup_steps 200 --model_path NbAiLab/nb-bert-large --train_set ../../normed_neg_data/datasplit/NorMed_neg_train.sem --dev_set ../../normed_neg_data/datasplit/NorMed_neg_dev.sem --dir_path_prefix path/to/existing/outputdir --dir_name existing/subdir/in/dir_path/prefix --seed your_seed_here --trained_model_path path/to/finetuned/model

    python train_scopemodel.py --batch_size 16 --epochs 20 --initial_lr 0.00003 --warmup_steps 200 --model_path NbAiLab/nb-bert-large --train_set ../../normed_neg_data/datasplit/NorMed_neg_train.sem --dev_set ../../normed_neg_data/datasplit/NorMed_neg_dev.sem --dir_path_prefix path/to/existing/outputdir --dir_name existing/subdir/in/dir_path/prefix --seed your_seed_here --loss_mean auto --trained_model_path path/to/finetuned/model

    python evaluate.py --models_dir path/to/dir/where/cue/and/scope/models/are/stored --cue_model_path path/to/saved/cuemodel/inside/models_dir --scope_model_path path/to/saved/scopemodel/inside/models_dir --test_set ../../normed_neg_data/dataplit/NorMed_neg_test.sem --model_path_cue NbAiLab/nb-bert-large --model_path_scope NbAiLab/nb-bert-large --architecture_cue linear_only --architecture_scope linear_only --recurrent_layers_cue 0 --recurrent_layers_scope 0 --output_dir path/to/existing/output/dir


## General
evaluate.py outputs two .sem-files, one with cue predictions (predictions_cue_\*date_today\*.sem), and one with cue *and* scope predictions (predictions_scope_\*date_today\*.sem). 
These can be used for affix extraction (see format_conversion) to produce a prediction file where affixes are extracted from cues predicted as affixal (known as 'Original+RE' evaluation in the thesis). 

For the 'Original' and 'Adjusted' evaluation methods use predictions_scope_\*date_today\*.sem and the original or the simplified gold standard, respectively. 


To evaluate the models according to *SEM 2012, use the evaluation script in the evaluation directory:

    perl eval.sd-sco.pl -g gold_standard_sem_file -s system_predictions_sem_file

## Notes

**Note 1**: 5 different seeds were used for all modeling experiments: 1234, 2345, 3456, 2022, 5550.

**Note 2**: The code has been adapted so that it should no longer be dependent on my hierarchy of directories on Saga. If you get an error message while running, I might have made a little mistake here. 

**Note 3**: It might be confusing that arguments such as architecture_(cue/scope) and recurrent_layers_(cue/scope) are included. This is because I at one point considered experimenting with RNNs/LSTMs etc., but I did not use these options for the actual experiments. 

**Note 4**: evaluation.py code assumes that the names of the saved models end with SEED + ".pt".
