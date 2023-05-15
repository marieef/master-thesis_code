# master-thesis_code
Code used in my master's thesis in Natural Language Processing

## Contents
### compare_to_norwegian_negex

Contains code for an approximated comparison of the models developed as part of the thesis to the [Norwegian NegEx](https://daisy.dsv.su.se/fil/visa?id=233579) (Sadhukhan, 2021).

### norec_neg_data

The NoReC_neg datasplit, which was used for modeling, is included. It is obtained from the [NoReC_neg](https://github.com/ltgoslo/norec_neg) repo, which is associated with the paper [Negation in Norwegian: an annotated dataset](https://aclanthology.org/2021.nodalida-main.30) (Mæhlum et al., NoDaLiDa 2021). Note that the dataset is distributed under a Creative Commons Attribution-NonCommercial licence (CC BY-NC 4.0)

The data is provided in the original JSON format, and in \*SEM format as used in [*SEM 2012 Shared Task: Resolving the Scope and Focus of Negation](https://aclanthology.org/S12-1035) (Morante & Blanco, SemEval-*SEM 2012). The latter is the format used as input for training and evaluation. 


### normed_neg_data
One part of the thesis was to reannotate an existing negation dataset of biomedical articles.

The original dataset is [The Norwegian GastroSurgery Biomedical Negation Corpus](https://github.com/DebaratiSJ/NegEx-on-Norwegian-biomedical-text/blob/main/Gold%20standard%20biomedical%20corpus/Norwegian%20GastroSurgery%20Biomedical%20Negation%20Corpus.txt). See [Building and evaluating the NegEx negation detection system for Norwegian biomedical text](https://daisy.dsv.su.se/fil/visa?id=233579) (Sadhukhan, 2021) for more info.

The dataset has been through some preprocessing steps and has been reannotated with negation according to the [NoReC_neg guidelines](https://github.com/ltgoslo/norec_neg/blob/main/annotation_guidelines/guidelines_neg.md), as used in [Negation in Norwegian: an annotated dataset](https://aclanthology.org/2021.nodalida-main.30) (Mæhlum et al., NoDaLiDa 2021).

The resulting dataset is named NorMed_neg. It was used in the experiments and is thus included here. See the [NorMed_neg repo](https://github.com/marieef/NorMed_neg/) for more details.

**NorMed_neg.json**: the dataset in JSON format.
 
**NorMed_neg.sem**: the dataset in *SEM format. 

#### datasplit
Contains the datasplit used for further fine-tuning of models on NorMed_neg.

### data_analysis
Contains scripts and notebooks for computing general and negation-oriented corpus statistics, as well as code for error analysis of system predictions.


### format_conversion
Contains code for conversion of JSON to *SEM format, extraction of affixes in cases predicted as affixal negation, and simplification of the gold standard to the word-level (as opposed to the original subword-level gold standard).


### experiments
Contains:
- [finetune_on_norec_neg](https://github.com/marieef/master-thesis_code/tree/main/experiments/finetune_on_norec_neg): code for the best-performing system from Chapter 3 of the thesis, using NoReC_neg for training and evaluation
- [further_finetuning_on_normed_neg](https://github.com/marieef/master-thesis_code/tree/main/experiments/further_finetuning_on_normed_neg): code for further fine-tuning of models on NorMed_neg, from Chapter 6 of the thesis

A description for how to run the code is provided in [how_to_run.md](https://github.com/marieef/master-thesis_code/blob/main/how_to_run.md).


### evaluation
Contains the script to evaluate the results according to the metrics used in *SEM 2012, see [*SEM 2012 Shared Task: Resolving the Scope and Focus of Negation](https://aclanthology.org/S12-1035) (Morante & Blanco, SemEval-*SEM 2012). 