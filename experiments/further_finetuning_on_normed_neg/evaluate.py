from torch.utils.data import DataLoader
import torch
from tqdm import tqdm 
from collections import defaultdict
from metrics import *
from datasets import *
from datasets_extended import *
from utils import *
from model import *
from transformers import AutoTokenizer
from argparse import ArgumentParser
from datetime import datetime, date
import os
import re

def get_one_instance_per_negation(predictions, inv_label_mapper):
    '''
    NB: returns empty list for sentences without predicted cues
    '''
    # Example: If three cues are predicted in one sentence, create one instance for each of these 
    # predictions: the predictions for ONE sentence
    cue_indices = [(i, inv_label_mapper[predictions[i]]) for i in range(len(predictions)) if inv_label_mapper[predictions[i]] != "NOT_CUE"]
    # In theory, there could be som PAD predictions here, but these will just be ignored and not added to the returned instances
    
    instances = []
    c_multi = 0
    for idx, cue_type in cue_indices:
        # We say that all predicted MULTI in the sentence belong to the same cue
        if cue_type == "AFFIX" or cue_type == "NORMAL": 
            instances.append(["NOT_CUE" if i != idx else cue_type for i in range(len(predictions))])
        
        # We say that all predicted MULTI in the sentence belong to the same cue
        elif cue_type == "MULTI" and c_multi < 1: # This is done once, i.e. create an instance with all MULTI cues in the sentence 
            c_multi += 1
            multi_indices = [i for i, cue in cue_indices if cue == "MULTI"] 
            instances.append(["NOT_CUE" if i not in multi_indices else "MULTI" for i in range(len(predictions))])

    return instances 


def get_predictions(test_loader, tokenizer, device, model, args, model_type="cue"):
    with torch.no_grad():
        model.eval()
        data_with_pred_labels = []

        # For F1 computation:
        all_targets_test = []
        all_preds_test = []

        test_iter = tqdm(test_loader)
        for source, sentence, lengths, labels in test_iter: # batch size is 1
            encoding = tokenizer(sentence, return_tensors="pt", padding=True, is_split_into_words=True, return_offsets_mapping=True) # padding=True pads to the longest in _this_ batch
            mask = get_mask(encoding["offset_mapping"], lengths, encoding["input_ids"].size(1), labels.size(1))

            # To GPU
            del encoding['offset_mapping'] # This is unexpected as input to the model, so remove it
            encoding = {key: value.to(device) for key, value in encoding.items()}
            mask = mask.to(device)

            prediction = model(encoding, mask, lengths, args)

            if model_type == "cue":
                labels_flat = labels.flatten()
                preds_flat = prediction.argmax(-1).flatten()

                labels_flat = [l.item() for l in labels_flat if l!=4]
                preds_flat = [p.item() for p,l in zip(preds_flat, labels_flat) if l!=4]

                all_targets_test.append(labels_flat)
                all_preds_test.append(preds_flat)

            sent_pred_labels = None
            if model_type == "cue":
                sent_pred_labels = prediction.argmax(-1).tolist()[0] # -1 means argmax of the last dimension (corresponding to the 4 labels)
            else:
                # i.e. model_type = "scope"
                sigmoid = torch.nn.Sigmoid()
                sent_pred_labels = torch.squeeze(sigmoid(prediction) > 0.5).int().tolist()

            data_with_pred_labels.append((source[0], sentence[0], lengths[0], sent_pred_labels)) # tuple of sentence ID, sentence, lengths, predicted labels

        if model_type == "cue": 
            _,_,f1 = f1_cues([l_i for l in all_targets_test for l_i in l], [p_i for p in all_preds_test for p_i in p]) # checked that type is int in both lists
            print("CUE F1 ON DEV DATA:", f1)

    return data_with_pred_labels


def predict_on_test(cue_args, scope_args, c_model, s_model):
    X_test_cue = NegationCueDataset(cue_args.test_set) # One instance per sentence
    cue_test_loader = DataLoader(X_test_cue, batch_size=cue_args.test_batch_size, shuffle=False, collate_fn=collate_fn_cue)
    
    # Predict cues and use them as input to the scope detection model 
    cue_tokenizer = AutoTokenizer.from_pretrained(cue_args.model_path)
    all_pred_labels = get_predictions(cue_test_loader, cue_tokenizer, device, c_model, args=cue_args, model_type="cue") 
    print("\n20 first of all_pred_labels", all_pred_labels[:20])

    with open(cue_args.outfile_test, 'w', encoding="utf8") as f:
        for n, (source, sentence, lengths, pred_labels) in enumerate(all_pred_labels):
            cue_instances = get_one_instance_per_negation(pred_labels, X_test_cue.inv_label_mapper)

            # lines represents a sentence, one line per word
            lines = [f"_\t{source}\t{i}\t{sentence[i]}\t_\t_\t_" for i in range(len(sentence))]

            if len(cue_instances) == 0:
                # If no cues found in this sentence, add ***
                lines = [f"{lines[i]}\t***" for i in range(len(lines))]
            else:
                for c_i in range(len(cue_instances)):
                    for i in range(len(lines)):
                        if cue_instances[c_i][i] == "NORMAL":
                            lines[i] = f"{lines[i]}\t{sentence[i]}\t_\t_"
                        elif cue_instances[c_i][i] == "MULTI":
                            lines[i] = f"{lines[i]}\t{sentence[i]}\t_\t_" # treated the same way as 'NORMAL'
                        elif cue_instances[c_i][i] == "AFFIX":
                            lines[i] = f"{lines[i]}\t{sentence[i]}*\t_\t_" # add a * to the word to mark (for scope prediction) that this was predicted as an affix!
                        else:
                            lines[i] = f"{lines[i]}\t_\t_\t_"

            f.write("\n".join(lines) + "\n\n")

        #files.download(cue_args.outfile_test)
        print(f"Cue predictions written to {cue_args.outfile_test}.")            

    # Create scope dataset from the cue predictions file
    X_test_scope = NegationScopeDatasetForEvaluation(cue_args.outfile_test)
    scope_test_loader = DataLoader(X_test_scope, batch_size=scope_args.test_batch_size, shuffle=False, collate_fn=collate_fn_scope)

    # Scopes
    scope_tokenizer = AutoTokenizer.from_pretrained(scope_args.model_path)
    all_pred_labels = get_predictions(scope_test_loader, scope_tokenizer, device, s_model, args=scope_args, model_type="scope")
    
    # We sometimes have > 1 entry per sentence, so map source to entries:
    source_to_scope_preds = defaultdict(lambda: {}) 

    for source, sentence, lengths, pred_labels in all_pred_labels:
        source_to_scope_preds[source]["sentence"] = sentence
        source_to_scope_preds[source]["lengths"] = lengths
        
        if "pred_labels" not in source_to_scope_preds[source]:
            source_to_scope_preds[source]["pred_labels"] = []
        
        if isinstance(pred_labels, int):
            pred_labels = [pred_labels] # Added this because a one-word sentence caused problems (it was not wrapped in a list)
        source_to_scope_preds[source]["pred_labels"].append(pred_labels)

    with open(scope_args.outfile_test, 'w', encoding='utf8') as out_f, open(cue_args.outfile_test, encoding='utf8') as in_f:
        data = [item for item in in_f.read().split("\n\n") if item != '']
        sents = [[line.split("\t") for line in item.split("\n")] for item in data]

        # For those that have cues, make sure the * marking cues predicted as affixes is removed
        # Not doing this will lead to trouble when running the evaluation script 
        last_idx_no_neg = 7 # constant
        for i in range(len(sents)):
            cue_indices = [last_idx_no_neg + n*3 for n in range(len(sents[i][0])) if last_idx_no_neg + n*3 < len(sents[i][0])]
            for j in range(len(sents[i])):
                for k in cue_indices:
                    if sents[i][j][k] != "_" and sents[i][j][k] != "***":
                        # This is to replace affix cues marked by *
                        # we can do this because this model always puts the label on a whole word
                        sents[i][j][k] = sents[i][j][3] 

        for i in range(len(sents)): # i.e. for each sentence
            source = sents[i][0][1]
            if not source in source_to_scope_preds:
                # No cue prediction for this one, so it was not included in the NegationScopeDataset
                continue 
            pred_scopes = source_to_scope_preds[source]["pred_labels"]
            sentence = source_to_scope_preds[source]["sentence"]

            # The order in pred_scopes should be correct,i.e. the scope is connected to the correct cue (in case of multiple cues in sentence)
            # This must be true because of the ordering of the instances in NegationScopeDataset, where an instance is created for the cue on the 
            # first cue index of the sentence, then an instance for the second cue index of the sentence, and so on...
            for s_i in range(len(pred_scopes)):
                scope_index = last_idx_no_neg + s_i*3+1
                for j in range(len(sents[i])): # for each word in sent
                    if pred_scopes[s_i][j] == 1:

                        # This line would crash if we tried to set a scope label for a word 
                        # with no cue predicted, i.e. "***" on index 7 (last index) of the line 
                        sents[i][j][scope_index] = sents[i][j][3] # fill in word as scope word

                    # if not part of scope, do nothing 
                        
        for i in range(len(sents)):
            lines = []

            for j in range(len(sents[i])):
                lines.append("\t".join(sents[i][j]))

            out_f.write("\n".join(lines) + "\n\n")

        # This file contains both cue and scope predictions and can be used as input
        # to the starsem 2012 evaluation script
        # files.download(scope_args.outfile_test)
        print(f"Scope predictions written to {scope_args.outfile_test}.")

class TestArgs:
    def __init__(self, outfile_test, test_set, mtype, hidden_size, bidirectional, model_path, architecture, recurrent_layers, test_batch_size):
        self.outfile_test = outfile_test
        self.test_set = test_set
        self.mtype = mtype
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.model_path = model_path
        self.architecture = architecture
        self.recurrent_layers = recurrent_layers
        self.test_batch_size = test_batch_size


if __name__ == "__main__":
    # Create a folder from the timestamp that pred files will be put in
    parser = ArgumentParser()
    parser.add_argument("--test_batch_size", "-tbs", type=int, default=1) 
    parser.add_argument("--models_dir", "-mdir", type=str, required=True, help="Path to dir where both models are stored")
    parser.add_argument("--cue_model_path", "-cmp", type=str, required=True) 
    parser.add_argument("--scope_model_path", "-smp", type=str, required=True)  
    parser.add_argument("--test_set", "-test", type=str, required=True) # full path
    parser.add_argument("--mtype_cue", "-mtc", type=str, default=None)
    parser.add_argument("--mtype_scope", "-mts", type=str, default=None)
    parser.add_argument("--hidden_size_cue", "-hsc", type=int, default=None)
    parser.add_argument("--hidden_size_scope", "-hss", type=int, default=None)
    parser.add_argument("--bidirectional_cue", "-bidirc", type=int, default=None)
    parser.add_argument("--bidirectional_scope", "-bidirs", type=int, default=None)
    parser.add_argument("--model_path_cue", "-mpc", type=str, required=True) # The language model used 
    parser.add_argument("--model_path_scope", "-mps", type=str, required=True) # The language model used 
    parser.add_argument("--architecture_cue", "-archc", type=str, required=True)
    parser.add_argument("--architecture_scope", "-archs", type=str, required=True)
    parser.add_argument("--recurrent_layers_cue", "-rlc", type=int, required=True)
    parser.add_argument("--recurrent_layers_scope", "-rls", type=int, required=True)
    parser.add_argument("--cue_outfile_test", "-oftc", type=str, default=None)
    parser.add_argument("--scope_outfile_test", "-ofts", type=str, default=None)
    parser.add_argument("--output_dir", "-odir", type=str, required=True, help="Dir to output predictions to")


    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    time_str = str(datetime.now().date()) + "_" + "".join(re.split(r'[:\.]', str(datetime.now().time())))

    # args.cue_model_path[-7:-3] is the seed - need it to be sure directory name is unique
    dir_path = os.path.join(args.output_dir, f'{time_str}_{args.cue_model_path[-7:-3]}') 

    os.system(f"mkdir {dir_path}")

    cue_outfile_test = args.cue_outfile_test if args.cue_outfile_test else os.path.join(dir_path, f'predictions_cue_{date.today()}.sem')
    scope_outfile_test = args.scope_outfile_test if args.scope_outfile_test else os.path.join(dir_path, f'predictions_scope_{date.today()}.sem')

    # Ideally, these params should be read from a stored config file
    cue_args = TestArgs(outfile_test=cue_outfile_test, test_set=args.test_set,
        mtype=args.mtype_cue, hidden_size=args.hidden_size_cue,
        bidirectional=args.bidirectional_cue, model_path=args.model_path_cue,
        architecture=args.architecture_cue, recurrent_layers=args.recurrent_layers_cue,
        test_batch_size=args.test_batch_size)
    scope_args = TestArgs(outfile_test=scope_outfile_test, test_set=args.test_set,
        mtype=args.mtype_scope, hidden_size=args.hidden_size_scope,
        bidirectional=args.bidirectional_scope, model_path=args.model_path_scope,
        architecture=args.architecture_scope, recurrent_layers=args.recurrent_layers_scope,
        test_batch_size=args.test_batch_size)

    c_full_path = os.path.join(args.models_dir, f"{args.cue_model_path}")
    s_full_path = os.path.join(args.models_dir, f"{args.scope_model_path}")
    
    # Create readme to remember what models were tested
    with open(os.path.join(dir_path, "readme.txt"), 'w', encoding='utf8') as f:
        f.write("Predictions for system consisting of:\n")
        f.write(f"CUE-model: {c_full_path}\n")
        f.write(f"SCOPE-model: {s_full_path}\n")

    # Cue model
    vocab = ["AFFIX", "NORMAL", "MULTI", "NOT_CUE", "PAD"]

    cue_model = GeneralNegationModel(len(vocab), cue_args)
    cue_model.load_state_dict(torch.load(c_full_path, map_location=torch.device('cpu')))

    # Scope model
    scope_model = GeneralNegationModel(2, scope_args)
    scope_model.load_state_dict(torch.load(s_full_path, map_location=torch.device('cpu')))

    predict_on_test(cue_args, scope_args, cue_model, scope_model)
