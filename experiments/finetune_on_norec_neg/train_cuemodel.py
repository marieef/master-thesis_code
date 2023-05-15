"""
The training loop is based on code provided as part of the IN5550 course ('Neural Methods in Natural Language Processing') at the University of Oslo,
spring 2022. This is the link to the course web page: https://www.uio.no/studier/emner/matnat/ifi/IN5550/v22/index.html

"""

import torch
from transformers import AutoTokenizer
from tqdm import tqdm
import logging
from argparse import ArgumentParser
import transformers
from datasets import *
from model import *
from torch.utils.data import DataLoader
from utils import *
from earlystopping import *
from metrics import *
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from datetime import date


def train(model, device, args, optimizer, scheduler, criterion, train_loader, dev_loader, label_mapper, ignore_idx, path=None):
    train_accuracy_scores = []
    global_train_losses = []
    validation_accuracy_scores = []
    global_validation_losses = []

    global_train_f1_scores = []
    global_validation_f1_scores = []
    stopped_early_at_epoch = None

    not_cue_index, ignore_index = label_mapper['NOT_CUE'], ignore_idx
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    early_stopping = None
    if args.early_stop==1:
        early_stopping = EarlyStopping(patience=args.es_patience, verbose=True, save_path=path) if path else EarlyStopping(patience=args.es_patience, verbose=True)

    for i in tqdm(range(args.epochs)):
        # Store loss and f1 for each batch in epoch
        train_losses = []
        validation_losses = []

        all_targets_train = []
        all_preds_train = []
        all_targets_dev = []
        all_preds_dev = []

        model.train()
        logger.info("Epoch {} started".format(i+1))
        print("Epoch {} started".format(i+1))
        print("\tTRAIN")

        train_iter = tqdm(train_loader)
        n_correct, n_total = 0, 0
        for _, sentences, lengths, labels in train_iter:
            optimizer.zero_grad()
    
            # Does not seem necessary to set truncation=True (if so, we would need to specify the length to truncate to)
            encoding = tokenizer(sentences, return_tensors="pt", padding=True, is_split_into_words=True, return_offsets_mapping=True) # padding=True pads to the longest in _this_ batch
            mask = get_mask(encoding["offset_mapping"], lengths, encoding["input_ids"].size(1), labels.size(1))

            # To GPU
            del encoding['offset_mapping'] # This is unexpected as input to the model, so remove it
            encoding = {key: value.to(device) for key, value in encoding.items()}
            mask = mask.to(device)
            labels = labels.to(device)

            prediction = model(encoding, mask, lengths, args)
            pred_labels = prediction.argmax(-1)

            n_correct += ((pred_labels == labels) & (labels != not_cue_index) & (labels != ignore_index)).sum()
            n_total += ((labels != not_cue_index) & (labels != ignore_index)).sum()

            loss = criterion(prediction.flatten(0, 1), labels.flatten())

            loss.backward()
            optimizer.step()
            scheduler.step()

            train_losses.append(loss.item())
            train_iter.set_postfix_str(f"loss: {loss.item()}")

            labels_flat = labels.flatten()
            preds_flat = pred_labels.flatten()

            # Create mask of indexes where labels are not pad 
            indices = ((labels_flat != ignore_index).nonzero(as_tuple=True)[0]).long()
            labels_flat = labels_flat[indices]
            preds_flat = preds_flat[indices]

            all_targets_train.append([l.item() for l in labels_flat])
            all_preds_train.append([p.item() for p in preds_flat])


        n_total = 0.00001 if n_total == 0 else n_total # Added this to avoid division by zero error
        train_accuracy = n_correct / n_total
        train_accuracy_scores.append(train_accuracy) 
        logger.info(f"Train accuracy = {n_correct} / {n_total} = {train_accuracy}")
        print(f"Train accuracy = {n_correct} / {n_total} = {train_accuracy}")

        # Calculate train F1 score for epoch
        _,_,train_f1_epoch = f1_cues([l_i for l in all_targets_train for l_i in l], [p_i for p in all_preds_train for p_i in p]) # checked that type is int in both lists
        global_train_f1_scores.append(train_f1_epoch)
        print(f"Epoch {i+1} train F1 = {train_f1_epoch}")

        train_loss = sum(train_losses) / len(train_losses)
        global_train_losses.append(train_loss)
        logger.info(f"Epoch {i+1} train loss = {train_loss}")
        print(f"Epoch {i+1} train loss = {train_loss}")

        print("\tEVAL")
        model.eval()
        dev_iter = tqdm(dev_loader)
        n_correct, n_total = 0, 0 
        with torch.no_grad():
            for _, sentences, lengths, labels in dev_iter:
                encoding = tokenizer(sentences, return_tensors="pt", padding=True, is_split_into_words=True, return_offsets_mapping=True) # padding=True pads to the longest in _this_ batch
                mask = get_mask(encoding["offset_mapping"], lengths, encoding["input_ids"].size(1), labels.size(1))

                # To GPU
                del encoding['offset_mapping'] # This is unexpected as input to the model, so remove it
                encoding = {key: value.to(device) for key, value in encoding.items()}
                mask = mask.to(device)
                labels = labels.to(device)

                prediction = model(encoding, mask, lengths, args)
                pred_labels = prediction.argmax(-1) # -1 must mean that this is the argmax of the last dimension (corresponding to the 5 labels)         

                n_correct += ((pred_labels == labels) & (labels != not_cue_index) & (labels != ignore_index)).sum()
                n_total += ((labels != not_cue_index) & (labels != ignore_index)).sum()

                loss = criterion(prediction.flatten(0, 1), labels.flatten())

                validation_losses.append(loss.item())
                dev_iter.set_postfix_str(f"loss: {loss.item()}")

                labels_flat = labels.flatten()
                preds_flat = pred_labels.flatten()

                # Create mask of indexes where labels are not pad 
                indices = ((labels_flat != ignore_index).nonzero(as_tuple=True)[0]).long()
                labels_flat = labels_flat[indices]
                preds_flat = preds_flat[indices]

                all_targets_dev.append([l.item() for l in labels_flat])
                all_preds_dev.append([p.item() for p in preds_flat])

        n_total = 0.00001 if n_total == 0 else n_total # Added this to avoid division by zero error

        val_accuracy = n_correct / n_total
        validation_accuracy_scores.append(val_accuracy) 
        logger.info(f"Validation accuracy = {n_correct} / {n_total} = {val_accuracy}")
        print(f"Validation accuracy = {n_correct} / {n_total} = {val_accuracy}")

        # Calculate dev F1 score for epoch
        _,_,dev_f1_epoch = f1_cues([l_i for l in all_targets_dev for l_i in l], [p_i for p in all_preds_dev for p_i in p])
        global_validation_f1_scores.append(dev_f1_epoch)
        print(f"Epoch {i+1} dev F1 = {dev_f1_epoch}")

        val_loss = sum(validation_losses) / len(validation_losses)
        global_validation_losses.append(val_loss)
        logger.info(f"Epoch {i+1} validation loss = {val_loss}")
        print(f"Epoch {i+1} dev loss = {val_loss}")

        # Invert the loss because the module checks if new score is lower than previous best 
        if early_stopping: 
            early_stopping(dev_f1_epoch, model)
            if early_stopping.early_stop:
                print(f"Stopped early after epoch {i+1}")
                stopped_early_at_epoch = i+1
                break
    
    return train_accuracy_scores, global_train_losses, global_train_f1_scores, validation_accuracy_scores, global_validation_losses, global_validation_f1_scores, stopped_early_at_epoch 



if __name__ == "__main__":
    parser = ArgumentParser()
    # parser.add_argument("--dir_path_prefix", "-dpp", default="~/master_thesis_2023/models/")
    parser.add_argument("--dir_path_prefix", "-dpp", default=os.path.join(os.environ["USERWORK"],"models"))
    
    # ALWAYS CREATE AN EMPTY DIR WITH THIS NAME INSIDE ../models before training
    # Do it this way in order to keep control (not accidentally override other results etc.)
    parser.add_argument("--dir_name", "-dn", type=str, required=True) 
    parser.add_argument("--seed", "-seed", type=int, required=True)
    parser.add_argument("--batch_size", "-bs", type=int, default=32)
    parser.add_argument("--test_batch_size", "-tbs", type=int, default=1)
    parser.add_argument("--drop_last", "-dl", type=int, default=1, help="0 (false) or 1 (true)")
    parser.add_argument("--initial_lr", "-lr", type=float, default=3e-5)
    parser.add_argument("--warmup_steps", "-wus", type=int, default=200)
    parser.add_argument("--epochs", "-ep", type=int, default=20)
    parser.add_argument("--mtype", "-mt", type=str, default=None)
    parser.add_argument("--architecture", "-arch", type=str, default="linear_only", help="linear_only or rnn_linear")
    parser.add_argument("--recurrent_layers", "-rl", type=int, default=0)
    parser.add_argument("--bidirectional", "-bidir", type=int, default=None)
    parser.add_argument("--hidden_size", "-hs", type=int, default=None)
    parser.add_argument("--train_set", "-trs", type=str, required=True)
    parser.add_argument("--dev_set", "-ds", type=str, required=True)
    #parser.add_argument("--test_set", "-tes", type=str, required=True) # We don't actually do testing in this script...
    parser.add_argument("--early_stop", "-es", type=int, default=1)
    parser.add_argument("--es_patience", "-esp", type=int, default=6)
    parser.add_argument("--model_path", "-mp", type=str, default="ltgoslo/norbert2") # language model path

    args = parser.parse_args()

    # Logging
    logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Set seeds
    seed_everything(seed_value=args.seed)

    # Device - use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    print()

    # Datasets and loaders
    X_train = NegationCueDataset(args.train_set)
    train_loader = DataLoader(X_train, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_cue, drop_last=args.drop_last == 1) #batch_size=32, collate_fn=collate_fn, drop_last=True 
    X_dev = NegationCueDataset(args.dev_set)
    dev_loader = DataLoader(X_dev, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_cue, drop_last=args.drop_last == 1)
    #X_test = NegationCueDataset(args.test_set)
    #test_loader = DataLoader(X_test, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn_cue, drop_last=args.drop_last == 1)
    logger.info(f"Datasets and loaders created")
    print("LEN DATASETS")
    print("TRAIN:", len(X_train))
    print("DEV:", len(X_dev))
    #print("TEST:", len(X_test))

    # Model
    vocab = ["AFFIX", "NORMAL", "MULTI", "NOT_CUE", "PAD"]

    model = GeneralNegationModel(len(vocab), args).to(device)

    ignore_idx = 4
    # Loss, optimizer, scheduler
    loss = CrossEntropyLoss(ignore_index=ignore_idx)
    optimizer = AdamW(params=model.parameters(), lr=args.initial_lr)
    scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.epochs * len(train_loader))
    logger.info("Initialized loss, optimizer and scheduler")

    # PATH AND FILENAME FOR STORING MODEL
    model_type_str = None
    if args.architecture == "linear_only":
        model_type_str = "linear"
    elif args.architecture == "crf_linear":
        model_type_str = "crf"
    elif args.architecture == "rnn_linear":
        model_type_str = f"bidir{args.bidirectional}_" + args.mtype
    elif args.architecture == "rnn_crf_linear":
        model_type_str = f"bidir{args.bidirectional}_" + args.mtype + "_crf"

    lang_mod = args.model_path.split("/")[-1]

    model_save_name = f'cuemodel_{date.today()}_warmup{args.warmup_steps}_ep{args.epochs}_{model_type_str}_lr{args.initial_lr}_bs{args.batch_size}_dl{args.drop_last}_{lang_mod}_{model.__class__.__name__}_SEED_{args.seed}.pt'
    
    
    dir_path = os.path.join(args.dir_path_prefix, args.dir_name)
    #path = f"{dir_path}/{model_save_name}" 
    path = os.path.join(dir_path, model_save_name)

    label_mapper = X_train.label_mapper
    # Include path as param, because we need to tell the EarlyStopping module where to save the checkpoints 
    train_accuracies, train_losses, train_f1, val_accuracies, val_losses, val_f1, early_stop_epoch = train(model, device, args, optimizer, scheduler, loss, train_loader, dev_loader, label_mapper, ignore_idx, path) # REMOVE NONE


    res_file = f"{dir_path}/SEED_{args.seed}_results_cue.txt"
    f = open(res_file, 'w', encoding='utf8')
    f.write(f"Output dir: {dir_path}\n")
    f.write(f"Model saved at {path}\n\n")
    f.write(f"ARGUMENTS:\n{str(args)}\n\n")
    f.write(f"Train accuracies: {','.join([str(a.item()) for a in train_accuracies])}\n")
    f.write(f"Validation accuracies: {','.join([str(a.item()) for a in val_accuracies])}\n")
    f.write(f"Train losses: {','.join([str(l) for l in train_losses])}\n")
    f.write(f"Validation losses: {','.join([str(l) for l in val_losses])}\n")
    f.write(f"Train F1: {','.join([str(f1) for f1 in train_f1])}\n")
    f.write(f"Validation F1: {','.join([str(f1) for f1 in val_f1])}\n\n")

    if early_stop_epoch:
        f.write(f"Stopped early at epoch {early_stop_epoch}\n")
    elif not early_stop_epoch and args.early_stop==1:
        f.write("Early stopping used, but did not stop early\n")
    else:
        f.write("Early stopping was not used in training\n")
    f.close()

    print("Train accuracies:", train_accuracies)
    print("Validation accuracies:", val_accuracies)

    print("Train losses:", train_losses)
    print("Validation losses:", val_losses)

    print("Train F1:", train_f1)
    print("Validation F1:", val_f1)

    if not args.early_stop==1: # Model has not already been saved 
        torch.save(model.state_dict(), path)
        print(f"Saved model at {path}")
