
"""
The training loop is based on code provided as part of the IN5550 course ('Neural Methods in Natural Language Processing') at the University of Oslo,
spring 2022. This is the link to the course web page: https://www.uio.no/studier/emner/matnat/ifi/IN5550/v22/index.html

"""

from utils import *
from earlystopping import *

from tqdm import tqdm 
from transformers import AutoTokenizer
from sklearn.metrics import classification_report
import torch
import logging
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from model import *
from datasets import *
from datetime import date
import transformers
from torch.nn import BCEWithLogitsLoss
from torch.optim import AdamW



def train(model, device, args, optimizer, scheduler, criterion, train_loader, dev_loader, path=None):
    train_accuracy_scores = []
    global_train_losses = []
    global_train_f1_scores = [] 
    validation_accuracy_scores = []
    global_validation_losses = []
    global_validation_f1_scores = []
    stopped_early_at_epoch = None

    not_scope_index, ignore_index = 0, 2
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # Early stopping
    early_stopping = None
    if args.early_stop == 1:
        early_stopping = EarlyStopping(patience = args.es_patience, verbose=True, save_path=path) if path else EarlyStopping(patience = args.es_patience, verbose=True)

    for i in tqdm(range(args.epochs)):
        train_losses = []
        validation_losses = []
        sigmoid = torch.nn.Sigmoid()

        # Use these to compute dev F1 after each epoch
        # Add all preds / targets to lists to avoid zero division error
        all_preds_dev = []
        all_targets_dev = []

        all_preds_train = []
        all_targets_train = []

        model.train()
        logger.info("Epoch {} started".format(i+1))
        print("Epoch {} started".format(i+1))
        print("\tTRAIN")

        train_iter = tqdm(train_loader)
        print("len train_iter:", len(train_iter))
        n_correct, n_total = 0, 0

        for j, (_, sentences, lengths, labels) in enumerate(train_iter):
            encoding = tokenizer(sentences, return_tensors="pt", padding='max_length', truncation=True, max_length=args.max_length, is_split_into_words=True, return_offsets_mapping=True) # padding=True pads to the longest in _this_ batch
            mask = get_mask(encoding["offset_mapping"], lengths, encoding["input_ids"].size(1), labels.size(1))

            # To GPU
            del encoding['offset_mapping'] # This is unexpected as input to the model, so remove it
            encoding = {key: value.to(device) for key, value in encoding.items()}
            mask = mask.to(device)
            labels = labels.to(device)
                        
            prediction = model(encoding, mask, lengths, args)

            # Get predicted labels by applying sigmoid:
            pred_labels = torch.squeeze(sigmoid(prediction) > 0.5).int()

            n_correct += ((pred_labels == labels) & (labels != not_scope_index) & (labels != ignore_index)).sum()
            n_total += ((labels != not_scope_index) & (labels != ignore_index)).sum()

            # Used this approach: https://discuss.pytorch.org/t/unclear-about-weighted-bce-loss/21486
            weight = torch.tensor([1, 1, 0]) # index 2 corresponds to padding label
            weight_ = torch.unsqueeze(weight[labels.flatten().data.view(-1).long()].view_as(labels.flatten()), 1).to(device) 

            loss = criterion(prediction.flatten(0, 1).float(), torch.unsqueeze(labels.flatten().float(), 1)).to(device)
            loss = loss * weight_ # class weighted
            
            if args.loss_mean == "auto":
                loss = loss.mean() # ORIGINAL SOLUTION
            elif args.loss_mean == "manual":
                loss = loss.sum() / weight_.sum() # Other solution, not used in our model.     

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad() # Place anywhere but not between loss.backward() and optimizer.step(): https://discuss.pytorch.org/t/where-should-i-place-zero-grad/101886

            train_losses.append(loss.item())
            train_iter.set_postfix_str(f"loss: {loss.item()}")

            # For F1 computation: ignore the indices where target is pad label
            labels = labels.flatten()
            pred_labels = pred_labels.flatten()

            indices = ((labels != 2).nonzero(as_tuple=True)[0]).long()
            labels = labels[indices]
            pred_labels = pred_labels[indices]

            all_targets_train.append([l.item() for l in labels.cpu()])
            all_preds_train.append([p.item() for p in pred_labels.cpu()])
        
        n_total = 0.00001 if n_total == 0 else n_total # Added this to avoid division by zero error
        train_accuracy = n_correct / n_total
        train_accuracy_scores.append(train_accuracy) 
        logger.info(f"Train accuracy = {n_correct} / {n_total} = {train_accuracy}")
        print(f"Train accuracy = {n_correct} / {n_total} = {train_accuracy}")

        classification_dict_train = classification_report([l_i for l in all_targets_train for l_i in l], [p_i for p in all_preds_train for p_i in p], output_dict=True, labels=[0,1])
        f1 = classification_dict_train["1"]["f1-score"] # F1 score for class 1 (in-scope)
        global_train_f1_scores.append(f1)
        print("Train F1:", global_train_f1_scores[-1])

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
                encoding = tokenizer(sentences, return_tensors="pt", padding='max_length', truncation=True, max_length=args.max_length, is_split_into_words=True, return_offsets_mapping=True) # padding=True pads to the longest in _this_ batch
                mask = get_mask(encoding["offset_mapping"], lengths, encoding["input_ids"].size(1), labels.size(1))

                # To GPU
                del encoding['offset_mapping'] # This is unexpected as input to the model, so remove it
                encoding = {key: value.to(device) for key, value in encoding.items()}
                mask = mask.to(device)
                labels = labels.to(device)

                prediction = model(encoding, mask, lengths, args)
                pred_labels = torch.squeeze((prediction > 0.5).int())

                n_correct += ((pred_labels == labels) & (labels != not_scope_index) & (labels != ignore_index)).sum()
                n_total += ((labels != not_scope_index) & (labels != ignore_index)).sum()
            
                weight = torch.tensor([1, 1, 0]) # index 2 corresponds to padding label
                weight_ = torch.unsqueeze(weight[labels.flatten().data.view(-1).long()].view_as(labels.flatten()), 1).to(device)
                
                loss = criterion(prediction.flatten(0, 1).float(), torch.unsqueeze(labels.flatten().float(), 1)).to(device)
                loss = loss * weight_ # class weighted
                
                if args.loss_mean == "auto":
                    loss = loss.mean() # ORIGINAL SOLUTION
                elif args.loss_mean == "manual":
                    loss = loss.sum() / weight_.sum() # # Other solution, not used in our model.             

                validation_losses.append(loss.item())
                dev_iter.set_postfix_str(f"loss: {loss.item()}")

                # For F1 computation: ignore the indices where target is pad label
                labels = labels.flatten()
                pred_labels = pred_labels.flatten()

                indices = ((labels != 2).nonzero(as_tuple=True)[0]).long()
                labels = labels[indices]
                pred_labels = pred_labels[indices]

                all_targets_dev.append([l.item() for l in labels.cpu()])
                all_preds_dev.append([p.item() for p in pred_labels.cpu()])

        n_total = 0.00001 if n_total == 0 else n_total # Added this to avoid division by zero error

        val_accuracy = n_correct / n_total
        validation_accuracy_scores.append(val_accuracy) 
        logger.info(f"Validation accuracy = {n_correct} / {n_total} = {val_accuracy}")
        print(f"Validation accuracy = {n_correct} / {n_total} = {val_accuracy}")

        classification_dict_dev = classification_report([l_i for l in all_targets_dev for l_i in l], [p_i for p in all_preds_dev for p_i in p], output_dict=True, labels=[0,1])
        f1 = classification_dict_dev["1"]["f1-score"] # F1 score for class 1 (in-scope)
        global_validation_f1_scores.append(f1)
        print(f"Validation F1:", global_validation_f1_scores[-1])

        val_loss = sum(validation_losses) / len(validation_losses)
        global_validation_losses.append(val_loss)
        logger.info(f"Epoch {i+1} validation loss = {val_loss}")
        print(f"Epoch {i+1} validation loss = {val_loss}")

        if early_stopping: 
            early_stopping(global_validation_f1_scores[-1], model)
            
            # Invert the loss because the module is based on a value which should INcrease (F1)
            #early_stopping(-global_validation_losses[-1], model) # Early stopping based on validation loss
            if early_stopping.early_stop:
                print(f"Stopped early after epoch {i+1}")
                stopped_early_at_epoch = i+1
                break

    return train_accuracy_scores, global_train_losses, global_train_f1_scores, validation_accuracy_scores, global_validation_losses, global_validation_f1_scores, stopped_early_at_epoch


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dir_path_prefix", "-dpp", default=os.path.join(os.environ["USERWORK"],"finetune_on_ngsbnc"))
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
    parser.add_argument("--loss_mean", "-lmean", type=str, default="auto", help="auto or manual")
    parser.add_argument("--trained_model_path", "-tmp", type=str, required=True) # model to finetune, full path
    parser.add_argument("--max_length", "-maxl", type=int, default=128) # max sequence length
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

    if device == 'cuda':
        torch.cuda.empty_cache()
        print("torch.cuda.empty_cache()")

    # Datasets and loaders
    X_train = NegationScopeDataset(args.train_set)
    train_loader = DataLoader(X_train, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_scope, drop_last=args.drop_last == 1) #batch_size=32, collate_fn=collate_fn, drop_last=True 
    X_dev = NegationScopeDataset(args.dev_set)
    dev_loader = DataLoader(X_dev, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_scope, drop_last=args.drop_last == 1)
    #X_test = NegationScopeDataset(args.test_set)
    #test_loader = DataLoader(X_test, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn_scope, drop_last=args.drop_last == 1)
    logger.info(f"Datasets and loaders created")
    print("LEN DATASETS")
    print("TRAIN:", len(X_train))
    print("DEV:", len(X_dev))
    #print("TEST:", len(X_test))

    # Model
    #full_path = os.path.join(os.environ["USERWORK"], args.trained_model_path)
    model = GeneralNegationModel(n_classes=2, args=args).to(device)
    model.load_state_dict(torch.load(args.trained_model_path))

    # Loss, optimizer, scheduler
    loss = BCEWithLogitsLoss(reduction='none')

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

    #lang_mod = args.model_path.split("/")[-1]
    lang_mod = "nbbert-large"
    model_save_name = f'scopemodel_{date.today()}_warmup{args.warmup_steps}_ep{args.epochs}_{model_type_str}_lr{args.initial_lr}_bs{args.batch_size}_dl{args.drop_last}_{lang_mod}_{model.__class__.__name__}_lmean-{args.loss_mean}_SEED_{args.seed}.pt'
    
    dir_path = os.path.join(args.dir_path_prefix, args.dir_name)    
    path = f"{dir_path}/{model_save_name}" 

    # Include path as param, because we need to tell the EarlyStopping module where to save the checkpoints 
    train_accuracies, train_losses, train_f1, val_accuracies, val_losses, val_f1, early_stop_epoch = train(model, device, args, optimizer, scheduler, loss, train_loader, dev_loader, path)

    res_file = f"{dir_path}/SEED_{args.seed}_results_scope.txt"
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

    print("Train accuracies:", train_accuracies)
    print("Validation accuracies:", val_accuracies)

    print("Train losses:", train_losses)
    print("Validation losses:", val_losses)

    print("Train F1:", train_f1)
    print("Validation F1:", val_f1)

    if early_stop_epoch:
        f.write(f"Stopped early at epoch {early_stop_epoch}\n")
    elif not early_stop_epoch and args.early_stop == 1:
        f.write("Early stopping used, but did not stop early\n")
    else:
        f.write("Early stopping was not used in training\n")
    f.close()

    if not args.early_stop == 1: # Model has not already been saved 
        print("Early stopping not used, so save at end.")
        torch.save(model.state_dict(), path)
        print(f"Saved model at {path}")
    print(f"Saved model at {path}")
