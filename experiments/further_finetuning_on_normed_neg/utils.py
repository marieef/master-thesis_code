"""
Functions for seeding, padding and creating a subword mask are borrowed from code provided as part of the 
IN5550 course ('Neural Methods in Natural Language Processing') at the University of Oslo, spring 2022. 
The webpage of the course can be found here: https://www.uio.no/studier/emner/matnat/ifi/IN5550/v22/index.html

"""

import torch
import torch.nn.functional as F
import os
import random

def seed_everything(seed_value=5550):
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def collate_fn_cue(batch):
    """
    Padding for cue detection model. 4 is used as the pad label.
    """
    sources, sentences, lengths, labels = zip(*batch)

    longest = max(label.size(0) for label in labels)
    labels = torch.stack([F.pad(label, (0, longest - label.size(0)), value=4) for label in labels])

    return list(sources), list(sentences), list(lengths), labels


def collate_fn_scope(batch):
    """
    Padding for scope resolution model. 2 is used as the pad label.
    """
    sources, sentences, lengths, labels = zip(*batch)

    longest = max(label.size(0) for label in labels)
    labels = torch.stack([F.pad(label, (0, longest - label.size(0)), value=2) for label in labels])

    return list(sources), list(sentences), list(lengths), labels


def get_mask(offset_mapping, lengths, n_subwords: int, n_words: int):
    """
    Create mask for mapping between words and subwords (which subwords belong to which word). 
    Used for subword pooling. 
    """
    offset_mapping = offset_mapping.tolist()
    mask = torch.zeros(len(lengths), n_words, n_subwords)

    for i_batch in range(len(lengths)):
        current_word, remaining_len = 0, lengths[i_batch][0]

        for i, (start, end) in enumerate(offset_mapping[i_batch]):
            if start == end:
                continue

            mask[i_batch, current_word, i] = 1
            remaining_len -= end - start

            if remaining_len <= 0 and current_word < len(lengths[i_batch]) - 1:
                current_word += 1
                remaining_len = lengths[i_batch][current_word]

    return mask


class Arguments:
    """
    Used in google colab, should not be needed on Saga.
    """
    def __init__(self, epochs, warmup_steps, lr, test, batch_size, drop_last, outfile_test, 
                train_set, dev_set, test_set, mtype, hidden_size, bidirectional, model_path, 
                architecture, recurrent_layers, seed, early_stop, patience):
        self.epochs = epochs
        self.warmup_steps = warmup_steps
        self.initial_lr = lr
        self.test = test
        self.batch_size = batch_size
        self.test_batch_size = 1
        self.drop_last = drop_last
        self.outfile_test = outfile_test
        self.train_set = train_set
        self.dev_set = dev_set
        self.test_set = test_set
        self.mtype = mtype # rnn, gru, lstm
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.model_path = model_path
        self.architecture = architecture
        self.recurrent_layers = recurrent_layers
        self.seed = seed
        self.early_stop = early_stop
        self.es_patience = patience

    def __str__(self):
        return f"""epochs: {self.epochs}\nwarmup_steps: {self.warmup_steps}\nlr: {self.initial_lr}\ntest (boolean): {self.test}\nbs: {self.batch_size}\ntbs: {self.test_batch_size}\ndl: {self.drop_last}
train: {self.train_set}\ndev: {self.dev_set}\ntest: {self.test_set}\nmtype: {self.mtype}\nhs: {self.hidden_size}\nbidir: {self.bidirectional}\nmodel_path: {self.model_path}
architecture: {self.architecture}\nrecurrent_layers: {self.recurrent_layers}\nseed: {self.seed}\nearly stop: {self.early_stop}\nes_patience: {self.es_patience}"""