from torch.utils.data import Dataset
import torch
from dataset_helpers import *

class NegationCueDataset(Dataset):
    def __init__(self, sem_file):
        data = None
        with open(sem_file, encoding='utf8') as f: 
            data = [item for item in f.read().split("\n\n") if item != '']

        lines = [[line.split("\t") for line in item.split("\n")] for item in data]
        self.source = [lines[i][0][1] for i in range(len(lines))] # 
        self.sentences = [[lines[j][i][3] for i in range(len(lines[j]))] for j in range(len(lines))]
        self.labels = [[get_cue_label(lines[j], i) for i in range(len(lines[j]))] for j in range(len(lines))]
        # lines[j] is a list with one inner list (=one line in sem-file) for each word in the sentence. lines[j] represents the whole sentence.

        self.label_mapper = {"AFFIX": 0, "NORMAL": 1, "MULTI": 2, "NOT_CUE": 3, "PAD": 4}
        self.inv_label_mapper = {i:label for label, i in self.label_mapper.items()}
        print("Label mapper:", self.label_mapper)
        print("Inverse label mapper:", self.inv_label_mapper)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, i):
        source = self.source[i]
        words = self.sentences[i]
        # We need the length of each word to use with offsets_mapping in get_mask
        lengths = [len(word) for word in words]
        # Get numerical labels
        labels = torch.LongTensor([self.label_mapper[label] for label in self.labels[i]])
        #labels = torch.LongTensor([self.label_mapper.get(label, -1) for label in self.labels[i]])
        return source, words, lengths, labels


class NegationScopeDataset(Dataset):
    def __init__(self, sem_file):
        data = None
        with open(sem_file, encoding='utf8') as f: 
            data = [item for item in f.read().split("\n\n") if item != '']
        
        lines = [[line.split("\t") for line in item.split("\n")] for item in data]
        # Include 1 copy of the sentence if it has no negations. Else: one copy per negation.
        # self.n_copies = [1 if s[0][7] == "***" else int((len(s[0])-7) / 3) for s in lines]
        self.n_copies = [1 if len(s[0]) == 8 else int((len(s[0])-7) / 3) for s in lines]
        self.source = [lines[i][0][1] for i in range(len(lines)) for j in range(self.n_copies[i])] 
        sentences = [[lines[j][i][3] for i in range(len(lines[j]))] for j in range(len(lines)) for k in range(self.n_copies[j])]
        
        cue_labels = [[get_cue_label_neg(lines[j], i, k) for i in range(len(lines[j]))] for j in range(len(lines)) for k in range(self.n_copies[j])]
        self.labels = [[get_scope_label(lines[j], i, k) for i in range(len(lines[j]))] for j in range(len(lines)) for k in range(self.n_copies[j])]

        # Sentences with cues marked by a string consisting of special cue token 'token[n]' + the cue word itself
        self.sentences = [[get_special_cue_token(sentences[i][j], cue_labels[i][j]) for j in range(len(sentences[i]))] for i in range(len(sentences))]

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, i):
        source = self.source[i]
        words = self.sentences[i]
        # We need the length of each word to use with offsets_mapping in get_mask
        lengths = [len(word) for word in words]
        # Get numerical labels
        labels = torch.LongTensor([label for label in self.labels[i]])

        return source, words, lengths, labels