from torch.utils.data import Dataset
import torch
from dataset_helpers import *

class NegationScopeDatasetForEvaluation(Dataset):
    """
    Includes only those sentences which has a predicted cue. Includes each sentence once for each predicted cue.
    """
    def __init__(self, sem_file):
        # When evaluating, sem_file is a file with predicted cues
        data = None
        with open(sem_file, encoding='utf8') as f: 
            data = [item for item in f.read().split("\n\n") if item != '']
        lines = [[line.split("\t") for line in item.split("\n")] for item in data]

        # If no cue: int((8-7)/3) = int(1/3) = 0 --> sentence not included
        # Else: int((3*N + 7 - 7)) = int(N) = N, i.e. 1 per cue
        self.n_copies = [int((len(s[0])-7) / 3) for s in lines]
        self.source = [lines[i][0][1] for i in range(len(lines)) for j in range(self.n_copies[i])] 
        sentences = [[lines[j][i][3] for i in range(len(lines[j]))] for j in range(len(lines)) for k in range(self.n_copies[j])] 
        
        # When reading from a sem-file with cue predictions, affixal cue predictions are marked with a "*" to make it possible for get_cue_label_scope 
        # to know that it is a predicted affixal cue
        cue_labels = [[get_cue_label_neg(lines[j], i, k) for i in range(len(lines[j]))] for j in range(len(lines)) for k in range(self.n_copies[j])] 

        # self.labels will always be 0 when reading from cue prediction file
        self.labels = [[get_scope_label(lines[j], i, k) for i in range(len(lines[j]))] for j in range(len(lines)) for k in range(self.n_copies[j])] 

        # Sentences with cue replaced by special cue token 'token[n]'
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
