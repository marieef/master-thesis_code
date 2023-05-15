"""
Can be used for both cue and scope model
Requires n_classes as first parameter (not a list of the actual labels)

Based on code provided as part of the IN5550 course ('Neural Methods in Natural Language Processing') at the University of Oslo,
spring 2022. This is the link to the course web page: https://www.uio.no/studier/emner/matnat/ifi/IN5550/v22/index.html
"""

from transformers import AutoModel
import torch
from torch import nn


class GeneralNegationModel(torch.nn.Module):
    def __init__(self, n_classes, args, freeze=False):
        super().__init__()
        self.bert = AutoModel.from_pretrained(args.model_path)

        print("N CLASSES:", n_classes)
        if n_classes < 2: 
            print("There must be at least 2 classes!")

        if freeze:
            self.bert.requires_grad_(False)
        
        self.recurrent = None
        if args.mtype: # None when these are not to be used
            modules = {
                    "rnn": nn.RNN,
                    "gru": nn.GRU,
                    "lstm": nn.LSTM
            }

            module = modules[args.mtype]
            self.recurrent = module(input_size=self.bert.config.hidden_size, hidden_size=args.hidden_size, num_layers=args.recurrent_layers, bidirectional=args.bidirectional==1, batch_first=True) 
        # Set input size of final linear layer:
        linear_input_size = None
        if args.architecture == "linear_only":
            linear_input_size = self.bert.config.hidden_size
        elif args.architecture == "rnn_linear": 
            linear_input_size = 2 * args.hidden_size if args.bidirectional else args.hidden_size

        self.linear = torch.nn.Linear(linear_input_size, 1 if n_classes == 2 else n_classes) 

    def forward(self, encoding, mask, lengths, args):
        # add if-statements -> different behaviour depending on model architecture 
        output = self.bert(**encoding).last_hidden_state    # shape: [B, N_subwords, D]
        pooled = torch.einsum("bsd,bws->bwd", output, mask) 
        pooled = pooled / mask.sum(-1, keepdim=True).clamp(min=1.0)

        if args.architecture == "linear_only":
            output = self.linear(pooled)

        elif args.architecture == "rnn_linear": 
            pooled = nn.utils.rnn.pack_padded_sequence(pooled, [len(sent) for sent in lengths], batch_first=True, enforce_sorted=False)
            states, _ = self.recurrent(pooled)
            states = nn.utils.rnn.pad_packed_sequence(states, batch_first=True)[0]
            output = self.linear(states)

        return output