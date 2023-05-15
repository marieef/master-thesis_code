def get_cue_label(sent_lines, word_idx):
    '''
    This version is used in NegationCueDataset.

    We want to know if the word is a cue in one of the negations in the sentence, 
    and if so, what type of cue.
    '''
    last_idx_no_neg = 7 # constant
    # use sent_lines[0] because each line has same lengths
    cue_indices = [last_idx_no_neg + i*3 for i in range(len(sent_lines[0])) if last_idx_no_neg + i*3 < len(sent_lines[0])]

    cue_label = "NOT_CUE" # Assuming that a cue word can only be part of ONE negation 
    for cue_idx in cue_indices:
        c = 0
        for word_line in sent_lines:
            if word_line[cue_idx] != "_" and word_line[cue_idx] != "***": # assume that _ and *** cannot be cues
                c+=1
        if c > 1 and sent_lines[word_idx][cue_idx] != "***" and sent_lines[word_idx][cue_idx] != "_": 
            # There is more than 1 cue word in this negation, and this particular word is one of them
            cue_label = "MULTI"
        elif sent_lines[word_idx][cue_idx] == "***" or sent_lines[word_idx][cue_idx] == "_":
            continue # The label is already set to NOT_CUE by default, and will be changed if it IS part of a cue 
        elif sent_lines[word_idx][cue_idx] == sent_lines[word_idx][3]:
            cue_label = "NORMAL"
        elif sent_lines[word_idx][cue_idx] != sent_lines[word_idx][3]:
            # Have checked that it is not a "NOT_CUE", then it must be an affix
            cue_label = "AFFIX"
        else:
            continue

    # IF there happens to be > 1 cue as part of the word, the last one of these will be returned
    return cue_label

def get_special_cue_token(word, label):
    """
    Modified for augment method.
    Remove the space between special token and word.
    Used in scope dataset.
    """
    token = word # will remain like this unless one of the following applies: 
    if label == 'AFFIX':
        token = 'token[0]' + token
    elif label == 'NORMAL':
        token = 'token[1]' + token
    elif label == 'MULTI':
        token = 'token[2]' + token
    return token

def get_scope_label(line, word_idx, neg_idx):
    """
    Used in scope dataset.
    """
    last_idx_no_neg = 7 # this is constant
    scope_idx = last_idx_no_neg + neg_idx*3 + 1
    
    # line[word_idx][last_idx_no_neg] == "***": # replaced by the line below
    if len(line[word_idx]) == last_idx_no_neg + 1:
        return 0
    elif line[word_idx][scope_idx] == "_":
        return 0
    return 1 # is in scope

def get_cue_label_neg(sent_lines, word_idx, neg_idx):
    """
    This version works for one specific negation (third argument neg_idx).
    Used in scope dataset.
    """
    last_idx_no_neg = 7 # constant
    cue_idx = last_idx_no_neg + neg_idx*3
    c = 0
    for word_line in sent_lines:
        if word_line[cue_idx] != "_" and word_line[cue_idx] != "***": # assume that _ and *** cannot be cues
            c+=1
    if c > 1 and sent_lines[word_idx][cue_idx] != "***" and sent_lines[word_idx][cue_idx] != "_":
        # There is more than 1 cue word in this negation, and this particular word is part of it
        return "MULTI"
    if sent_lines[word_idx][cue_idx] == "***" or sent_lines[word_idx][cue_idx] == "_":
        return "NOT_CUE"
    elif sent_lines[word_idx][cue_idx] == sent_lines[word_idx][3]:
        # The whole word is the cue
        return "NORMAL"
    elif sent_lines[word_idx][cue_idx] != sent_lines[word_idx][3]:
        # A part of the word is the cue (it is not NOT_CUE, this was checked above)
        return "AFFIX"
    return "NOT_CUE"
    