from argparse import ArgumentParser
import re
import sys

# Lists of prefixes, suffixes and no-affixes are taken from:
# https://github.com/ltgoslo/norec_neg/blob/main/modeling/evaluation/convert_to_starsem.py
prefixes = ['U', 'ikke-', 'in', 'mis', 'utenom', 'u']
suffixes = ['frie', 'fritt', 'fri', 'løse', 'løst', 'løs', 'tomme']
no_affix = ['Unntaket', 'Uten', 'fri', 'inga', 'ingen', 'ingenting', 'inget',
            'intet', 'mister', 'mistet', 'unngå', 'unngår', 'unntak',
            'unntaket', 'unntatt', 'utan', 'uteble', 'uteblir', 'uten']





if __name__ == "__main__": 
    parser = ArgumentParser()
    parser.add_argument("--cue_file", required=True)
    parser.add_argument("--scope_file", required=True)
    parser.add_argument("--out_file", required=True)
    parser.add_argument("--dataset", default="norec_neg", help="norec_neg or normed_neg")
    
    args = parser.parse_args()


    if args.dataset == "norec_neg":
        print("Use NoReC_neg patterns") # do nothing 
    elif args.dataset == "normed_neg":
        print("Use NorMed_neg patterns")
        prefixes = ['ab', 'an', 'A', 'a', 'im'] + prefixes  # Adding affixes from NorMed_neg (no new suffixes in NorMed_neg)
    else:
        sys.exit("args.dataset must be either norec_neg or normed_neg")    

    print(prefixes)

    cue_data = None
    scope_data = None
    with open(args.cue_file, encoding='utf8') as cue, open(args.scope_file, encoding='utf8') as scope: 
        cue_data = [item for item in cue.read().split("\n\n") if item != '']
        scope_data = [item for item in scope.read().split("\n\n") if item != '']

    cue_sents = [[line.split("\t") for line in item.split("\n")] for item in cue_data]
    scope_sents = [[line.split("\t") for line in item.split("\n")] for item in scope_data]


    last_idx_no_neg = 7
    no_affix_re = re.compile("^" + "$|^".join(no_affix) + "$")
    prefix_re = re.compile("^" + "|^".join(prefixes))
    suffix_re = re.compile("$|".join(suffixes) + "$")


    for i in range(len(cue_sents)):
        #print(cue_sents[i])
        cue_indices = [last_idx_no_neg + n*3 for n in range(len(cue_sents[i][0])) if last_idx_no_neg + n*3 < len(cue_sents[i][0])]
        #print("cue_indices:", cue_indices)

        for c_i in cue_indices:
            for j in range(len(cue_sents[i])): 
                word = cue_sents[i][j][c_i]
                if word == f'{cue_sents[i][j][3]}*': # this was predicted as an AFFIX cue (added a * at the end)
                    print("word predicted as AFFIX:", word)
                    no_affix_match = re.search(no_affix_re, word[:len(word)-1])
                    if no_affix_match:
                        print("matched no_affix")
                        print(no_affix_match.group(0), no_affix_match.span())
                        continue
                    prefix_match = re.search(prefix_re, word[:len(word)-1])
                    if prefix_match: 
                        print("matched prefix")
                        print(prefix_match.group(0), prefix_match.span())

                        scope_sents[i][j][c_i] = prefix_match.group(0) # Update match in predictions file

                        if scope_sents[i][j][c_i+1] != "_": # if the word is predicted as part of scope as well
                            from_idx = prefix_match.span()[1]
                            rest_of_word = word[:len(word)-1][from_idx:]
                            print("rest_of_word:", rest_of_word)
                            scope_sents[i][j][c_i+1] = rest_of_word # Set the rest of the word (excl. the identified affix) as part of scope

                        continue     

                    suffix_match = re.search(suffix_re, word[:len(word)-1])
                    if suffix_match: 
                        print("matched suffix")
                        print(suffix_match.group(0), suffix_match.span())

                        scope_sents[i][j][c_i] = suffix_match.group(0) # Update match in predictions file

                        if scope_sents[i][j][c_i+1] != "_": # if the word is predicted as part of scope as well
                            to_idx = suffix_match.span()[0]
                            rest_of_word = word[:len(word)-1][0:to_idx]
                            print("rest_of_word:", rest_of_word)
                            scope_sents[i][j][c_i+1] = rest_of_word # Set the rest of the word (excl. the identified affix) as part of scope

                        continue

                        # Matched neither no_affix, nor prefixes or suffixes, do nothing 

    with open(args.out_file, 'w', encoding='utf8') as new_pred_file: 
        new_pred_file.write("\n\n".join(["\n".join(["\t".join(line) for line in sent]) for sent in scope_sents]))
        new_pred_file.write("\n\n") # Need some empty lines at the end for the evaluation script not to complain
