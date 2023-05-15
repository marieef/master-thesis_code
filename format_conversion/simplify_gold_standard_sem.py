import argparse

"""
Convert the original gold standard (in starsem format) to a simplified version. 
Example: the word "uvanlig" (unusual) where "u" is annotated as cue and "vanlig" as scope: 
cue --> "uvanlig", scope --> "uvanlig"
"""

def convert(in_path, out_path):
    with open(in_path, encoding='utf8') as infile, open(out_path, 'w', encoding='utf8') as outfile:
        sents = [[item.split("\t") for item in s.split("\n")] for s in infile.read().split("\n\n") if s != ""]

        print("LEN(sents):", len(sents))

        for i in range(len(sents)):
            # Find possible cue and scope indices in this sentence:
            last_idx_no_neg = 7 # This index contains *** if there is no negation
            cue_indices = [last_idx_no_neg + n*3 for n in range(len(sents[i][0])) if last_idx_no_neg + n*3 < len(sents[i][0])]
            scope_indices = [last_idx_no_neg + n*3 + 1 for n in range(len(sents[i][0])) if last_idx_no_neg + n*3 + 1 < len(sents[i][0])]

            # if len(sents[i][0]) > 8:
            #     print("Cue indices:", cue_indices)
            #     print("Scope indices:", scope_indices)

            if len(sents[i][0]) > last_idx_no_neg+1: # If there is at least one negation in the sentence
                for j in range(len(sents[i])): # for word in sentence 
                    for c_i in cue_indices:
                        #print("CUE BEFORE:", sents[i][j][c_i])
                        sents[i][j][c_i] = sents[i][j][3] if (sents[i][j][c_i] != "***" and sents[i][j][c_i] != "_") else sents[i][j][c_i]
                        #print("CUE AFTER:", sents[i][j][c_i])

                    for s_i in scope_indices:
                        #print("SCOPE BEFORE:", sents[i][j][s_i])
                        sents[i][j][s_i] = sents[i][j][3] if sents[i][j][s_i] != "_" else sents[i][j][s_i]
                        #print("SCOPE AFTER:", sents[i][j][s_i])

        result = "\n\n".join(["\n".join(["\t".join(line) for line in sent]) for sent in sents])

        # Write sents (modified) to file:
        outfile.write(result)
        print(f"Simplified gold standard written to {out_path}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", required=True)
    parser.add_argument("--outfile", required=True)

    args = parser.parse_args()

    convert(args.infile, args.outfile)