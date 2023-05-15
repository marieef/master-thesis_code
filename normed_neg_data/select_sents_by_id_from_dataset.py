"""
Extract sentences by id from a *SEM-formatted file and write to output *SEM file.
Sentence ids must be given as a csv file.
"""

from argparse import ArgumentParser



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ids_file", required=True, help="Path to .csv file containing sentence ids")
    parser.add_argument("--dataset", required=True, help="Path to .sem file with the complete dataset")
    parser.add_argument("--outfile", required=True, help="Path to .sem file to output the selected sentences to")

    args = parser.parse_args()

    with open(args.dataset, encoding="utf8") as data, open(args.ids_file, encoding="utf8") as ids_file, open(args.outfile, "w", encoding="utf8") as outfile:
        ids = ids_file.read().split(",")
        selected_sents = [sent for sent in data.read().split("\n\n") if sent != '' and sent.split("\n")[0].split("\t")[1] in ids]
        #for s in selected_sents:
        #    print(s)
        #    print()

        print("LEN selected sents:", len(selected_sents))
        print("LEN ids:", len(ids))
        #print(ids)

        outfile.write("\n\n".join(selected_sents))
