"""
Take as input two files: one with the gold annotations and one with the predictions. 
Use two ists of predefined medical terms defined by Sadhukhan and check for each occurrence if these are negated or not in gold and pred. 
Compute TP, TN, FP, FN and use this to compute F-score (F1). 

"""

from argparse import ArgumentParser
import json
import os
import re
from collections import defaultdict, Counter

def sanity_check(gold_dict, pred_dict): 
    for term in gold_dict: 
        for sid in gold_dict[term]: 
            gold_spans = sorted([s for s,_ in gold_dict[term][sid]])
            pred_spans = sorted([s for s,_ in pred_dict[term][sid]])
            assert gold_spans == pred_spans

    for term in pred_dict:
        for sid in pred_dict[term]:
            gold_spans = sorted([s for s,_ in gold_dict[term][sid]])
            pred_spans = sorted([s for s,_ in pred_dict[term][sid]])
            assert gold_spans == pred_spans

    print("Sanity check passed :-)")

def sanity_check_no_sorting(gold_dict, pred_dict): 
    for term in gold_dict: 
        for sid in gold_dict[term]: 
            gold_spans = [s for s,_ in gold_dict[term][sid]]
            pred_spans = [s for s,_ in pred_dict[term][sid]]
            #if gold_spans != []:
            #    print("gold_spans:", gold_spans)
            #    print("pred_spans:", pred_spans)
            assert gold_spans == pred_spans

    for term in pred_dict:
        for sid in pred_dict[term]:
            gold_spans = [s for s,_ in gold_dict[term][sid]]
            pred_spans = [s for s,_ in pred_dict[term][sid]]
            #if gold_spans != []:
            #    print("gold_spans:", gold_spans)
            #    print("pred_spans:", pred_spans)
            assert gold_spans == pred_spans

    print("Sanity check (no sorting of lists) passed :-)")

def tp(gold_dict, pred_dict):
    counter = 0
    instances = []
    for term in gold_dict: 
        for sid in gold_dict[term]: 
            for ((span_g,neg_g),(span_p,neg_p)) in zip(gold_dict[term][sid],pred_dict[term][sid]):
                #print(f"\n'{term}' in {sid}")
                #print("gold:", span_g, neg_g)
                #print("pred:", span_p, neg_p)
                if neg_g and neg_p: # both are True, i.e. negated in gold standard and negated in predictions
                    counter += 1
                    instances.append((term,sid,span_g))
    return counter, instances

def tn(gold_dict, pred_dict):
    counter = 0
    instances = []
    for term in gold_dict:
        for sid in gold_dict[term]:
            for ((span_g,neg_g),(span_p,neg_p)) in zip(gold_dict[term][sid],pred_dict[term][sid]):
                if not neg_g and not neg_p: 
                    counter += 1
                    instances.append((term,sid,span_g))
    return counter, instances

def fp(gold_dict, pred_dict):
    counter = 0
    instances = []
    for term in gold_dict:
        for sid in gold_dict[term]:
            for ((span_g,neg_g),(span_p,neg_p)) in zip(gold_dict[term][sid],pred_dict[term][sid]):
                if not neg_g and neg_p: 
                    counter += 1
                    instances.append((term,sid,span_g))
    return counter, instances

def fn(gold_dict, pred_dict):
    counter = 0
    instances = []
    for term in gold_dict:
        for sid in gold_dict[term]:
            for ((span_g,neg_g),(span_p,neg_p)) in zip(gold_dict[term][sid],pred_dict[term][sid]):
                if neg_g and not neg_p: 
                    counter += 1
                    instances.append((term,sid,span_g))
    return counter, instances

def precision(tp, fp):
    return tp / (tp + fp)

def recall(tp, fn):
    return tp / (tp + fn)

def f1(precision, recall):
    return 2 * ((precision * recall) / (precision + recall))


def get_overlapping_terms(terms_list):
    overlap = defaultdict(lambda: [])
    for t1 in terms_list: 
        for t2 in terms_list: 
            if t1 in t2 and t1 != t2: 
                overlap[t1].append(t2)
    return overlap



if __name__ == "__main__": 
    parser = ArgumentParser()
    parser.add_argument("--gold", required=True, help="Path to the gold standard (json)")
    parser.add_argument("--pred", required=True, help="Path to the prediction file (sem)")
    parser.add_argument("--terms_in_dev", default="sadhukhan_files/myWords_present_in_dev.txt", help="Path to file with custom list of medical terms, only thos present in dev(train+devtest)")
    parser.add_argument("--terms", default="sadhukhan_files/myWords.txt", help="Path to file with custom list of medical terms")
    parser.add_argument("--normedterms", default="sadhukhan_files/NorMedTermCondition.txt", help="Path to file with predefined terms related to conditions")
    parser.add_argument("--testset_ids", required=True, help="Path to file containing sent_ids of sents in heldout test set, separated by commas")
    parser.add_argument("--testset_type", required=True, help="A short string to add to the output filename, to separate results for different test sets")
    parser.add_argument("--output_file", default="evaluation_sadhukhan_GOLD_marie_NEGATED_SENTS.txt")

    args = parser.parse_args()

    g_sents, p_sents, terms, terms_pred = None, None, None, None 

    with open(args.gold, encoding="utf8") as g, open(args.pred, encoding="utf8") as p, open(args.terms, encoding="utf8") as t, open(args.terms_in_dev, encoding="utf8") as tdev, open(args.normedterms, encoding="utf8") as nmt, open(args.testset_ids, encoding="utf8") as test:
        test_ids = test.read().split(",")
        
        g_sents = [s for s in json.load(g) if s["sent_id"] in test_ids]
        #p_sents = [word_line for sent in p.read().split("\n\n") for word_line in sent.split("\n") if sent != '' and word_line.split("\t")[1] in test_ids]
        p_sents = [[word_line.split("\t") for word_line in sent.split("\n") if word_line.split("\t")[1] in test_ids] for sent in p.read().split("\n\n") if sent != '']
        p_sents = [s for s in p_sents if len(s) > 0]

        terms = [line.strip().lower() for line in t if line != '']
        normedterms = [line.strip().lower() for line in nmt if line != '']

        # All terms, including those custom terms only present in held-out testset
        terms += normedterms # Use both the terms in myWords.txt and those in the large NorMedTermClinical.txt file

        # The terms to look for in pred: do not include the custom terms only present in held-out testset
        terms_pred = [line.strip().lower() for line in tdev if line != ''] + normedterms


        print("Terms:", len(terms))
        terms += [t.replace(",", "") for t in terms if t.replace(",", "") not in terms] # Sadhukhan does this
        print("Terms after adding terms with commas removed:", len(terms))

        print("Terms available for recognition:", len(terms_pred))
        terms_pred += [t.replace(",", "") for t in terms_pred if t.replace(",", "") not in terms_pred]
        print("Terms available for recognition, after adding terms with commas removed::", len(terms_pred))

        print("GOLD:", len(g_sents))
        print(g_sents[:10])

        print("PRED:", len(p_sents))
        for s in p_sents[:10]: 
            print(s)
            print()

        terms = sorted(list(set(terms))) # one term occurs twice in the original list
        terms_pred = sorted(list(set(terms_pred)))
        print("Terms after removing duplicates:", len(terms))
        #print("set(terms), lowercased:", len(set([t.lower() for t in terms])))
        print(terms[:10])
        print("Terms available for recognition, after removing duplicates:", len(terms_pred))
        print("len(test_ids):", len(test_ids))
    
    overlapping_terms = get_overlapping_terms(terms)

    all_occurrences_boolean_map = defaultdict(lambda: defaultdict(lambda: []))

    # Count terms (negated and non negated) in gold 
    for term in terms: 
        re_term = term.replace("-", "\-")
        re_term = re_term.replace("*", "\*")

        for sent in g_sents: # json  
            matches = re.finditer(rf'\b(?<!\-){re_term}\b', sent['text'], flags=re.IGNORECASE)
            matches = [m for m in matches]

            for m in matches: 
                not_matching_any_scope_fragment = True 

                for neg in sent['negations']: 
                    for i in range(len(neg['Scope'][0])): # for each scope fragment 
                        span = neg['Scope'][1][i].split(":") # scope part span in sent['text']
                        start, end = int(span[0]), int(span[1])

                        scope_frag_match = re.finditer(rf'\b(?<!\-){re_term}\b', neg['Scope'][0][i])

                        for sfm in scope_frag_match:
                            if sfm.span(0)[0] + start == m.span(0)[0] and sfm.span(0)[1] + start == m.span(0)[1]: 
                                #negated_occurrences_in_ngsbnc[term].append((sent['sent_id'], m.group(0), m.span(0)))
                                all_occurrences_boolean_map[term][sent['sent_id']].append((m.span(0), True))
                                not_matching_any_scope_fragment = False 
                                '''
                                print("\nNEGATED MATCH for", term, "in", sent['sent_id'])
                                print("sfm.span(0)[0]", sfm.span(0)[0])
                                print("sfm.span(0)[0] + start", sfm.span(0)[0] + start)
                                print("m.span(0)[0]", m.span(0)[0])
                                print("sfm.span(0)[1]", sfm.span(0)[1])
                                print("sfm.span(0)[1] + start", sfm.span(0)[1] + start)
                                print("m.span(0)[1]", m.span(0)[1])
                                print()
                                '''
                            #else:
                            #    print("No match here")
    
                if not_matching_any_scope_fragment:
                    #non_negated_occurrences_in_ngsbnc[term].append((sent['sent_id'], m.group(0), m.span(0)))
                    all_occurrences_boolean_map[term][sent['sent_id']].append((m.span(0), False))
        #print()
    
    print("LOOK AT PREDS")
    all_pred_terms_boolean_map = defaultdict(lambda: defaultdict(lambda: [])) # term -> sent_id -> list((span, bool) for each occurrence)
    # Count terms (negated and non negated) in pred
    for term in all_occurrences_boolean_map: # loop through the instances in gold corpus and check if they are predicted as negated or not in pred
        re_term = term.replace("-", "\-")
        re_term = re_term.replace("*", "\*")
        
        for i in range(len(p_sents)): 
            sent_id = p_sents[i][0][1]
            #print("sent_id:", sent_id)

            # Find cue indices
            cue_indices = [j for j in range(7, len(p_sents[i][0])) if len(p_sents[i][0]) > 8 and (j-7) % 3 == 0]
            #print(cue_indices)

            #full_sent = " ".join([p_sents[i][j][3] for j in range(len(p_sents[i]))])
            full_sent = g_sents[i]['text'] # Something wrong with the number of spaces when joining on a single space? 

            reg = r'\S+'

            matches = re.finditer(reg, full_sent)
            word_idx_to_span = {wi:ws for wi,ws in zip([w_i for w_i in range(len(full_sent.split()))],[(m.span(0)[0],m.span(0)[1]) for m in matches])}

            start_i_to_word_idx = {v[0]:k for k,v in word_idx_to_span.items()} #word start index to word index 

            #print(full_sent)

            pred_scopes = []
            #...
            for c_i in cue_indices: 
                scope_idx = c_i+1 
                pred_scopes.append([p_sents[i][j][scope_idx] for j in range(len(p_sents[i]))])
            #print(pred_scopes)               


            for span, is_neg in all_occurrences_boolean_map[term][sent_id]: 

                NOT_PRESENT = -1
                start_w_idx = start_i_to_word_idx.get(span[0], NOT_PRESENT)         
                print("DEBUG", full_sent[span[0]:span[1]])
                print(full_sent)
                assert start_w_idx != NOT_PRESENT # Should not happen when we look at full words only 

                not_matching_any_scope = True
                is_match = None # debug
                for pred_scope in pred_scopes: 
                    # TODO: Only count once if term is inside multiple scopes
                    term_len = len(term.split()) # number of words in term 
                   
                    is_match = True
                    for p_i in range(len(pred_scope[start_w_idx:(start_w_idx)+term_len])): 
                        if not pred_scope[start_w_idx:(start_w_idx)+term_len][p_i].lower() == term.split()[p_i].lower():
                            is_match = False
                            break
                    if is_match:
                        if not term in terms_pred: # The model can only discover custom terms present in dev (train+devtest)
                            all_pred_terms_boolean_map[term][sent_id].append((span, False))
                        else: # the term is in terms_pred
                            all_pred_terms_boolean_map[term][sent_id].append((span, True))
                        not_matching_any_scope = False
                        break
                
                if not_matching_any_scope: 

                    all_pred_terms_boolean_map[term][sent_id].append((span, False))

    print("\nNumber of terms before greedy - GOLD:")
    c_before_g = 0
    for t in all_occurrences_boolean_map:
        for sid in all_occurrences_boolean_map[t]: 
            c_before_g += len(all_occurrences_boolean_map[t][sid])
    print(c_before_g)
    print("\nNumber of terms before greedy - PRED:")
    c_before_p = 0
    for t in all_pred_terms_boolean_map:
        for sid in all_pred_terms_boolean_map[t]: 
            c_before_p += len(all_pred_terms_boolean_map[t][sid])
    print(c_before_p)


    # Take a greedy approach when multiple terms are matched in the same location - e.g. "bivirkninger" and "økte bivirkninger" -> only count the longest one 
    print("\nGREEDY APPROACH - GOLD")
    for term in overlapping_terms: 
        # For gold: 
        for sid in all_occurrences_boolean_map[term]:
            term_occurrences = all_occurrences_boolean_map[term][sid]
            idx_to_pop = []
            term_spans = [s for s,_ in term_occurrences]
            for o_term in overlapping_terms[term]:
                o_term_occurrences = all_occurrences_boolean_map[o_term][sid]
                o_term_spans = [s for s,_ in o_term_occurrences]
                # If a span in o_term_spans overlaps with a span in term_spans: remove corresponding element in term_occurrences
                for start, end in o_term_spans:
                    for i in range(len(term_spans)): 
                        ts_start, ts_end = term_spans[i][0], term_spans[i][1]
                        if ts_start >= start and ts_end <= end: 
                            idx_to_pop.append(i)
                            #print("term:", term, "o_term:", o_term)

            all_occurrences_boolean_map[term][sid] = [term_occurrences[i] for i in range(len(term_occurrences)) if i not in idx_to_pop]

    print("\nGREEDY APPROACH - PRED")
    for term in overlapping_terms:
        # Do the same as above for preds: 
        for sid in all_pred_terms_boolean_map[term]:
            term_occurrences = all_pred_terms_boolean_map[term][sid]
            idx_to_pop = []
            term_spans = [s for s,_ in term_occurrences]
            for o_term in overlapping_terms[term]:
                o_term_occurrences = all_pred_terms_boolean_map[o_term][sid]
                o_term_spans = [s for s,_ in o_term_occurrences]
                # If a span in o_term_spans overlaps with a span in term_spans: remove corresponding element in term_occurrences
                for start, end in o_term_spans:
                    for i in range(len(term_spans)): 
                        ts_start, ts_end = term_spans[i][0], term_spans[i][1]
                        if ts_start >= start and ts_end <= end: 
                            idx_to_pop.append(i)
                            #print("term:", term, "o_term:", o_term)

            
            all_pred_terms_boolean_map[term][sid] = [term_occurrences[i] for i in range(len(term_occurrences)) if i not in idx_to_pop]

    print("\nNumber of terms after greedy - GOLD:")
    c_after_g = 0
    for t in all_occurrences_boolean_map:
        for sid in all_occurrences_boolean_map[t]: 
            c_after_g += len(all_occurrences_boolean_map[t][sid])
    print(c_after_g)
    print("\nNumber of terms after greedy - PRED:")
    c_after_p = 0
    for t in all_pred_terms_boolean_map:
        for sid in all_pred_terms_boolean_map[t]: 
            c_after_p += len(all_pred_terms_boolean_map[t][sid])
    print(c_after_p)


    # PRINT RESULT FOR GOLD
    print("\n----- GOLD -----")
    print("\nBEFORE REMOVAL OF DUPLICATES")
    all_occurrences = []
    for k in all_occurrences_boolean_map:
        for sid in all_occurrences_boolean_map[k]:
            for occ in all_occurrences_boolean_map[k][sid]:
                all_occurrences.append(occ)
    print(len(all_occurrences))
    print()


    for k in all_occurrences_boolean_map: 
        for sid in all_occurrences_boolean_map[k]: 
            all_occurrences_boolean_map[k][sid] = sorted(list(set([occ for occ in all_occurrences_boolean_map[k][sid]])))


    print("\nAFTER REMOVAL OF DUPLICATES")
    all_occurrences = []
    for k in all_occurrences_boolean_map:
        for sid in all_occurrences_boolean_map[k]:
            for occ in all_occurrences_boolean_map[k][sid]:
                all_occurrences.append(occ)
    print(len(all_occurrences))
    print()


    ########################################
    # Write gold standard terms to file: 
    c = 0
    with open(args.output_file, 'w', encoding='utf8') as f:
        for t in all_occurrences_boolean_map: 
            for sent_id in all_occurrences_boolean_map[t]:
                for span, inscope in all_occurrences_boolean_map[t][sent_id]:
                    if inscope: 
                        f.write(f"\nt: {t}, SPAN: {span}\n")
                        sentence = [s['text'] for s in g_sents if s['sent_id'] == sent_id]
                        assert len(sentence) == 1
                        f.write(f"{sent_id}: {sentence[0]}\n")
                        c += 1
        
        f.write(f"\n\nTOTAL TERMS INSIDE GOLD NEGATION SCOPES: {c}\n")

    print(f"GOLD TERMS WRITTEN TO FILE {args.output_file}")

    ########################################


    # PRINT RESULT FOR PRED
    print("\n----- PREDS -----")
    print("\nBEFORE REMOVAL OF DUPLICATES")
    all_occurrences = []
    for k in all_pred_terms_boolean_map:
        for sid in all_pred_terms_boolean_map[k]:
            for occ in all_pred_terms_boolean_map[k][sid]:
                all_occurrences.append(occ)
    print(len(all_occurrences))
    print()


    for k in all_pred_terms_boolean_map: 
        for sid in all_pred_terms_boolean_map[k]: 
            all_pred_terms_boolean_map[k][sid] = sorted(list(set([occ for occ in all_pred_terms_boolean_map[k][sid]])))


    print("\nAFTER REMOVAL OF DUPLICATES")
    all_occurrences = []
    for k in all_pred_terms_boolean_map:
        for sid in all_pred_terms_boolean_map[k]:
            for occ in all_pred_terms_boolean_map[k][sid]:
                all_occurrences.append(occ)
    print(len(all_occurrences))
    print()


    sanity_check(all_occurrences_boolean_map, all_pred_terms_boolean_map)
    #sanity_check_no_sorting(all_occurrences_boolean_map, all_pred_terms_boolean_map)

    # COMPUTE F1
    true_pos, tp_instances = tp(all_occurrences_boolean_map, all_pred_terms_boolean_map)
    true_neg, tn_instances = tn(all_occurrences_boolean_map, all_pred_terms_boolean_map)
    false_pos, fp_instances = fp(all_occurrences_boolean_map, all_pred_terms_boolean_map)
    false_neg, fn_instances = fn(all_occurrences_boolean_map, all_pred_terms_boolean_map)
    print()
    print("TP:", true_pos)
    print("TN:", true_neg)
    print("FP:", false_pos)
    print("FN:", false_neg)

    prec = precision(true_pos, false_pos)
    rec = recall(true_pos, false_neg)
    print("Precision:", prec)
    print("Recall:", rec)

    f1_score = f1(prec, rec)

    print("F1 score:", f1_score)

    with open(f'{args.pred[:-4]}_RESULTS_marie_gold_{args.testset_type}.txt', 'w', encoding="utf8") as res:
        res.write(f"----- ALL TP -----\n")
        res.write("\n".join([str(i) for i in tp_instances]))
        res.write("\n\n")
        res.write(f"----- ALL TN -----\n")
        res.write("\n".join([str(i) for i in tn_instances]))
        res.write("\n\n")
        res.write(f"----- ALL FP -----\n")
        res.write("\n".join([str(i) for i in fp_instances]))
        res.write("\n\n")
        res.write(f"----- ALL FN -----\n")
        res.write("\n".join([str(i) for i in fn_instances]))
        res.write("\n\n")
        res.write(f"Gold: {args.gold}\n")
        res.write(f"Pred: {args.pred}\n")
        res.write(f"Test ids: {args.testset_ids}\n")
        res.write(f"TP: {true_pos}\n")
        res.write(f"TN: {true_neg}\n")
        res.write(f"FP: {false_pos}\n")
        res.write(f"FN: {false_neg}\n")
        res.write(f"Precision: {prec}\n")
        res.write(f"Recall: {rec}\n")
        res.write(f"F1: {f1_score}\n")


