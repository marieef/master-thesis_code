def f1_cues(y_true, y_pred):
    '''From NegBERT code:
    https://github.com/adityak6798/Transformers-For-Negation-and-Speculation/blob/master/Transformers_for_Negation_and_Speculation.ipynb,
    with slight modifications (see comments).
    
    Needs flattened cues.
    '''

    #print("y_true", y_true)
    #print("y_pred", y_pred)

    tp = sum([1 for i,j in zip(y_true, y_pred) if (i==j and i!=3)]) # i is never pad
    fp = sum([1 for i,j in zip(y_true, y_pred) if ((j!=3 and j!=4) and i==3)]) # Added "and j!=4" to only count actual cue predictions (in the place of a not-cue) as false positives
    fn = sum([1 for i,j in zip(y_true, y_pred) if (i!=3 and (j==3 or j==4))]) # Added the part after or to 
    if tp==0:
        prec = 0.0001
        rec = 0.0001
    else:
        prec = tp/(tp+fp)
        rec = tp/(tp+fn)

    print("TP", tp)
    print("FP", fp)
    print("FN", fn)

    print(f"Precision: {prec}")
    print(f"Recall: {rec}")
    print(f"F1 Score: {2*prec*rec/(prec+rec)}")
    return prec, rec, 2*prec*rec/(prec+rec)
