import numpy as np

def semantic_segm_f1_score(true_mask, false_mask):
    tp = np.count_nonzero(true_mask)
    fn = len(true_mask) - tp
    tn = np.count_nonzero(false_mask)
    fp = len(false_mask) - tn
    f1_score = tp/(tp + ((fp+fn)/2))
    return f1_score

def semantic_segm_recall(true_mask):
    tp = np.count_nonzero(true_mask)
    fn = len(true_mask) - tp
    recall = tp/(tp+fn)
    return recall