
import numpy as np
from itertools import chain


def get_precision_recall(label, pred):
    if len(label) == 0:
        return 0
    inter = set(pred).intersection(set(label))
    precision = min(1, len(inter)/len(pred))
    recall = min(1, len(inter)/len(label))
    return precision, recall


def get_ndcg(label, pred, top_k):
    if len(label) == 0:
        return 0
    if top_k > len(label):
        top_k = len(label)
    score = 0.0
    for i, p in enumerate(pred):
        if p in label and p not in pred[:i]:
            score += 1.0 / np.log2(i + 2.0)
    return score/min(len(label), top_k)
