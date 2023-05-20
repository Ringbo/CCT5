# coding:utf-8

import argparse
import json
import re
from typing import List
import os
import sys
sys.path.append(os.getcwd())
from src.evaluator.smooth_bleu import bleu_fromstr
from src.evaluator.compute_gleu_cup import calcGleu

stop_words = {}  # 1004
connectOp = {'.', '<con>'}
symbol = {"{", "}", ":", ",", "_", ".", "-", "+", ";", "<con>"}
stripAll = re.compile('[\s]+')


def stripAllSymbol(x): return re.sub(
    "[~!@#$%^&*()_\+\-\=\[\]\{\}\|;:\'\"<,>.?/]", '', x)


def formatString(string):
    string = "".join([x for x in string if x.isalnum() or x == ' '])
    string = " ".join([x for x in string.split(' ') if x.isalnum()])
    string = stripAll.sub('', string.lower())
    return string


def calcAccuracy(reference_strings, predicted_strings):
    assert(len(reference_strings) == len(predicted_strings))
    correct = 0.0
    idx_rec = []
    for i in range(len(reference_strings)):
        if formatString(reference_strings[i]) == formatString(predicted_strings[i]):
            correct += 1
            idx_rec.append(i)
    return 100 * correct/float(len(reference_strings))


def split(comment: List[str]):
    comment = " ".join(comment).replace(" <con> ,", " ,").replace(" <con> #", " #").replace(" <con> (", " (") \
        .replace("( <con> ", "( ").replace(" <con> )", " )").replace(") <con> ", ") ").replace(" <con> {", " {") \
        .replace(" <con> }", " }").replace(" <con> @", " @").replace("# <con> ", "# ").replace(" <con> ", "") \
        .strip(' !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~_')
    return comment.split(" ")


def calcBleu(preds, refs, rmstop):
    preds = [" ".join(x) for x in preds]
    refs = [" ".join(x) for x in refs]
    score = bleu_fromstr(preds, refs, rmstop=rmstop)
    return score


def computeMetrics(src_instances: List[List[str]], references: List[List[str]], pred_instances: List[List[str]]):
    gleu = calcGleu(src_instances, references, pred_instances, lowercase=True)
    bleu = calcBleu(pred_instances, references, rmstop=False)
    acc = calcAccuracy(pred_instances, references)
    metrics = {
        "gleu": gleu,
        "bleu": bleu,
        "acc": acc
    }
    return metrics
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", type=str, default='./results/CommentUpdate/CCT5Res_for_eval.json')
    args = parser.parse_args()
    with open(args.filepath, 'r', encoding='utf8') as f:
        tmp = json.load(f)
        pred_instances, references, src_instances = tmp
        references = [x[0] for x in references]
        src_instances = [x[0] for x in src_instances]
        
    print(computeMetrics(src_instances, references, pred_instances))
