#!/usr/bin/env python

# Courtney Napoles
# <courtneyn@jhu.edu>
# 21 June 2015
# ##
# compute_gleu
# 
# This script calls gleu.py to calculate the GLEU score of a sentence, as
# described in our ACL 2015 paper, Ground Truth for Grammatical Error
# Correction Metrics by Courtney Napoles, Keisuke Sakaguchi, Matt Post,
# and Joel Tetreault.
# 
# For instructions on how to get the GLEU score, call "compute_gleu -h"
#
# Updated 2 May 2016: This is an updated version of GLEU that has been
# modified to handle multiple references more fairly.
#
# This script was adapted from compute-bleu by Adam Lopez.
# <https://github.com/alopez/en600.468/blob/master/reranker/>

import argparse
import sys
import os
import scipy.stats
import numpy as np
import random
import math
from collections import Counter
from typing import List

class GLEU :

    def __init__(self, n=4) :
        self.order = 4

    def load_hypothesis_sentence(self,hypothesis) :
        self.hlen = len(hypothesis)
        self.this_h_ngrams = [ self.get_ngram_counts(hypothesis,n)
                               for n in range(1,self.order+1) ]

    def load_sources(self,spath) :
        self.all_s_ngrams = [ [ self.get_ngram_counts(line.split(),n)
                                for n in range(1,self.order+1) ]
                              for line in open(spath, encoding='utf8') ]

    def load_sources_from_list(self, sources: List[List[str]]):
        self.all_s_ngrams = [[self.get_ngram_counts(line, n)
                              for n in range(1, self.order+1)]
                             for line in sources]

    def load_references(self,rpaths) :
        self.refs = [ [] for i in range(len(self.all_s_ngrams)) ]
        self.rlens = [ [] for i in range(len(self.all_s_ngrams)) ]
        for rpath in rpaths :
            for i,line in enumerate(open(rpath, encoding='utf8')) :
                self.refs[i].append(line.split())
                self.rlens[i].append(len(line.split()))

        # count number of references each n-gram appear sin
        self.all_rngrams_freq = [ Counter() for i in range(self.order) ]

        self.all_r_ngrams = [ ]
        for refset in self.refs :
            all_ngrams = []
            self.all_r_ngrams.append(all_ngrams)

            for n in range(1,self.order+1) :
                ngrams = self.get_ngram_counts(refset[0],n)
                all_ngrams.append(ngrams)

                for k in ngrams.keys() :
                    self.all_rngrams_freq[n-1][k]+=1

                for ref in refset[1:] :
                    new_ngrams = self.get_ngram_counts(ref,n)
                    for nn in new_ngrams.elements() :
                        if new_ngrams[nn] > ngrams.get(nn,0) :
                            ngrams[nn] = new_ngrams[nn]

    def load_references_from_list(self, references: List[List[str]]):
        self.refs = [ [] for i in range(len(self.all_s_ngrams)) ]
        self.rlens = [ [] for i in range(len(self.all_s_ngrams)) ]
        for i,line in enumerate(references) :
            self.refs[i].append(line)
            self.rlens[i].append(len(line))

        # count number of references each n-gram appear sin
        self.all_rngrams_freq = [ Counter() for i in range(self.order) ]

        self.all_r_ngrams = [ ]
        for refset in self.refs :
            all_ngrams = []
            self.all_r_ngrams.append(all_ngrams)

            for n in range(1,self.order+1) :
                ngrams = self.get_ngram_counts(refset[0],n)
                all_ngrams.append(ngrams)

                for k in ngrams.keys() :
                    self.all_rngrams_freq[n-1][k]+=1

                for ref in refset[1:] :
                    new_ngrams = self.get_ngram_counts(ref,n)
                    for nn in new_ngrams.elements() :
                        if new_ngrams[nn] > ngrams.get(nn,0) :
                            ngrams[nn] = new_ngrams[nn]

    def get_ngram_counts(self,sentence,n) :
        return Counter([tuple(sentence[i:i+n])
                        for i in range(len(sentence)+1-n)])

    # returns ngrams in a but not in b
    def get_ngram_diff(self,a,b) :
        diff = Counter(a)
        for k in (set(a) & set(b)) :
            del diff[k]
        return diff

    def normalization(self,ngram,n) :
        return 1.0*self.all_rngrams_freq[n-1][ngram]/len(self.rlens[0])

    # Collect BLEU-relevant statistics for a single hypothesis/reference pair.
    # Return value is a generator yielding:
    # (c, r, numerator1, denominator1, ... numerator4, denominator4)
    # Summing the columns across calls to this function on an entire corpus
    # will produce a vector of statistics that can be used to compute GLEU
    def gleu_stats(self,i,r_ind=None):

        hlen = self.hlen
        rlen = self.rlens[i][r_ind]

        yield hlen
        yield rlen

        for n in range(1,self.order+1):
            h_ngrams = self.this_h_ngrams[n-1]
            s_ngrams = self.all_s_ngrams[i][n-1]
            r_ngrams = self.get_ngram_counts(self.refs[i][r_ind],n)

            s_ngram_diff = self.get_ngram_diff(s_ngrams,r_ngrams)

            yield max([ sum( (h_ngrams & r_ngrams).values() ) - \
                        sum( (h_ngrams & s_ngram_diff).values() ), 0 ])

            yield max([hlen+1-n, 0])

    # Compute GLEU from collected statistics obtained by call(s) to gleu_stats
    def gleu(self, stats, smooth=False):
        # smooth 0 counts for sentence-level scores
        if smooth:
            stats = [s if s != 0 else 1 for s in stats]
        if len(list(filter(lambda x: x==0, stats))) > 0:
            return 0
        (c, r) = stats[:2]
        log_gleu_prec = sum([math.log(float(x)/y)
                             for x,y in zip(stats[2::2],stats[3::2])]) / 4
        return math.exp(min([0, 1-float(r)/c]) + log_gleu_prec)

def split(comment: List[str]):
    comment = " ".join(comment).replace(" <con> ,", " ,").replace(" <con> #", " #").replace(" <con> (", " (") \
        .replace("( <con> ", "( ").replace(" <con> )", " )").replace(") <con> ", ") ").replace(" <con> {", " {") \
        .replace(" <con> }", " }").replace(" <con> @", " @").replace("# <con> ", "# ").replace(" <con> ", "") \
        .strip(' !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~_')
    return comment.lower().split(" ")


def get_gleu_stats(scores) :
    mean = np.mean(scores)
    std = np.std(scores)
    ci = scipy.stats.norm.interval(0.95,loc=mean,scale=std)
    return ['%f'%mean,
            '%f'%std,
            '(%.3f,%.3f)'%(ci[0], ci[1])]

def calcGleu(sources: List[List[str]], references: List[List[str]], hypothesises: List[List[str]], maxN=4, lowercase=False):
    for i, hypo in enumerate(hypothesises):
        if hypo is None:
            hypothesises[i] = sources[i]
    if lowercase is True:
        for i, src in enumerate(sources):
            sources[i] = split([x.lower() for x in src])
        for i, ref in enumerate(references):
            references[i] = split([x.lower() for x in ref])
        for i, hypo in enumerate(hypothesises):
            hypothesises[i] = split([x.lower() for x in hypo])
    gleu_calculator = GLEU(maxN)
    gleu_calculator.load_sources_from_list(sources)
    gleu_calculator.load_references_from_list(references)
    indices = []
    hyp = hypothesises
    num_iterations = 1
    for j in range(num_iterations):
        random.seed(j*101)
        indices.append([random.randint(0, 0)
                        for i in range(len(hyp))])

    iter_stats = [[0 for i in range(2*maxN+2)]
                   for j in range(num_iterations)]

    for i, h in enumerate(hyp):

        gleu_calculator.load_hypothesis_sentence(h)
        # we are going to store the score of this sentence for each ref
        # so we don't have to recalculate them 500 times

        stats_by_ref = [ None for r in range(1)]

        for j in range(num_iterations):
            ref = indices[j][i]
            this_stats = stats_by_ref[ref]

            if this_stats is None :
                this_stats = [ s for s in gleu_calculator.gleu_stats(
                    i,r_ind=ref) ]
                stats_by_ref[ref] = this_stats

            iter_stats[j] = [ sum(scores)
                              for scores in zip(iter_stats[j], this_stats)]
    return float(get_gleu_stats([gleu_calculator.gleu(stats) for stats in iter_stats])[0])


if __name__ == '__main__' :
    with open('./src.txt', encoding='utf8') as f:
        src = [x.strip().split() for x in f.readlines()]
    with open('./ref.txt', encoding='utf8') as f:
        ref = [x.strip().split() for x in f.readlines()]
    with open('./hypo.txt', encoding='utf8') as f:
        hypo = [x.strip().split() for x in f.readlines()]
    calcGleu(src, ref, hypo)

