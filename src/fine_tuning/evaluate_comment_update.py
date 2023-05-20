# coding:utf-8


from ntpath import join
import os
import sys
import json
import re
from tqdm import tqdm
from Levenshtein import distance
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from nltk.corpus import stopwords
# stop_words = set(stopwords.words('english'))  # 1002
# stop_words = {}  # 1004
from typing import List,Dict
# from tfidf_utils import *
import javalang
from difflib import SequenceMatcher
from match import match, match_token, equal
from eval import EditDistance
from collections import defaultdict
from functools import cmp_to_key
from copy import deepcopy
from utils.compute_gleu_cup import calGleu
from utils.tokenizer import Tokenizer
from utils.smooth_bleu import bleu_fromstr

stop_words = {}  # 1004
connectOp = {'.', '<con>'}
symbol = {"{","}",":",",","_",".","-","+",";","<con>"}
stripAll = re.compile('[\s]+')
stripAllSymbol = lambda x: re.sub("[~!@#$%^&*()_\+\-\=\[\]\{\}\|;:\'\"<,>.?/]",'', x)


def formatString_1(string):
    string = "".join([x for x in string if x.isalnum() or x == ' '])
    string = " ".join([x for x in string.split(' ') if x.isalnum()])
    string = stripAll.sub('', string.lower())
    return string


def compute_accuracy(reference_strings, predicted_strings):
    assert(len(reference_strings) == len(predicted_strings))
    correct = 0.0
    idx_rec = []
    for i in range(len(reference_strings)):
        if formatString_1(reference_strings[i]) == formatString_1(predicted_strings[i]):
            correct += 1
            idx_rec.append(i)
            # print(reference_strings[i])
        #     if formatString_1(reference_strings[i]) != formatString_1(predicted_strings[i]):
        #         print("")
        # if formatString_2(reference_strings[i]) != formatString_2(predicted_strings[i]):
        #     if formatString_1(reference_strings[i]) == formatString_1(predicted_strings[i]):
        #         print("")
    return 100 * correct/float(len(reference_strings))


def lookBack(code_change_seq):
    def removeDup(words):
        temp = set()
        for word in words:
            word = [x.lower() for x in word]
            word.reverse()
            temp.add("|".join(word))

        temp = list(temp)
        temp = [[y for y in x.split('|') if y != ''] for x in temp]  # remove ''
        return [x for x in temp if x.__len__() != 0]

    def itemIsConnect(item):
        if item[0] in connectOp or item[1] in connectOp:
            return True
        else:
            return False

    def getSubset(words):
        subwords = [tuple(word) for word in words]
        for word in words:
            for i in range(len(word)):
                for j in range(i, len(word) + 1):
                    temp = word[i:j]
                    if temp.__len__() == 0:
                        continue
                    if temp[0] in symbol:
                        temp.pop(0)
                    if temp.__len__() == 0:
                        continue
                    if temp[-1] in symbol:
                        temp.pop(-1)
                    subwords.append(tuple(temp)) if temp.__len__() != 0 else None

        # subwords = [x.replace("<con>", '') for x in subwords]
        subwords = [x for x in subwords if x.__len__() != 0 and x != tuple('.')]
        # subwords = [x.strip('.').strip('\"') for x in subwords if x != '' and x not in symbol]
        return set(subwords)

    def combineTuple(mixedTuple):
        res = tuple()
        for x in mixedTuple:
            if isinstance(x, tuple):
                res += x
            else:
                res += tuple((x,))

        if res.__len__() and res[0] in connectOp:
            res = tuple(res[1:])
        if res.__len__() and res[-1] in connectOp:
            res = tuple(res[:-1])
        return res

    def getSubsetMapping(modifiedMapping):
        tempMapping = deepcopy(modifiedMapping)
        for buggyWord in tempMapping:
            for fixedWord in tempMapping[buggyWord]:
                if buggyWord.__len__() == fixedWord.__len__():
                    for i in range(buggyWord.__len__()):
                        for j in range(i + 1, buggyWord.__len__() + 1):
                            if buggyWord[i:j][0] not in connectOp and buggyWord[i:j][-1] not in connectOp \
                                    and fixedWord[i:j][0] not in connectOp and fixedWord[i:j][-1] not in connectOp \
                                    and buggyWord[i:j] != fixedWord[i:j]:
                                modifiedMapping[tuple(buggyWord[i:j])].add(tuple(fixedWord[i:j]))
                else:
                    tempBuggy = list(buggyWord)
                    tempFixed = list(fixedWord)

                    '''
                    Find different part
                                    (pop ->)___________x___(<- pop)
                                    (pop ->)___________xx___(<- pop)
                    '''
                    left_i, left_j, right_i, right_j = 0, 0, tempBuggy.__len__() - 1, tempFixed.__len__() - 1
                    while left_i < tempBuggy.__len__() and left_j < tempFixed.__len__():
                        if tempBuggy[left_i].lower() == tempFixed[left_i].lower():
                            left_i += 1
                            left_j += 1
                        else:
                            left_i = max(0, left_i - 1)
                            left_j = max(0, left_j - 1)
                            break
                    if left_i == tempBuggy.__len__() or left_j == tempFixed.__len__():
                        left_i = max(0, left_i - 1)
                        left_j = max(0, left_j - 1)

                    while right_i >= left_i and right_j >= left_j:
                        if tempBuggy[right_i].lower() == tempFixed[right_j].lower():
                            right_i -= 1
                            right_j -= 1
                        else:
                            right_i += 1
                            right_j += 1
                            break
                    if right_i < 0 or right_j < 0:
                        return modifiedMapping
                    # modifiedMapping[tuple(tempBuggy[left_i:right_i])] = tuple(tempFixed[left_j:right_j])
                    alignedBuggy = tempBuggy[:left_i] + [tuple(tempBuggy[left_i:right_i + 1])] + tempBuggy[right_i + 1:]
                    alignedFixed = tempFixed[:left_j] + [tuple(tempFixed[left_j:right_j + 1])] + tempFixed[right_j + 1:]

                    for i in range(alignedBuggy.__len__()):
                        for j in range(i + 1, alignedFixed.__len__() + 1):
                            key = combineTuple(alignedBuggy[i:j])
                            value = combineTuple(alignedFixed[i:j])
                            if key != value and key.__len__() != 0 and value.__len__() != 0:
                                modifiedMapping[key].add(value)
        return modifiedMapping

    buggyWords = []
    fixedWords = []
    allIndex = []
    lastItem = ['', '', 'equal']
    preHasValidOp = False
    modifiedMapping = defaultdict(set)
    for i, x in enumerate(code_change_seq):
        if x[2] != 'equal':
            allIndex.append(i)
            preHasValidOp = True
        elif (itemIsConnect(lastItem) or itemIsConnect(x)) and preHasValidOp:
            allIndex.append(i)
        else:
            preHasValidOp = False
        lastItem = x

    for i, index in enumerate(allIndex):
        connectFlag = False
        lastItem = code_change_seq[index]
        reversedSeq = list(reversed(code_change_seq[:index]))
        curBuggyWords = []
        curFixedWords = []
        for j, seq in enumerate(reversedSeq):
            if j < index and reversedSeq[j][0] in connectOp or connectFlag:
                curBuggyWords.append(lastItem[0]) if not curBuggyWords.__len__() else None
                curBuggyWords.append(reversedSeq[j][0])
                connectFlag = True

            if j < index and reversedSeq[j][1] in connectOp or connectFlag:
                curFixedWords.append(lastItem[1]) if not curFixedWords.__len__() else None
                curFixedWords.append(reversedSeq[j][1])
                connectFlag = True
            if j < index and reversedSeq[j][0] not in connectOp and reversedSeq[j][1] not in connectOp:
                if connectFlag is False:
                    break
                connectFlag = False
        buggyWords.append(tuple(reversed(tuple(x for x in curBuggyWords if x!=''))))
        fixedWords.append(tuple(reversed(tuple(x for x in curFixedWords if x!=''))))
        if buggyWords[-1].__len__() != 0 and fixedWords[-1].__len__() != 0:
            modifiedMapping[buggyWords[-1]].add(fixedWords[-1])
        if code_change_seq[index][2] == 'replace' and code_change_seq[index][0] not in symbol and code_change_seq[index][1] not in symbol:
            modifiedMapping[tuple((code_change_seq[index][0],))].add(tuple((code_change_seq[index][1],)))

    modifiedMapping = getSubsetMapping(modifiedMapping)
    # return getSubset(buggyWords), getSubset(fixedWords)
    return modifiedMapping


def getPossibleWords(fileInfo):
    codeSeq = fileInfo["code_change_seq"]
    buggyStream = []
    fixedStream = []
    changed = set()
    for x in codeSeq:
        buggyStream.append(x[0])
        fixedStream.append(x[1])
        if x[2] != "equal":
            changed.add(x[0].lower()) if x[0] != '' and x[0] != '<con>' and x[0].isalpha() and x[
                0] not in stop_words else None
            changed.add(x[1].lower()) if x[1] != '' and x[1] != '<con>' and x[1].isalpha() and x[
                1] not in stop_words else None

    possibleConWords = lookBack(fileInfo["code_change_seq"])

    return changed | possibleConWords[0] | possibleConWords[1]


def getTokenStream(fileInfo):
    if "code_change_seq" not in fileInfo:
        return False
    codeSeq = fileInfo["code_change_seq"]
    buggyStream = []
    fixedStream = []
    changed = set()
    for x in codeSeq:
        buggyStream.append(x[0])
        fixedStream.append(x[1])
        if x[2] != "equal":
            changed.add(x[0].lower()) if x[0] != '' and x[0] != '<con>' and x[0].isalpha() and x[0] not in stop_words else None
            changed.add(x[1].lower()) if x[1] != '' and x[1] != '<con>' and x[1].isalpha() and x[1] not in stop_words else None
    buggyStream = [x.lower() for x in buggyStream if x != '' and x !='<con>' and x not in stop_words]
    fixedStream = [x.lower() for x in fixedStream if x != ''and x != '<con>' and x not in stop_words]
    oldComment = [x for x in fileInfo["src_desc_tokens"] if x != '']
    newComment = [x for x in fileInfo["dst_desc_tokens"] if x != '']
    return buggyStream, fixedStream, oldComment, newComment, changed



def sortMapping(streamPair):
    modifiedMapping = streamPair[5]
    possibleMapping = []
    for x in modifiedMapping:
        modifiedMapping[x] = list(modifiedMapping[x])
        modifiedMapping[x].sort(key=lambda x:x.__len__(), reverse=True)
        possibleMapping.append((x,modifiedMapping[x]))
    possibleMapping.sort(key=lambda x: x[0].__len__(), reverse=True)
    return possibleMapping

def evaluateCorrectness(possibleMapping, streamPair, k=1):

    def isEqual(pred, oracle):
        predStr = stripAll.sub(' ', " ".join(pred).replace("<con>", '')).strip(' !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~_')
        oracleStr = stripAll.sub(' ', " ".join(oracle).replace("<con>", '')).strip(' !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~_')
        if predStr.lower() == oracleStr.lower():
            return True
        else:
            return False

    def tryAllPossible(possibleMapping,streamPair,matchLevel, k):
        cnt = 0
        for x in possibleMapping:
            oldComment = streamPair[2]
            newComment = streamPair[3]
            if cnt >= k:
                break
            pattern = [x.lower() for x in x[0]]
            # index = KMP([x.lower() for x in oldComment], pattern)
            indexes = match(oldComment, pattern, matchLevel)
            if indexes == []:
                continue
            else:
                bias = 0
                for index in indexes:
                    predComment = oldComment[:index + bias] + list(x[1][0]) + oldComment[index + pattern.__len__() + bias:]
                    oldComment = predComment
                    bias = bias + x[1][0].__len__() - x[0].__len__()
                if isEqual(predComment, newComment):
                    return True
            cnt += 1
        if cnt == 0:
            return None
        else:
            return False

    for i in range(3):
        matchRes = tryAllPossible(possibleMapping, streamPair, matchLevel=i, k=k)
        if matchRes is None:
            continue
        elif matchRes is True:
            return True
        else:
            return False
    return None


def split(comment: List[str]):
    comment = " ".join(comment).replace(" <con> ,", " ,").replace(" <con> #", " #").replace(" <con> (", " (") \
        .replace("( <con> ", "( ").replace(" <con> )", " )").replace(") <con> ", ") ").replace(" <con> {", " {") \
        .replace(" <con> }", " }").replace(" <con> @", " @").replace("# <con> ", "# ").replace(" <con> ", "") \
        .strip(' !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~_')
    return comment.split(" ")

def evaluateCorrectness_test(possibleMapping, streamPair, k=1):

    def genAllpossible(pred):
        allCur = [[]]
        if pred is None:
            return []
        for x in pred:
            tepAllCur = allCur.copy()
            for i in range(allCur.__len__()):
                if isinstance(x, str):
                    tepAllCur[i].append(x)
                elif isinstance(x, list):
                    cur = tepAllCur[i].copy()
                    tepAllCur[i] = None
                    for dst in x:
                        tepAllCur.append(cur + list(dst))
            allCur = [x for x in tepAllCur if x is not None]
        return allCur

    def commentTokenizer(comment):
        return re.sub("[(}).,{\[\];\n#@']"," ",comment).split(" ")

    def possibleMappingFilter(possibleMapping, oldCodeToken, newCodeToken):
        validMapping = []
        for mapping in possibleMapping:
            oldCode = "".join(oldCodeToken)
            newCode = "".join(newCodeToken)
            oldHook = "".join(mapping[0]).replace("<con>","").lower()
            newHook = "".join(mapping[1][0]).replace("<con>","").lower()
            if oldCode.replace(oldHook, newHook).lower() != newCode.lower():
                continue
            else:
                validMapping.append(mapping)
        return validMapping

    def isEqual_token(pred: List[str], oracle, k):
        if k==1 and pred:
            return Equal_1(pred[0], oracle)
        elif k > 1:
            return Equal_k(pred, oracle, k)
        else:
            return False

    def isEqual(pred, oracle):
        predStr = stripAll.sub(' ', " ".join(pred).replace("<con>", '')).strip(' !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~_')
        oracleStr = stripAll.sub(' ', " ".join(oracle).replace("<con>", '')).strip(' !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~_')
        predStr = "".join([x for x in predStr if x.isalnum()])
        oracleStr = "".join([x for x in oracleStr if x.isalnum()])
        if predStr.lower() == oracleStr.lower():
            return True
        else:
            return False

    def Equal_1(pred, oracle):
        # predStr = stripAll.sub(' ', " ".join(pred).replace("<con>", '')).strip(' !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~_').replace(' . ', '.')
        # oracleStr = stripAll.sub(' ', " ".join(oracle).replace("<con>", '')).strip(' !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~_').replace(' . ', '.')
        predStr = "".join(pred).replace("<con>", '')
        oracleStr = "".join(oracle).replace("<con>", '')
        if predStr.lower() == oracleStr.lower():
            return True
        else:
            return False

    def Equal_k(pred: List[str], oracle, k):
        pred.sort(key=lambda x:x.__len__(), reverse=True)
        pred = pred[:k]
        for x in pred:
            if Equal_1(x, oracle):
                return True
        return False

    def tryAllPossible(possibleMapping, streamPair, matchLevel, k):
        cnt = 0
        predComment_token, predComment_subtoken = None, None
        oldComment_token, oldComment_subtoken = None, None
        newComment_token = split(streamPair[3])
        newComment_subtoken = streamPair[3]
        # possibleMapping = possibleMappingFilter(possibleMapping, streamPair[0], streamPair[1]) # test
        for x in possibleMapping:
            if cnt >= 1:
                break
            if oldComment_token is None:
                oldComment_token = split(streamPair[2])
                oldComment_subtoken = streamPair[2]
            pattern_token = " ".join(x[0]).replace(" <con> ", "").replace(" . ",".")
            new_token = " ".join(x[1][0]).replace(" <con> ", "").replace(" . ",".")
            # if abs(new_token.__len__() - pattern_token.__len__()) > 15:
            #     continue
            pattern_suboten = [x.lower() for x in x[0]]
            pattern_splited = [x.lower() for x in x[0] if x !="<con>"]
            indexes_token = match_token(oldComment_token, pattern_token, matchLevel)
            indexes_subtoken = match(oldComment_subtoken, pattern_suboten, matchLevel)

            if not indexes_token:
                pass
            else:
                if equal(pattern_token, oldComment_token[indexes_token[0]],1) and not equal(pattern_token, oldComment_token[indexes_token[0]], 0):
                    if pattern_token[-1] != 's':
                        x[1][0] = tuple((x[1][0][0] + 's',))
                    else:
                        x[1][0] = tuple((x[1][0][0][:-1],))
                for index in indexes_token:
                    oldComment_token[index] = x[1]
                    predComment_token = oldComment_token
                cnt += 1
            # bias = 0
            if indexes_subtoken:
                bias = 0
                for index in indexes_subtoken:
                    predComment_subtoken = oldComment_subtoken[:index + bias] + list(x[1][0]) + oldComment_subtoken[index + pattern_suboten.__len__() + bias:]
                    oldComment_subtoken = predComment_subtoken
                    bias = bias + x[1][0].__len__() - x[0].__len__()
                cnt += 1

            '''
            Code Change: isEmptyInitCall -> isInitCall
            Comment Change: empty init -> init
            '''
            # oldComment_subtoken = streamPair[2]  # Reset the sub-tokens in old comment to avoid the replaced sub-tokens
            # in the above code.
            indexes_splited = match(oldComment_subtoken, pattern_splited, matchLevel) if pattern_splited else None
            # if indexes_splited:
            if (indexes_splited and oldComment_subtoken == streamPair[2]) \
                    or (indexes_splited and pattern_splited.__len__() > 1):  # test
                bias = 0
                for index in indexes_splited:
                    predComment_subtoken = oldComment_subtoken[:index + bias] + [y for y in list(x[1][0]) if y != "<con>"] + oldComment_subtoken[index + pattern_splited.__len__() + bias:]
                    oldComment_subtoken = predComment_subtoken
                    bias = bias + x[1][0].__len__() - x[0].__len__()
                cnt += 1

        predComment_token = genAllpossible(predComment_token)

        if predComment_token is not None and isEqual_token(predComment_token, newComment_token, k):
            return True
        elif predComment_subtoken is not None and isEqual(predComment_subtoken, newComment_subtoken):
            return True
        elif isEqual(streamPair[2], newComment_subtoken):
            return True
        if cnt == 0:
            return None
        else:
            return False

    def cmp(mapping_1, mapping_2):

        if mapping_1[0].__len__() > mapping_2[0].__len__():
            return 1
        elif mapping_1[0].__len__() < mapping_2[0].__len__():
            return -1
        elif mapping_1[0].__len__() == mapping_2[0].__len__():
            if mapping_1[2] > mapping_2[2]:
                return 1
            elif mapping_1[2] < mapping_2[2]:
                return -1
            if distance(mapping_1[0], mapping_1[1]) > distance(mapping_2[0], mapping_2[1]):
                return -1
            elif distance(mapping_1[0], mapping_1[1]) < distance(mapping_2[0], mapping_2[1]):
                return 1
            else:
                return 0

    def tryPurePossible(stremPair, mode='token'):
        if mode == 'token':
            pureMapping = genPureMapping(stremPair[6]['src_method'], stremPair[6]['dst_method'], mode='token')
            pureMapping = sorted(pureMapping, key=cmp_to_key(cmp),reverse=True)
        elif mode == 'subtoken':
            pureMapping = genPureMapping(stremPair[6]['src_method'], stremPair[6]['dst_method'], mode='subtoken')
            pureMapping = sorted(pureMapping, key=cmp_to_key(cmp),reverse=True)
        elif mode == 'all':
            pureMapping = sorted(genPureMapping(stremPair[6]['src_method'], stremPair[6]['dst_method'], mode='token'), key=cmp_to_key(cmp), reverse=True) + \
                          sorted(genPureMapping(stremPair[6]['src_method'], stremPair[6]['dst_method'], mode='subtoken'), key=cmp_to_key(cmp), reverse=True)
        oldComment_token = commentTokenizer(stremPair[6]['src_desc'])
        newComment_token = commentTokenizer(stremPair[6]['dst_desc'])
        # pureMapping.sort(key=lambda x:x[0].__len__(),reverse=True)
        predComment, newComment, oldComment = None, None, None
        if not pureMapping:
            return None
        for mapping in pureMapping:
            if mapping[0].strip() == "" or abs(mapping[1].__len__() - mapping[0].__len__()) > 20:
                continue
            oldHook = mapping[0].strip(",.\"\'") + ' '
            newHook = mapping[1].strip(",.\"\'") + ' '
            oldHook_splited =" ".join(camel_case_split(oldHook))
            newHook_splited =" ".join(camel_case_split(newHook))
            oldComment = " ".join(oldComment_token)
            newComment = " ".join(newComment_token)
            predComment = oldComment.replace(oldHook, newHook)
            if predComment == oldComment:
                predComment = oldComment.lower().replace(oldHook.lower(), newHook.lower())
                if oldHook_splited.split(" ").__len__() > 1 and predComment.lower() == oldComment.lower():
                    predComment = predComment.lower().replace(oldHook_splited.lower(), newHook_splited.lower())
            if predComment.lower() == oldComment.lower():
                continue
            else:
                break
        if predComment is None:
            return None
        elif predComment.lower().replace(" ","") == newComment.lower().replace(" ",""):
            return True
        elif predComment.lower() == oldComment.lower():
            return None
        else:
            return False

    matchRes_pure = tryPurePossible(streamPair,mode='all')
    if matchRes_pure is None:
        matchRes_pure = tryPurePossible(streamPair,mode='subtoken')
    elif matchRes_pure is True:
        return True
    else:
        return False
    #     pass

    for i in range(3):
        matchRes = tryAllPossible(possibleMapping, streamPair, matchLevel=i, k=k)
        if matchRes is True and matchRes_pure is False:
            print(streamPair[6]['sample_id'])
            tryPurePossible(streamPair, mode='all')
        if matchRes is None:
            continue
        elif matchRes is True:
            return True
        else:
            return False
    return None


def printRes(word, weight):
    for i in range(len(weight)):
        for j in range(len(word)):
            print(word[j], weight[i][j])


def getRes(possibleMapping, streamPair, k=1):
    def genAllpossible(pred):
        allCur = [[]]
        if pred is None:
            return []
        for x in pred:
            tepAllCur = allCur.copy()
            for i in range(allCur.__len__()):
                if isinstance(x, str):
                    tepAllCur[i].append(x)
                elif isinstance(x, list):
                    cur = tepAllCur[i].copy()
                    tepAllCur[i] = None
                    for dst in x:
                        tepAllCur.append(cur + list(dst))
            allCur = [x for x in tepAllCur if x is not None]
        return allCur

    def isEqual_token(pred: List[str], oracle, k):
        if k == 1 and pred:
            return Equal_1(pred[0], oracle)
        elif k > 1:
            return Equal_k(pred, oracle, k)
        else:
            return False

    def isEqual(pred, oracle):
        predStr = stripAll.sub(' ', " ".join(pred).replace("<con>", '')).strip(' !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~_')
        oracleStr = stripAll.sub(' ', " ".join(oracle).replace("<con>", '')).strip(
            ' !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~_')
        predStr = "".join([x for x in predStr if x.isalnum()])
        oracleStr = "".join([x for x in oracleStr if x.isalnum()])
        if predStr.lower() == oracleStr.lower():
            return True
        else:
            return False

    def Equal_1(pred, oracle):
        # predStr = stripAll.sub(' ', " ".join(pred).replace("<con>", '')).strip(' !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~_').replace(' . ', '.')
        # oracleStr = stripAll.sub(' ', " ".join(oracle).replace("<con>", '')).strip(' !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~_').replace(' . ', '.')
        predStr = "".join(pred).replace("<con>", '')
        oracleStr = "".join(oracle).replace("<con>", '')
        if predStr.lower() == oracleStr.lower():
            return True
        else:
            return False

    def Equal_k(pred: List[str], oracle, k):
        pred.sort(key=lambda x: x.__len__(), reverse=True)
        pred = pred[:k]
        for x in pred:
            if Equal_1(x, oracle):
                return True
        return False

    def split(comment: List[str]):
        comment = " ".join(comment).replace(" <con> ,", " ,").replace(" <con> #", " #").replace(" <con> (", " (") \
            .replace(" <con> )", " )").replace(" <con> {", " {").replace(" <con> }", " }").replace(" <con> @", " @") \
            .replace("# <con> ", "# ").replace(" <con> ", "").strip(' !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~_')
        return comment.split(" ")

    def tryAllPossible(possibleMapping, streamPair, matchLevel, k):
        cnt = 0

        predComment_token, predComment_subtoken = None, None
        oldComment_token, oldComment_subtoken = None, None
        newComment_token = split(streamPair[3])
        newComment_subtoken = streamPair[3]
        for x in possibleMapping:
            if cnt >= 1:
                break
            if oldComment_token is None:
                oldComment_token = split(streamPair[2])
                oldComment_subtoken = streamPair[2]
            pattern_token = " ".join(x[0]).replace(" <con> ", "").replace(" . ", ".")
            pattern_suboten = [x.lower() for x in x[0]]
            pattern_splited = [x.lower() for x in x[0] if x != "<con>"]
            indexes_token = match_token(oldComment_token, pattern_token, matchLevel)
            indexes_subtoken = match(oldComment_subtoken, pattern_suboten, matchLevel)
            indexes_splited = match(oldComment_subtoken, pattern_splited, matchLevel) if pattern_splited else None
            if not indexes_token:
                pass
            else:
                if equal(pattern_token, oldComment_token[indexes_token[0]], 1) and not equal(pattern_token,
                                                                                             oldComment_token[
                                                                                                 indexes_token[0]], 0):
                    if pattern_token[-1] != 's':
                        x[1][0] = tuple((x[1][0][0] + 's',))
                    else:
                        x[1][0] = tuple((x[1][0][0][:-1],))
                for index in indexes_token:
                    oldComment_token[index] = x[1]
                    predComment_token = oldComment_token
                cnt += 1

            if indexes_subtoken:
                bias = 0
                for index in indexes_subtoken:
                    predComment_subtoken = oldComment_subtoken[:index + bias] + list(x[1][0]) + oldComment_subtoken[
                                                                                                index + pattern_suboten.__len__() + bias:]
                    oldComment_subtoken = predComment_subtoken
                    bias = bias + x[1][0].__len__() - x[0].__len__()
                cnt += 1

            if indexes_splited:
                bias = 0
                for index in indexes_splited:
                    predComment_subtoken = oldComment_subtoken[:index + bias] + [y for y in list(x[1][0]) if
                                                                                 y != "<con>"] + oldComment_subtoken[
                                                                                                 index + pattern_splited.__len__() + bias:]
                    oldComment_subtoken = predComment_subtoken
                    bias = bias + x[1][0].__len__() - x[0].__len__()
                cnt += 1

        predComment_token = genAllpossible(predComment_token)

        if predComment_token is not None and isEqual_token(predComment_token, newComment_token, k):
            return predComment_token[0]
        elif predComment_subtoken is not None and isEqual(predComment_subtoken, newComment_subtoken):
            return predComment_subtoken
        if cnt == 0:
            return None

        # if predComment_subtoken:
        #     return predComment_subtoken
        # elif predComment_token:
        #     return predComment_token[0]
        # if cnt == 0:
        #     return None

    for i in range(3):
        predRes = tryAllPossible(possibleMapping, streamPair, matchLevel=i, k=k)
        if predRes is None:
            continue
        else:
            return predRes


def saveRes(testPath, Respath, predRes, flags):
    def isEqual(pred, oracle):
        predStr = stripAll.sub(' ', " ".join(pred).replace("<con>", '').replace('\n',' ')).strip(' !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~_')
        oracleStr = stripAll.sub(' ', " ".join(oracle).replace("<con>", '').replace('\n',' ')).strip(' !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~_')
        predStr = "".join([x for x in predStr if x.isalnum()])
        oracleStr = "".join([x for x in oracleStr if x.isalnum()])
        if predStr.lower() == oracleStr.lower():
            return True
        else:
            return False
    src_desc = []
    dst_desc = []
    with open(testPath, 'r', encoding='utf8') as f, open(Respath, 'w', encoding='utf8') as f_res:
        for i, x in enumerate(tqdm(f.readlines())):
            if i == 165:
                print("")
            res = {}
            fileInfo = json.loads(x)
            res["Origin"] = fileInfo["src_desc"]
            res["Reference"] = fileInfo["dst_desc"]
            src_desc.append(fileInfo["src_desc_tokens"])
            dst_desc.append(fileInfo["dst_desc_tokens"])
            # res["CUP"] = fileInfo["CUP_hypo_desc"]
            if predRes[i] is None:
                res["HebCup"] = fileInfo["src_desc"]
            else:
                res["HebCup"] = " ".join(predRes[i]).replace(" <con> ", "").replace(" . ", ".").replace(" }", "}").replace("{ ", "{") \
                    .replace(" )", ")").replace("( ", "(").replace(" # ", "#").replace(" ,", ",")
            # json.dump(res, f_res,indent=4)
            # if not flags[i]:
            #     continue
            json.dump(res, f_res)
            f_res.write('\n')
    return src_desc, dst_desc


def eval_AED_RED(src_desc, dst_desc, hypo_desc, removeSymbol=False):
    evalClass = EditDistance()
    # dst_desc = [[y.lower() for y in x] for x in dst_desc]
    # src_desc = [[y.lower() for y in x] for x in src_desc]
    dst_desc = [[y for y in x] for x in dst_desc]
    src_desc = [[y for y in x] for x in src_desc]
    for i, x in enumerate(hypo_desc):
        if x is None:
            hypo_desc[i] = [x.lower() for x in src_desc[i]]
        else:
            hypo_desc[i] = [x.lower() for x in hypo_desc[i]]
    return evalClass.eval(hypo_desc, dst_desc, src_desc, removeSymbol)


def camel_case_split(identifier):
    temp = re.sub(r'([A-Z][a-z])', r' \1', re.sub(r'([A-Z]+)', r' \1', identifier)).strip().split()
    return [x.lower() for x in temp if x!=""]


def genPureMapping(src_method, dst_method, mode='subtoken'):
    '''
    To generate pure mapping without "Replacement pair construction" in the paper.
    :return:
    '''
    oldTokens = [x.value for x in list(javalang.tokenizer.tokenize(src_method)) if not isinstance(x, javalang.tokenizer.Separator)]
    newTokens = [x.value for x in list(javalang.tokenizer.tokenize(dst_method)) if not isinstance(x, javalang.tokenizer.Separator)]
    ops = list(SequenceMatcher(None, oldTokens, newTokens).get_opcodes())
    cnt = defaultdict(int)
    tmpPairs = set()
    pairs = set()
    finalPairs = []
    for op in ops:
        if op[0] == 'equal':
            continue
        if op[1] - op[2] == op[3] - op[4]:  # same length
            for i in range(op[2] - op[1]):
                tmpPairs.add((oldTokens[op[1] + i], newTokens[op[3] + i]))
                cnt[str((oldTokens[op[1] + i], newTokens[op[3] + i]))] += 1
        else:
            tmpPairs.add((" ".join(oldTokens[op[1]:op[2]]), " ".join(newTokens[op[3]:op[4]])))
            cnt[str((" ".join(oldTokens[op[1]:op[2]]), " ".join(newTokens[op[3]:op[4]])))] += 1
    if mode == 'token':
        for x in tmpPairs:
            finalPairs.append((x[0], x[1], cnt[str(x)]))
        return finalPairs
    for x in tmpPairs:
        oldToken, newToken = camel_case_split(x[0]), camel_case_split(x[1])
        oldToken, newToken = [stripAllSymbol(x) for x in oldToken], [stripAllSymbol(x) for x in newToken]
        letterOps = list(SequenceMatcher(None, oldToken, newToken).get_opcodes())
        for op in letterOps:
            if op[0] != 'equal':
                pairs.add((" ".join(oldToken[op[1]:op[2]]).lower(), " ".join(newToken[op[3]:op[4]]).lower()))
                cnt[str((" ".join(oldToken[op[1]:op[2]]).lower(), " ".join(newToken[op[3]:op[4]]).lower()))] += 1
    for x in pairs:
        finalPairs.append((x[0],x[1],cnt[str(x)]))
    return finalPairs

def saveUnfixedItems(failedIDs, ACLItemsPath, outputPath):
    with open(ACLItemsPath, 'r', encoding='utf8') as f:
        ACLItems = json.loads(f.read())
    unfixedItems = []
    savedID = set()
    for item in ACLItems:
        if item['id'] in failedIDs:
            unfixedItems.append(item)
            savedID.add(item['id'])
        else:
            continue
    with open(outputPath, 'w', encoding='utf8') as f:
        f.write('[\n')
        for i, x in enumerate(unfixedItems):
            if i != unfixedItems.__len__() - 1:
                f.write(json.dumps(x) + ',' + '\n')
            else:
                f.write(json.dumps(x) + '\n')
        f.write(']\n')
        
def re_tokenize(instances):
    # excluded_set = {"<con>", "{", "}"}
    excluded_set = {"<con>"}
    new_instances = []
    for cur_instance in instances:
        cur_new_instance = []
        for x in cur_instance:
            cur_new_instance.extend([x for x in Tokenizer.tokenize_desc_with_con(x.replace("``", "\"")) if x not in excluded_set])
            # cur_new_instance.extend([x for x in Tokenizer.tokenize_desc_with_con(x)])
        new_instances.append(cur_new_instance)
    return new_instances

def cal_bleu(preds, refs,rmstop):
    preds = [" ".join(x).lower() for x in preds]
    refs = [" ".join(x).lower() for x in refs]
    score = bleu_fromstr(preds, refs, rmstop=rmstop)
    return score


if __name__ == '__main__':
   
    # -------- Code for Hyatt --------
    # with open('./dataset/ASTUpdaterRes/ASTUpdaterResult_Hyatt_OnAll.json', 'r', encoding='utf8') as f:  # Hyatt
    # with open('./dataset/CCBertRes/CCBertResForToper.json', 'r', encoding='utf8') as f:  # Toper for CCBert
    # with open('./dataset/CCBertRes/CCBertRes_All.json', 'r', encoding='utf8') as f:  
    # with open('./dataset/CodeT5Res/CodeT5_for_eval.json', 'r', encoding='utf8') as f: 
    # with open('./dataset/CodeT5Res/CodeT5_for_eval_refined.json', 'r', encoding='utf8') as f: 
    # with open('./dataset/ASTUpdaterRes/ASTUpdaterResult_AllInTest.json', 'r', encoding='utf8') as f: 
    # with open('./dataset/ASTUpdaterRes/ASTUpdaerRes_for_eval.json', 'r', encoding='utf8') as f: 
    # with open('./dataset/ASTUpdaterRes/ToperRes.json', 'r', encoding='utf8') as f: 
    # with open('./dataset/ASTUpdaterRes/ToperRes_for_eval_sorted.json', 'r', encoding='utf8') as f: 
    # with open('./dataset/ASTUpdaterRes/ToperRes_for_eval.json', 'r', encoding='utf8') as f: 
    # with open('./dataset/ASTUpdaterRes/ToperRes_for_eval_2.json', 'r', encoding='utf8') as f: 
    # with open('./dataset/ASTUpdaterRes/ToperRes_for_eval_refined.json', 'r', encoding='utf8') as f: 
    # with open('./dataset/ASTUpdaterRes/ToperRes_for_eval_refined_2.json', 'r', encoding='utf8') as f: 
    # with open('./dataset/ASTUpdaterRes/ASTUpdaterResult_classified_InValid.json', 'r', encoding='utf8') as f:  # Hyatt in valid
    # with open('./dataset/ASTUpdaterRes/ASTUpdaterReclassified_InValid_All.json', 'r', encoding='utf8') as f:  # Hyatt in valid
    # with open('./dataset/CCBertRes/CCBertRes_for_eval.json', 'r', encoding='utf8') as f: 
    # with open('./dataset/CCBertRes/CCBertv2Res_for_eval.json', 'r', encoding='utf8') as f: 
    with open('./dataset/CCBertRes/CCBertv2Res_for_eval_refined.json', 'r', encoding='utf8') as f: 
    # with open('./dataset/CCBertRes/CCBertRes_for_eval_refined.json', 'r', encoding='utf8') as f: 
    # with open('./dataset/CCBertRes/CoditT5Res_for_eval.json', 'r', encoding='utf8') as f: 
    # with open('./dataset/CCBertRes/CoditT5Res_for_eval_refined.json', 'r', encoding='utf8') as f: 
    # with open('./dataset/CCBertRes/CodeReviewerRes_for_eval.json', 'r', encoding='utf8') as f: 
    # with open('./dataset/CCBertRes/CCBert_Toper_for_eval.json', 'r', encoding='utf8') as f: 
    # with open('./dataset/CCBertRes/CCBert_Toper_for_eval_refined.json', 'r', encoding='utf8') as f: 
        tmp = json.load(f)
        pred_instances, references, src_instances = tmp
        references = [x[0] for x in references]
        src_instances = [x[0] for x in src_instances]
        
        # ------------------
        # pred_instances = re_tokenize(pred_instances)
        # references = re_tokenize(references)
        # src_instances = re_tokenize(src_instances)
        # ------------------
        src_desc = src_instances
        dst_desc = references
        pred_res_all = pred_instances
    # -------- end --------

    # print(eval_AED_RED(src_desc, dst_desc, pred_res_all))
    print("GLEU: ", calGleu(src_desc, dst_desc, pred_res_all, lowercase=True))
    print("BLEU: ", cal_bleu(pred_res_all, dst_desc, rmstop=False))
    # print(eval_AED_RED(src_instances, references, pred_instances))
    print(compute_accuracy([" ".join(split(x)) for x in pred_res_all], [" ".join(split(x)) for x in dst_desc]))
    # saveUnfixedItems(failedIDs=unfixedIDs, ACLItemsPath=ACLItemsPath, outputPath=unfixedItemsOutputPath)

    # TODO: 有部分typo可以被修复，例如reset the leader election throttle. -> reset the leader election throttles.
    #  同时在code change中也出现了
    # TODO: "sample_id": 5433548有时间可可以看一下
    # TODO: token权重相同的以出现次数排序