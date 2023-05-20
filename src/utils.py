import difflib
import enum
import json
import logging
import os
import random
import re
import time
from collections import defaultdict
from copy import deepcopy
import pickle
import numpy as np
import torch
from numpy import take
from tokenizers import ByteLevelBPETokenizer
from torch.utils.data import Dataset, TensorDataset
from tqdm import tqdm
from transformers import RobertaTokenizer, T5Tokenizer
from tree_sitter import Language, Parser


from myParser import (DFG_csharp, DFG_go, DFG_java, DFG_javascript, DFG_php,
                      DFG_python, DFG_ruby, index_to_code_token,
                      remove_comments_and_docstrings, tree_to_token_index,
                      tree_to_variable_index)

from sklearn import preprocessing

logger = logging.getLogger(__name__)

dfg_function = {
    'python': DFG_python,
    'java': DFG_java,
    'ruby': DFG_ruby,
    'go': DFG_go,
    'php': DFG_php,
    'javascript': DFG_javascript,
    'c_sharp': DFG_csharp,
}


def add_lang_by_task(target_str, task, sub_task):
    if task == 'summarize':
        target_str = '<en> ' + target_str
    elif task == 'refine':
        target_str = '<java> ' + target_str
    elif task == 'translate':
        if sub_task == 'java-cs':
            target_str = '<c_sharp> ' + target_str
        else:
            target_str = '<java> ' + target_str
    elif task == 'concode':
        target_str = '<java> ' + target_str
    elif task == 'defect':
        target_str = target_str
    return target_str


tag_matcher = re.compile(r"@@ -(\d+),(\d+) \+(\d+),(\d+) @@")


def apply_patch(old_file, diff):
    oldflines = old_file.split('\n')
    difflines = [line for line in diff.split('\n') if line !=
                 r"\ No newline at end of file"]
    matchres = tag_matcher.match(difflines[0])
    if matchres:
        startline, rangelen, startpos, endpos = matchres.groups()
    else:
        return None
    startline, rangelen = int(startline) - 1, int(rangelen)
    endline = startline + rangelen
    prevlines = oldflines[:startline]
    afterlines = oldflines[endline:]
    lines = []
    for line in difflines[1:]:
        if line.startswith("+"):
            lines.append(line[1:])
        elif not line.startswith("-"):
            lines.append(line[1:])
    new_lines = prevlines + lines + afterlines
    return "\n".join(new_lines)


def convert_defect_examples_to_features(item):
    example, example_index, tokenizer, args = item
    source_str = example.source
    code = tokenizer.encode(
        source_str, max_length=args.max_source_length, padding='max_length', truncation=True)
    return DefectInputFeatures(example_index, code, example.target)


class CloneInputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 example_id,
                 source_ids,
                 label,
                 url1,
                 url2
                 ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.label = label
        self.url1 = url1
        self.url2 = url2


class DefectInputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 example_id,
                 source_ids,
                 label
                 ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.label = label


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 example_id,
                 source_ids,
                 target_ids,
                 url=None
                 ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.url = url


class InputCCFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 example_id,
                 old_source_ids,
                 new_source_ids,
                 target_ids,
                 url=None
                 ):
        self.example_id = example_id
        self.old_source_ids = old_source_ids
        self.new_source_ids = new_source_ids
        self.target_ids = target_ids
        self.url = url


class Example(object):
    """A single training/test example."""

    def __init__(self,
                 idx,
                 source,
                 target,
                 url=None,
                 task='',
                 sub_task='',
                 meta_data=None
                 ):
        self.idx = idx
        self.source = source
        self.target = target
        self.url = url
        self.task = task
        self.sub_task = sub_task
        self.meta_data = meta_data


class CCExample(object):
    """A single training/test example."""

    def __init__(self,
                 idx,
                 old_source,
                 new_source,
                 diff,
                 target,
                 url=None,
                 task='',
                 sub_task='',
                 lang='',
                 meta_data=None
                 ):
        self.idx = idx
        self.old_source = old_source
        self.new_source = new_source
        self.diff = diff
        self.target = target
        self.url = url
        self.task = task
        self.sub_task = sub_task
        self.lang = lang
        self.meta_data = meta_data


class CloneExample(object):
    """A single training/test example."""

    def __init__(self,
                 code1,
                 code2,
                 label,
                 url1,
                 url2
                 ):
        self.source = code1
        self.target = code2
        self.label = label
        self.url1 = url1
        self.url2 = url2


def read_translate_examples(filename, data_num):
    """Read examples from filename."""
    examples = []
    assert len(filename.split(',')) == 2
    src_filename = filename.split(',')[0]
    trg_filename = filename.split(',')[1]
    idx = 0
    with open(src_filename) as f1, open(trg_filename) as f2:
        for line1, line2 in zip(f1, f2):
            src = line1.strip()
            trg = line2.strip()
            examples.append(
                Example(
                    idx=idx,
                    source=src,
                    target=trg,
                )
            )
            idx += 1
            if idx == data_num:
                break
    return examples


def read_refine_examples(filename, data_num):
    """Read examples from filename."""
    examples = []
    assert len(filename.split(',')) == 2
    src_filename = filename.split(',')[0]
    trg_filename = filename.split(',')[1]
    idx = 0

    with open(src_filename) as f1, open(trg_filename) as f2:
        for line1, line2 in zip(f1, f2):
            examples.append(
                Example(
                    idx=idx,
                    source=line1.strip(),
                    target=line2.strip(),
                )
            )
            idx += 1
            if idx == data_num:
                break
    return examples


def read_concode_examples(filename, data_num):
    """Read examples from filename."""
    examples = []

    with open(filename) as f:
        for idx, line in enumerate(f):
            x = json.loads(line)
            examples.append(
                Example(
                    idx=idx,
                    source=x["nl"].strip(),
                    target=" ".join(x["code"]).strip()  # test
                )
            )
            idx += 1
            if idx == data_num:
                break
    return examples


def read_CoRec_examples(filename, data_num):
    """Read examples from filename."""
    examples = []

    with open(filename) as f:
        for idx, line in enumerate(f):
            x = json.loads(line)
            examples.append(
                Example(
                    idx=idx,
                    source=x["code"].strip(),
                    target=x["nl"].strip()
                )
            )
            idx += 1
            if idx == data_num:
                break
    return examples


def read_codeSearchNet_examples(filename, data_num):
    """Read examples from filename."""
    examples = []

    with open(filename) as f:
        for idx, line in enumerate(f):
            x = json.loads(line)
            examples.append(
                Example(
                    idx=idx,
                    source=x["docstring"].strip(),
                    # target=x["code_tokens"].strip()
                    target=x["code"].strip()
                )
            )
            idx += 1
            if idx == data_num:
                break
    return examples


def read_summarize_examples(filename, data_num):
    """Read examples from filename."""
    examples = []
    with open(filename, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)
            if 'idx' not in js:
                js['idx'] = idx
            code = ' '.join(js['code_tokens']).replace('\n', ' ')
            code = ' '.join(code.strip().split())
            nl = ' '.join(js['docstring_tokens']).replace('\n', '')
            nl = ' '.join(nl.strip().split())
            examples.append(
                Example(
                    idx=idx,
                    source=code,
                    target=nl,
                )
            )
            if idx + 1 == data_num:
                break
    return examples


def read_defect_examples(filename, data_num):
    """Read examples from filename."""
    examples = []
    with open(filename, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)

            code = ' '.join(js['func'].split())
            examples.append(
                Example(
                    idx=js['idx'],
                    source=code,
                    target=js['target']
                )
            )
            if idx + 1 == data_num:
                break
    return examples


def read_clone_examples(filename, data_num):
    """Read examples from filename."""
    index_filename = filename
    url_to_code = {}
    with open('/'.join(index_filename.split('/')[:-1]) + '/data.jsonl') as f:
        for line in f:
            line = line.strip()
            js = json.loads(line)
            code = ' '.join(js['func'].split())
            url_to_code[js['idx']] = code

    data = []
    with open(index_filename) as f:
        idx = 0
        for line in f:
            line = line.strip()
            url1, url2, label = line.split('\t')
            if url1 not in url_to_code or url2 not in url_to_code:
                continue
            if label == '0':
                label = 0
            else:
                label = 1
            data.append(CloneExample(
                url_to_code[url1], url_to_code[url2], label, url1, url2))
            idx += 1
            if idx == data_num:
                break
    return data


def read_pretrain_eval_data(pretrain_data_dir):
    all_valid_files = [f for f in os.listdir(
        pretrain_data_dir) if f.endswith("_valid.jsonl")]
    languages = [f[:-12] for f in all_valid_files]
    print(f"Found Languages : {languages}")
    examples_dict = {}
    for lang in languages:
        fp = open(os.path.join(pretrain_data_dir, lang + "_valid.jsonl"))
        examples = []
        for li, line in enumerate(fp):
            d = json.loads(line.strip())
            examples.append(
                Example(
                    idx=li,
                    source=d['source'],
                    target=d['target'],
                    meta_data={
                        'transformer': d['transformer'],
                        'lang': lang
                    }
                )
            )
        examples_dict[lang] = examples
    return examples_dict


def calc_stats(examples, tokenizer=None, is_tokenize=False):
    avg_src_len = []
    avg_trg_len = []
    avg_src_len_tokenize = []
    avg_trg_len_tokenize = []
    for ex in examples:
        if is_tokenize:
            avg_src_len.append(len(ex.source.split()))
            avg_trg_len.append(len(str(ex.target).split()))
            avg_src_len_tokenize.append(len(tokenizer.tokenize(ex.source)))
            avg_trg_len_tokenize.append(
                len(tokenizer.tokenize(str(ex.target))))
        else:
            avg_src_len.append(len(ex.source.split()))
            avg_trg_len.append(len(str(ex.target).split()))
    if is_tokenize:
        logger.info("Read %d examples, avg src len: %d, avg trg len: %d, max src len: %d, max trg len: %d",
                    len(examples), np.mean(avg_src_len), np.mean(avg_trg_len), max(avg_src_len), max(avg_trg_len))
        logger.info("[TOKENIZE] avg src len: %d, avg trg len: %d, max src len: %d, max trg len: %d",
                    np.mean(avg_src_len_tokenize), np.mean(
                        avg_trg_len_tokenize), max(avg_src_len_tokenize),
                    max(avg_trg_len_tokenize))
    else:
        logger.info("Read %d examples, avg src len: %d, avg trg len: %d, max src len: %d, max trg len: %d",
                    len(examples), np.mean(avg_src_len), np.mean(avg_trg_len), max(avg_src_len), max(avg_trg_len))


def calc_stats_CC(examples, tokenizer=None, is_tokenize=False):
    avg_src_len = []
    avg_trg_len = []
    avg_src_len_tokenize = []
    avg_trg_len_tokenize = []
    for ex in examples:
        if is_tokenize:
            avg_src_len.append(len(ex.old_source.split()))
            avg_src_len.append(len(ex.new_source.split()))
            avg_trg_len.append(len(str(ex.target).split()))
            avg_src_len_tokenize.append(len(tokenizer.tokenize(ex.old_source)))
            avg_src_len_tokenize.append(len(tokenizer.tokenize(ex.new_source)))
            avg_trg_len_tokenize.append(
                len(tokenizer.tokenize(str(ex.target))))
        else:
            avg_src_len.append(len(ex.old_source.split()))
            avg_src_len.append(len(ex.new_source.split()))
            avg_trg_len.append(len(str(ex.target).split()))
    if is_tokenize:
        logger.info("Read %d examples, avg src len: %d, avg trg len: %d, max src len: %d, max trg len: %d",
                    len(examples), np.mean(avg_src_len), np.mean(avg_trg_len), max(avg_src_len), max(avg_trg_len))
        logger.info("[TOKENIZE] avg src len: %d, avg trg len: %d, max src len: %d, max trg len: %d",
                    np.mean(avg_src_len_tokenize), np.mean(
                        avg_trg_len_tokenize), max(avg_src_len_tokenize),
                    max(avg_trg_len_tokenize))
    else:
        logger.info("Read %d examples, avg src len: %d, avg trg len: %d, max src len: %d, max trg len: %d",
                    len(examples), np.mean(avg_src_len), np.mean(avg_trg_len), max(avg_src_len), max(avg_trg_len))


def get_elapse_time(t0):
    elapse_time = time.time() - t0
    if elapse_time > 3600:
        hour = int(elapse_time // 3600)
        minute = int((elapse_time % 3600) // 60)
        return "{}h{}m".format(hour, minute)
    else:
        minute = int((elapse_time % 3600) // 60)
        return "{}m".format(minute)


class ReviewFeatures(object):
    def __init__(self, example_id, source_ids, source_labels, target_ids, type):
        self.example_id = example_id
        self.source_ids = source_ids
        self.source_labels = source_labels
        self.target_ids = target_ids
        # assert type in ("label", "line", "genmsg", "daemsg")
        self.type = type

class ClsFeatures(object):
    def __init__(self, example_id, source_ids, y):
        self.example_id = example_id
        self.source_ids = source_ids
        self.y = y

class JITDPFeatures(object):
    def __init__(self, example_id, manual_feature, source_ids, y):
        self.example_id = example_id
        self.manual_feature = manual_feature
        self.source_ids = source_ids
        self.y = y  


class APCAFeatures(object):
    def __init__(self, example_id, source_ids, y, old_ids=None, new_ids=None):
        self.example_id = example_id
        self.source_ids = source_ids
        self.old_ids = old_ids
        self.new_ids = new_ids
        self.y = y 
        


class TextDataset(Dataset):

    def __init__(self, tokenizer, pool, args, file_path, samplenum=-1, random_sample_num=-1):
        self.cnt = 0
        self.tokenizer = tokenizer
        self.args = args
        if isinstance(tokenizer, T5Tokenizer):
            tokenizer_type = ""
        elif isinstance(tokenizer, RobertaTokenizer):
            tokenizer_type = "rb"
        else:
            tokenizer_type = "unk"

        savep = file_path.replace(".jsonl", tokenizer_type + ".exps")

        if os.path.exists(savep):
            logger.info("Loading examples from {}".format(savep))
            examples = torch.load(savep)
        else:
            logger.info("Reading examples from {}".format(file_path))
            start = time.time()
            # examples = read_review_examples(
            # args, file_path, samplenum, tokenizer=tokenizer)
            examples = read_CC_examples(
                args, file_path, samplenum, tokenizer=tokenizer)
            end = time.time()
            logger.info(f"Read examples time cost: {end-start}")
            logger.info(f"Tokenize examples: {file_path}")

            if args.debug:
                self.tokenize((examples[0], tokenizer, args))  # test

            examples = pool.map(self.tokenize,
                                [(example, tokenizer, args) for example in examples])
            torch.save(examples, savep)

        self.set_start_end_ids(examples)
        logger.info("Convert examples to features...")

        if random_sample_num != -1 and examples.__len__() > random_sample_num:
            examples = random.sample(examples, random_sample_num)
        else:
            examples = examples
        if args.debug:
            logger.info("Debug mode")
            logger.info(f"test random: {random.random()}")
            logger.info(f"Examples size: {examples.__len__()}")

        self.featss = pool.map(self.convert_examples_to_features,
                               [(example, tokenizer, args) for example in examples])
        logger.info(f"Examples converted")
        # expand the lists
        self.feats = [feat for feats in self.featss for feat in feats]

    def __len__(self):
        return len(self.feats)

    def __getitem__(self, i):
        return self.feats[i]

    def reset_len(self, data_len):
        assert len(self.feats) >= data_len
        self.feats = self.feats[:data_len]

    def set_start_end_ids(self, examples):
        for example in examples:
            labels = example.labels
            start_id = 0
            end_id = len(labels) - 1
            for i, label in enumerate(labels):
                if label != -100:               # find the first label
                    start_id = i
                    break
            for i in range(len(labels) - 1, -1, -1):
                label = labels[i]
                if label != -100:
                    end_id = i
                    break
            example.start_id = start_id
            example.end_id = end_id


    def tokenize(self, item):
        example, tokenizer, args = item
        # have disable the length limit or might cause mismatch between len(lables) and len(inputs)
        if example.tokenized is False:
            example.msg = self.encode_remove(tokenizer, example.msg, args)
            example.input = self.encode_remove(
                tokenizer, example.input, args, limit_length=False)
            e0id = tokenizer.special_dict["<e0>"]
            inputs = " ".join(str(id) for id in example.input)
            lines = inputs.split(" " + str(e0id) + " ")
            lines = [
                [int(v) for v in line.split(" ") if len(v) > 0] for line in lines
            ]  # just for integer the string
        else:
            lines = example.lines
        lens = [len(line) for line in lines]

        # assert [self.tokenizer.convert_tokens_to_ids(x) for x in example.encoded_lines] == lines # test
        lens = list(map(len, lines))
        curlen = len(lens) + sum(lens)  # \n + token ids
        left, right = 0, len(lines)
        # while curlen > args.max_source_length - 2:
        # compatibility for gen new code example. 22021027@Bo.
        while curlen > args.max_source_length - 2*len(lines) - example.msg.__len__() - 1:
            if left % 2 == 0:
                curlen -= 1 + len(lines[left])
                left += 1
            else:
                right -= 1
                curlen -= 1 + len(lines[right])
        lines = lines[left:right]
        labels = example.labels[left:right]
        assert len(lines) + sum(map(len, lines)) <= args.max_source_length - \
            2, "Too long inputs in TextDataset.tokenize."
        if len(lines) != len(labels):
            logger.info("Not equal length in TextDataset.tokenize.")
            lines = lines[:len(labels)]
            labels = labels[:len(lines)]
        example.lines = lines
        example.labels = labels

        return example

    def convert_examples_to_features(self, item):
        example, _, _ = item
        if len(example.msg) > 0:
            exs = []
            split_ratio = [20, 20, 20, 20, 20]
            for _ in range(4):  # up sampling
                if random.random() < (sum(split_ratio[:1])/sum(split_ratio)):
                    # MLM4CC
                    exs.append(self.gen_MLM4CC_example(item))

                elif random.random() < (sum(split_ratio[:2])/sum(split_ratio)):
                    # MLM4CM:
                    exs.append(self.gen_MLM4CM_example(item))

                elif random.random() < (sum(split_ratio[:3])/sum(split_ratio)):
                    # NL2PL
                    exs.append(self.gen_NL2PL_example(item))

                elif random.random() < (sum(split_ratio[:4])/sum(split_ratio)):
                    # PL2NL
                    exs.append(self.gen_PL2NL_example(item))
                else:
                    #CDG
                    tmp = self.gen_CDG_example(item)
                    if tmp is not None:
                        exs.append(tmp)
            return exs
    
        
    def get_DFG_parser(self, lang):
        tmp_parser = Parser()
        try:
            tmp_parser.set_language(Language(self.args.treesitter_path, lang))
        except Exception as e:
            print(e)
            return None
        return [tmp_parser, dfg_function[lang]]
                
            
        
    def gen_CDG_example(self, item):
        example, tokenizer, args = item
        lang = example.lang
        old_file = example.oldf
        ori_diff = example.diff
        
        cur_parser = self.get_DFG_parser(lang)
        new_file = apply_patch(old_file, ori_diff)
        try:
            old_file = remove_comments_and_docstrings(old_file, lang)
            new_file = remove_comments_and_docstrings(new_file, lang)
        except:
            return None
        diff = list(difflib.unified_diff(
            old_file.split('\n'), new_file.split('\n')))
        if diff.__len__() == 0:
            return None
        else:
            diff = diff[2:]
            diff[2] = diff[2].strip('\n')
        old_tokens, old_dfg, old_index_to_code = self.extract_dataflow(
        old_file, cur_parser, lang)  # index start from 0
        new_tokens, new_dfg, new_index_to_code = self.extract_dataflow(
            new_file, cur_parser, lang)
        if old_tokens.__len__() == 0:
            return None
        matchres = tag_matcher.match(diff[0])
        if matchres:
            source_start, source_length, target_start, target_length = matchres.groups()
            source_start, source_length, target_start, target_length = \
                int(source_start), int(source_length), int(
                    target_start), int(target_length)
        else:
            return None
        
        changed_old_dfg = self.filter_dfg(old_dfg, old_index_to_code, (
            source_start - 1, source_start + source_length))  # get the dfg within the line scope
        changed_new_dfg = self.filter_dfg(
            new_dfg, new_index_to_code, (target_start - 1, target_start + target_length))
        if self.is_equal_dfg(changed_old_dfg, changed_new_dfg):
            return None
        diff_str = ""
        sep = "<extra_id_0>"
        old_code_str = ""
        for line in diff[1:]:
            if line[0] == '+':
                diff_str += "<add>" + line[1:]
            elif line[0] == '-':
                diff_str += "<del>" + line[1:]
                old_code_str += "<del>" + line[1:]
            else:
                diff_str += "<keep>" + line[1:]
                
        tmp_dfg_str_list = []
        for edge in changed_old_dfg:
            for end_node in edge[3]:
                if edge[2] == 'comesFrom':
                    tmp_dfg_str_list.append(edge[0] + " " + end_node)
                elif edge[2] == 'computedFrom':
                    tmp_dfg_str_list.append(end_node + " " + edge[0])
                else:
                    raise("Node relationship wrong")
        old_dfg_str = sep.join(tmp_dfg_str_list)
        
        tmp_dfg_str_list = []
        for edge in changed_new_dfg:
            for end_node in edge[3]:
                if edge[2] == 'comesFrom':
                    tmp_dfg_str_list.append(edge[0] + " " + end_node)
                elif edge[2] == 'computedFrom':
                    tmp_dfg_str_list.append(end_node + " " + edge[0])
                else:
                    raise("Node relationship wrong")
        new_dfg_str = sep.join(tmp_dfg_str_list)
        
        # old data flow + new data flow + old code -> code diff
        input_str = old_dfg_str + sep + new_dfg_str + sep + old_code_str
        output_str = diff_str
        
        source_ids = self.encode_remove(tokenizer, input_str, args)
        target_ids = self.encode_remove(tokenizer, output_str, args)
        source_ids, target_ids = self.pad_assert(source_ids, target_ids, args, tokenizer)
        input_labels = [-100] * len(source_ids)
        
        return ReviewFeatures(example.idx, source_ids, input_labels, target_ids, type="gendfg")
    
    def filter_dfg(self, dfg, index, scope):
        valid_dfg = []
        for edge in dfg:
            src_pos = index[edge[1]]
            if src_pos != -1:
                src_pos = src_pos[0][0]
            if scope[0] <= src_pos < scope[1]:
                valid_dfg.append(edge)
        return valid_dfg
    
    def extract_dataflow(self, code, parser, lang):
        """
        remove comments, tokenize code and extract dataflow
        Args:
            code (_type_): _description_
            parser (_type_): _description_
            lang (_type_): _description_

        Returns:
            _type_: dataflow of input code
        """
        # remove comments
        try:
            code = remove_comments_and_docstrings(code, lang)
        except:
            pass
        # obtain dataflow
        if lang == "php":
            code = "<?php"+code+"?>"
        try:
            code_tokens = []
            code_to_index = defaultdict(lambda: -1)
            tree = parser[0].parse(bytes(code, 'utf8'))
            root_node = tree.root_node
            tokens_index = tree_to_token_index(root_node)
            code = code.split('\n')
            code_tokens = [index_to_code_token(x, code) for x in tokens_index]
            index_to_code = {}
            
            for idx, (index, code) in enumerate(zip(tokens_index, code_tokens)):
                index_to_code[index] = (idx, code)
                code_to_index[idx] = index
            try:
                DFG, _ = parser[1](root_node, index_to_code, {})
            except:
                DFG = []
            DFG = sorted(DFG, key=lambda x: x[1])
            indexs = set()
            for d in DFG:
                if len(d[-1]) != 0:
                    indexs.add(d[1])
                for x in d[-1]:
                    indexs.add(x)
            new_DFG = []
            for d in DFG:
                if d[1] in indexs:
                    new_DFG.append(d)
            dfg = new_DFG
        except:
            dfg = []
        return code_tokens, dfg, code_to_index


    def is_equal_dfg(self, dfg_a, dfg_b):
        for edge_a, edge_b in zip(dfg_a, dfg_b):
            if edge_a[0] == edge_b[0] and edge_a[2] == edge_b[2] and edge_a[3] == edge_b[3]:
                continue
            else:
                return False
        return True
    
    def encoder_example(self, item):
        # Diff tag prediction
        # take added, keep, del line as label:
        example, tokenizer, args = item
        lines = example.lines
        labels = example.labels
        target_ids = [tokenizer.pad_id] * args.max_target_length
        source_ids, input_labels = [], []
        for i, (line, label) in enumerate(zip(lines, labels)):
            if i == example.start_id:
                source_ids.append(tokenizer.start_id)
                input_labels.append(-100)
            # only insert special tokens at diffs, not context (since it only for predict diff tag --Bo.)
            if label != -100:
                source_ids.append(tokenizer.mask_id)
                input_labels.append(label)
            source_ids.extend(line)
            input_labels.extend([-100] * len(line))
            if i == example.end_id:
                source_ids.append(tokenizer.end_id)
                input_labels.append(-100)
        assert len(input_labels) == len(source_ids), "Not equal length."
        assert len(
            input_labels) <= args.max_source_length, f"Too long inputs: {len(input_labels)}."
        source_ids = source_ids[:args.max_source_length - 2]
        input_labels = input_labels[:args.max_source_length - 2]
        source_ids = [tokenizer.bos_id] + source_ids + [tokenizer.eos_id]
        input_labels = [-100] + input_labels + [-100]
        pad_len = args.max_source_length - len(source_ids)
        source_ids += [tokenizer.pad_id] * pad_len
        input_labels += [-100] * pad_len

        new_input_labels = []
        map_dict = {0: tokenizer.del_id,
                    1: tokenizer.add_id, 2: tokenizer.keep_id}
        for label in input_labels:
            if label == -100:
                new_input_labels.append(-100)
            else:
                new_input_labels.append(map_dict[label])
        input_labels = new_input_labels
        assert len(source_ids) == args.max_source_length, "Not equal length."
        assert len(input_labels) == args.max_source_length, "Not equal length."
        return ReviewFeatures(example.idx, source_ids, input_labels, target_ids, type="label")

    def gen_MLM4CC_example(self, item):
        example, tokenizer, args = item
        lines = example.lines
        labels = example.labels

        input_labels = [-100] * args.max_source_length
        source_ids, target_ids = [], []
        SPECIAL_ID = 0
        mask_idxs = random.choices(
            range(len(lines)), k=int(len(lines) * args.mask_rate))
        id_dict = {0: tokenizer.del_id,
                   1: tokenizer.add_id, 2: tokenizer.keep_id}
        for i, (line, label) in enumerate(zip(lines, labels)):
            if i == example.start_id:
                source_ids.append(tokenizer.start_id)
            if label in id_dict:
                source_ids.append(id_dict[label])
            if i in mask_idxs:
                source_ids.append(tokenizer.special_dict[f"<e{SPECIAL_ID}>"])
                target_ids.append(tokenizer.special_dict[f"<e{SPECIAL_ID}>"])
                target_ids.extend(line)
                if SPECIAL_ID < 99:     # only 0-99 ids in vocab
                    SPECIAL_ID += 1
            else:
                source_ids.extend(line)
            if i == example.end_id:
                source_ids.append(tokenizer.end_id)
        source_ids.append(tokenizer.msg_id)
        source_ids.extend(example.msg)
        source_ids, target_ids = self.pad_assert(
            source_ids, target_ids, args, tokenizer)
        return ReviewFeatures(example.idx, source_ids, input_labels, target_ids, type="line")

    def decoder_example(self, item):
        example, tokenizer, args = item
        lines = example.lines
        labels = example.labels

        input_labels = [-100] * args.max_source_length
        source_ids, target_ids = [], []
        SPECIAL_ID = 0
        mask_idxs = random.choices(
            range(len(lines)), k=int(len(lines) * args.mask_rate))
        id_dict = {0: tokenizer.del_id,
                   1: tokenizer.add_id, 2: tokenizer.keep_id}
        for i, (line, label) in enumerate(zip(lines, labels)):
            if i == example.start_id:
                source_ids.append(tokenizer.start_id)
            if label in id_dict:
                source_ids.append(id_dict[label])
            if i in mask_idxs:
                source_ids.append(tokenizer.special_dict[f"<e{SPECIAL_ID}>"])
                target_ids.append(tokenizer.special_dict[f"<e{SPECIAL_ID}>"])
                target_ids.extend(line)
                if SPECIAL_ID < 99:     # only 0-99 ids in vocab
                    SPECIAL_ID += 1
            else:
                source_ids.extend(line)
            if i == example.end_id:
                source_ids.append(tokenizer.end_id)
        source_ids, target_ids = self.pad_assert(
            source_ids, target_ids, args, tokenizer)
        return ReviewFeatures(example.idx, source_ids, input_labels, target_ids, type="line")

    def gen_NL2PL_example(self, item):
        example, tokenizer, args = item
        lines = example.lines
        labels = example.labels
        input_labels = [-100] * args.max_source_length
        source_ids, target_ids = [], []
        id_dict = {0: tokenizer.del_id,
                   1: tokenizer.add_id, 2: tokenizer.keep_id}

        for i, (line, label) in enumerate(zip(lines, labels)):
            if i == example.start_id:
                source_ids.append(tokenizer.start_id)
            if label == 0 or label == 2:
                source_ids.append(id_dict[label])
            elif label == 1:
                target_ids.append(tokenizer.add_id)
                target_ids.extend(line)
                continue
            source_ids.extend(line)
            if i == example.end_id:
                source_ids.append(tokenizer.end_id)
        source_ids.append(tokenizer.msg_id)
        source_ids.extend(example.msg)
        assert len(
            source_ids) <= args.max_source_length, f"Too long inputs: {len(source_ids)} in gen_NL2PL_example with example {example.idx}."
        source_ids, target_ids = self.pad_assert(
            source_ids, target_ids, args, tokenizer)
        return ReviewFeatures(example.idx, source_ids, input_labels, target_ids, type="gen_new_code")

    def gen_PL2NL_example(self, item):
        """generate pretraining example for commit message generation tasks

        Args:
            item (_type_): _description_

        Returns:
            _type_: _description_
        """
        example, tokenizer, args = item
        lines = example.lines
        labels = example.labels
        input_labels = [-100] * args.max_source_length
        source_ids, target_ids = [], []
        id_dict = {0: tokenizer.del_id,
                   1: tokenizer.add_id, 2: tokenizer.keep_id}
        for i, (line, label) in enumerate(zip(lines, labels)):
            if i == example.start_id:
                source_ids.append(tokenizer.start_id)
            if label != -100:
                source_ids.append(id_dict[label])
            source_ids.extend(line)
            if i == example.end_id:
                source_ids.append(tokenizer.end_id)
        target_ids.append(tokenizer.msg_id)
        target_ids.extend(example.msg)
        assert len(
            source_ids) <= args.max_source_length, f"Too long inputs: {len(source_ids)}."
        source_ids, target_ids = self.pad_assert(
            source_ids, target_ids, args, tokenizer)
        return ReviewFeatures(example.idx, source_ids, input_labels, target_ids, type="gen_msg")

    def gen_masked_ids(self, ids, mask_rate):

        source_ids, target_ids = [], []
        msg_ids = deepcopy(ids)
        masks = [random.random() < mask_rate for _ in range(len(msg_ids))]
        if sum(masks) == 0:
            idx = random.choice(range(len(msg_ids)))
            masks[idx] = True
        source_ids, target_ids = [], []
        i = 0
        SPECIAL_ID = 0
        while i < len(masks):
            j = i
            while j < len(masks) and not masks[j]:
                source_ids.append(msg_ids[j])
                j += 1
            if j == len(masks):
                break
            source_ids.append(self.tokenizer.special_dict[f"<e{SPECIAL_ID}>"])
            target_ids.append(self.tokenizer.special_dict[f"<e{SPECIAL_ID}>"])
            while j < len(masks) and masks[j]:
                target_ids.append(msg_ids[j])
                j += 1
            if SPECIAL_ID < 99:     # only 0-99 ids in vocab
                SPECIAL_ID += 1
            i = j

        return source_ids, target_ids

    def gen_MLM4CM_example(self, item):
        """
        Denoising Review Comment: masked message -> message, and 
        context + diff + context + masked message -> message

        Args:
            item (_type_): _description_

        Returns:
            _type_: _description_
        """
        example, tokenizer, args = item
        input_labels = [-100] * args.max_source_length
        if random.random() < 0.5:
            # update by 20221027@Bo.
            source_ids, target_ids = self.gen_masked_ids(example.msg, 0.2)
        else:
            source_ids, target_ids = [], []
            id_dict = {0: tokenizer.del_id,
                       1: tokenizer.add_id, 2: tokenizer.keep_id}
            for i, (line, label) in enumerate(zip(example.lines, example.labels)):
                if i == example.start_id:
                    source_ids.append(tokenizer.start_id)
                if label != -100:
                    # label 0 for <del>, 1 for <add>, 2 for context
                    source_ids.append(id_dict[label])
                source_ids.extend(line)
                if i == example.end_id:
                    # TODO: append end tag here or after the masked message ids
                    source_ids.append(tokenizer.end_id)

            masked_msg_ids, masked_msg_tgt_ids = self.gen_masked_ids(
                example.msg, 0.2)
            source_ids.extend(masked_msg_ids)
            target_ids.extend(masked_msg_tgt_ids)
            assert len(
                source_ids) <= args.max_source_length, f"Too long inputs: {len(source_ids)}."
        source_ids, target_ids = self.pad_assert(
            source_ids, target_ids, args, tokenizer)
        
        return ReviewFeatures(example.idx, source_ids, input_labels, target_ids, type="daemsg")


    def daemsg_example_2(self, item):
        """_summary_
            context + diff + context + masked commit message -> commit message
        Args:
            item (_type_): _description_

        Returns:
            _type_: _description_
        """
        example, tokenizer, args = item
        lines = example.lines
        labels = example.labels
        input_labels = [-100] * args.max_source_length
        source_ids, target_ids = [], []
        id_dict = {0: tokenizer.del_id,
                   1: tokenizer.add_id, 2: tokenizer.keep_id}
        for i, (line, label) in enumerate(zip(lines, labels)):
            if i == example.start_id:
                source_ids.append(tokenizer.start_id)
            if label != -100:
                # label 0 for <del>, 1 for <add>, 2 for context
                source_ids.append(id_dict[label])
            source_ids.extend(line)
            if i == example.end_id:
                # TODO: end id in here or after the masked message ids
                source_ids.append(tokenizer.end_id)

        masked_msg_ids, masked_msg_tgt_ids = self.gen_masked_ids(
            example.msg, 0.2)
        source_ids.extend(masked_msg_ids)
        target_ids.extend(masked_msg_tgt_ids)
        assert len(
            source_ids) <= args.max_source_length, f"Too long inputs: {len(source_ids)}."
        source_ids, target_ids = self.pad_assert(
            source_ids, target_ids, args, tokenizer)
        return ReviewFeatures(example.idx, source_ids, input_labels, target_ids, type="daemsg")

    def pad_assert(self, source_ids, target_ids, args, tokenizer):
        source_ids = source_ids[:args.max_source_length - 2]
        source_ids = [tokenizer.bos_id] + source_ids + [tokenizer.eos_id]
        pad_len = args.max_source_length - len(source_ids)
        source_ids += [tokenizer.pad_id] * pad_len
        target_ids = target_ids[:args.max_target_length - 1]
        target_ids = target_ids + [tokenizer.eos_id]
        pad_len = args.max_target_length - len(target_ids)
        target_ids += [tokenizer.pad_id] * pad_len
        assert len(source_ids) == args.max_source_length, "Not equal length."
        assert len(target_ids) == args.max_target_length, "Not equal length."
        return source_ids, target_ids

    def encode_remove(self, tokenizer, text, args, limit_length=True):
        if limit_length is True:
            text = tokenizer.encode(
                text, max_length=args.max_source_length - 2, truncation=True)
        else:
            text = tokenizer.encode(
                text)
        if type(tokenizer) == T5Tokenizer:
            return text[:-1]
        elif type(tokenizer) == RobertaTokenizer:
            return text[1:-1]
        else:
            raise NotImplementedError


class DFGGenDataset(TextDataset):

    def __init__(self, tokenizer, pool, args, file_path, samplenum=-1, random_sample_num=-1):
        self.tokenizer = tokenizer
        self.args = args
        self.language_parsers = self.construct_parsers(
            ['java', 'python', 'go', 'php', 'ruby', 'javascript','c_sharp'])
        if isinstance(tokenizer, T5Tokenizer):
            tokenizer_type = ""
        elif isinstance(tokenizer, RobertaTokenizer):
            tokenizer_type = "rb"
        else:
            tokenizer_type = "unk"
        savep = file_path.replace(".jsonl", tokenizer_type + ".dfggenexps")
        if os.path.exists(savep):
            logger.info("Loading examples from {}".format(savep))
            self.feats = torch.load(savep)
        else:
            data = read_jsonl(file_path)
            for i in range(len(data)):
                data[i]["idx"] = i
            logger.info(f"Tokenize examples: {file_path}")
            self.feats = [self.convert_examples_to_features_to_diff(
                (dic, tokenizer, args)) for dic in tqdm(data)]
            # self.feats = pool.map(self.convert_examples_to_features,
            #                     [(dic, tokenizer, args) for dic in data])
            self.feats = [x for x in self.feats if x]
            torch.save(self.feats, savep)
            
        if random_sample_num != -1 and self.feats.__len__() > random_sample_num:
            self.feats = random.sample(self.feats, random_sample_num)
            
            
    def construct_parsers(self, langs=['python']):
        local_parsers = {}
        for lang in langs:
            tmp_parser = Parser()
            try:
                tmp_parser.set_language(
                    Language(self.args.treesitter_path, lang))
                local_parsers[lang] = [tmp_parser, dfg_function[lang]]
            except Exception as e:
                print(e)
                continue
        return local_parsers


    def convert_examples_to_features_to_dfg(self, item):
        js, tokenizer, args = item
        # debug
        # if js["idx"] != 777:
        #     return None
        # print(js["idx"])
        
        if "lang" not in js:
            js["lang"] = ""
        if "old_file" in js:
            old_file = js["old_file"]
            ori_diff = js["diff"]
            msg = js["nl"] if "nl" in js else "",
            lang = js["lang"]
        elif "oldf" in js:
            old_file = js["oldf"]
            ori_diff = js["patch"]
            msg = js["msg"] if "msg" in js else "",
            lang = js["lang"]
        else:
            return
        cur_parser = self.language_parsers[lang]
        new_file = apply_patch(old_file, ori_diff)
        old_file = remove_comments_and_docstrings(old_file, lang)
        new_file = remove_comments_and_docstrings(new_file, lang)
        diff = list(difflib.unified_diff(
            old_file.split('\n'), new_file.split('\n')))
        if diff.__len__() == 0:
            return None
        else:
            diff = diff[2:]
            diff[2] = diff[2].strip('\n')
        old_tokens, old_dfg, old_index_to_code = self.extract_dataflow(
        old_file, cur_parser, lang)  # index start from 0
        new_tokens, new_dfg, new_index_to_code = self.extract_dataflow(
            new_file, cur_parser, lang)
        if old_tokens.__len__() == 0:
            return None

        matchres = tag_matcher.match(diff[0])
        if matchres:
            source_start, source_length, target_start, target_length = matchres.groups()
            source_start, source_length, target_start, target_length = \
                int(source_start), int(source_length), int(
                    target_start), int(target_length)
        else:
            return None

        changed_old_dfg = self.filter_dfg(old_dfg, old_index_to_code, (
            source_start - 1, source_start + source_length))  # get the dfg within the line scope
        changed_new_dfg = self.filter_dfg(
            new_dfg, new_index_to_code, (target_start - 1, target_start + target_length))
        if self.is_equal_dfg(changed_old_dfg, changed_new_dfg):
            return None
        old_dfg_normalized, old_var_mapping_anon, old_var_mapping = self.normalize_dataflow(
            changed_old_dfg)
        new_dfg_normalized, new_var_mapping_anon, new_var_mapping = self.normalize_dataflow(
            changed_new_dfg, old_var_mapping)
        old_updated_code = self.update_code(
            old_file, old_var_mapping_anon, old_index_to_code, (source_start - 1, source_start + source_length))
        new_updated_code = self.update_code(
            new_file, new_var_mapping_anon, new_index_to_code, (target_start - 1, target_start + target_length))
        normalized_diff = list(difflib.unified_diff(
            old_updated_code.split('\n'), new_updated_code.split('\n')))[2:]
        normalized_diff[2] = normalized_diff[2].strip('\n')
        
        input_str = ""
        sep = "<extra_id_0>"
        # sep = " "
        # diff 
        for line in normalized_diff[1:]:
            if line[0] == '+':
                input_str += "<add>" + line[1:]
            elif line[0] == '-':
                input_str += "<del>" + line[1:]
            else:
                input_str += "<keep>" + line[1:]
        
        tmp_dfg_str_list = []
        input_str += sep
        for edge in old_dfg_normalized:
            for end_node in edge[2]:
                if edge[1] == 'comesFrom':
                    tmp_dfg_str_list.append(edge[0] + " " + end_node)
                elif edge[1] == 'computedFrom':
                    tmp_dfg_str_list.append(end_node + " " + edge[0])
                else:
                    raise("Node relationship wrong")
    
        dfg_str = sep.join(tmp_dfg_str_list)
        input_str += dfg_str
        source_ids = self.encode_remove(tokenizer, input_str, args)
        
        output_str = sep
        tmp_dfg_str_list = []
        for edge in new_dfg_normalized:
            for end_node in edge[2]:
                if edge[1] == 'comesFrom':
                    tmp_dfg_str_list.append(edge[0] + " " + end_node)
                elif edge[1] == 'computedFrom':
                    tmp_dfg_str_list.append(end_node + " " + edge[0])
                else:
                    raise("Node relationship wrong")
        dfg_str = sep.join(tmp_dfg_str_list)
        output_str += dfg_str
        target_ids = self.encode_remove(tokenizer, output_str, args)
        source_ids, target_ids = self.pad_assert(
            source_ids, target_ids, args, tokenizer)
        input_labels = [-100] * len(source_ids)
        return ReviewFeatures(js["idx"], source_ids, input_labels, target_ids, type="gendfg")
    
    def convert_examples_to_features_to_diff_normalized(self, item):
        js, tokenizer, args = item
        # debug
        # if js["idx"] != 777:
        #     return None
        # print(js["idx"])
        
        if "lang" not in js:
            js["lang"] = ""
        if "old_file" in js:
            old_file = js["old_file"]
            ori_diff = js["diff"]
            msg = js["nl"] if "nl" in js else "",
            lang = js["lang"]
        elif "oldf" in js:
            old_file = js["oldf"]
            ori_diff = js["patch"]
            msg = js["msg"] if "msg" in js else "",
            lang = js["lang"]
        else:
            return
        cur_parser = self.language_parsers[lang]
        new_file = apply_patch(old_file, ori_diff)
        old_file = remove_comments_and_docstrings(old_file, lang)
        new_file = remove_comments_and_docstrings(new_file, lang)
        diff = list(difflib.unified_diff(
            old_file.split('\n'), new_file.split('\n')))
        if diff.__len__() == 0:
            return None
        else:
            diff = diff[2:]
            diff[2] = diff[2].strip('\n')
        old_tokens, old_dfg, old_index_to_code = self.extract_dataflow(
        old_file, cur_parser, lang)  # index start from 0
        new_tokens, new_dfg, new_index_to_code = self.extract_dataflow(
            new_file, cur_parser, lang)
        if old_tokens.__len__() == 0:
            return None

        matchres = tag_matcher.match(diff[0])
        if matchres:
            source_start, source_length, target_start, target_length = matchres.groups()
            source_start, source_length, target_start, target_length = \
                int(source_start), int(source_length), int(
                    target_start), int(target_length)
        else:
            return None

        changed_old_dfg = self.filter_dfg(old_dfg, old_index_to_code, (
            source_start - 1, source_start + source_length))  # get the dfg within the line scope
        changed_new_dfg = self.filter_dfg(
            new_dfg, new_index_to_code, (target_start - 1, target_start + target_length))
        if self.is_equal_dfg(changed_old_dfg, changed_new_dfg):
            return None
        old_dfg_normalized, old_var_mapping_anon, old_var_mapping = self.normalize_dataflow(
            changed_old_dfg)
        new_dfg_normalized, new_var_mapping_anon, new_var_mapping = self.normalize_dataflow(
            changed_new_dfg, old_var_mapping)
        old_updated_code = self.update_code(
            old_file, old_var_mapping_anon, old_index_to_code, (source_start - 1, source_start + source_length))
        new_updated_code = self.update_code(
            new_file, new_var_mapping_anon, new_index_to_code, (target_start - 1, target_start + target_length))
        normalized_diff = list(difflib.unified_diff(
            old_updated_code.split('\n'), new_updated_code.split('\n')))[2:]
        normalized_diff[2] = normalized_diff[2].strip('\n')
        
        diff_str = ""
        sep = "<extra_id_0>"
        # sep = " "
        # diff 
        for line in normalized_diff[1:]:
            if line[0] == '+':
                diff_str += "<add>" + line[1:]
            elif line[0] == '-':
                diff_str += "<del>" + line[1:]
            else:
                diff_str += "<keep>" + line[1:]
        
        tmp_dfg_str_list = []
        for edge in old_dfg_normalized:
            for end_node in edge[2]:
                if edge[1] == 'comesFrom':
                    tmp_dfg_str_list.append(edge[0] + " " + end_node)
                elif edge[1] == 'computedFrom':
                    tmp_dfg_str_list.append(end_node + " " + edge[0])
                else:
                    raise("Node relationship wrong")
    
        old_dfg_str = sep.join(tmp_dfg_str_list)
        
        tmp_dfg_str_list = []
        for edge in new_dfg_normalized:
            for end_node in edge[2]:
                if edge[1] == 'comesFrom':
                    tmp_dfg_str_list.append(edge[0] + " " + end_node)
                elif edge[1] == 'computedFrom':
                    tmp_dfg_str_list.append(end_node + " " + edge[0])
                else:
                    raise("Node relationship wrong")
        new_dfg_str = sep.join(tmp_dfg_str_list)
        
        # old data flow + new data flow -> code diff
        input_str = old_dfg_str + sep + new_dfg_str
        output_str = diff_str
        
        source_ids = self.encode_remove(tokenizer, input_str, args)
        target_ids = self.encode_remove(tokenizer, output_str, args)
        source_ids, target_ids = self.pad_assert(
        source_ids, target_ids, args, tokenizer)
        input_labels = [-100] * len(source_ids)
        
        return ReviewFeatures(js["idx"], source_ids, input_labels, target_ids, type="gendfg")
    

    def convert_examples_to_features_to_diff(self, item):
        js, tokenizer, args = item
        # debug
        # if js["idx"] != 777:
        #     return None
        # print(js["idx"])
        
        if "lang" not in js:
            js["lang"] = ""
        if "old_file" in js:
            old_file = js["old_file"]
            ori_diff = js["diff"]
            msg = js["nl"] if "nl" in js else "",
            lang = js["lang"]
        elif "oldf" in js:
            old_file = js["oldf"]
            ori_diff = js["patch"]
            msg = js["msg"] if "msg" in js else "",
            lang = js["lang"]
        else:
            return
        cur_parser = self.language_parsers[lang]
        new_file = apply_patch(old_file, ori_diff)
        old_file = remove_comments_and_docstrings(old_file, lang)
        new_file = remove_comments_and_docstrings(new_file, lang)
        diff = list(difflib.unified_diff(
            old_file.split('\n'), new_file.split('\n')))
        if diff.__len__() == 0:
            return None
        else:
            diff = diff[2:]
            diff[2] = diff[2].strip('\n')
        old_tokens, old_dfg, old_index_to_code = self.extract_dataflow(
        old_file, cur_parser, lang)  # index start from 0
        new_tokens, new_dfg, new_index_to_code = self.extract_dataflow(
            new_file, cur_parser, lang)
        if old_tokens.__len__() == 0:
            return None

        matchres = tag_matcher.match(diff[0])
        if matchres:
            source_start, source_length, target_start, target_length = matchres.groups()
            source_start, source_length, target_start, target_length = \
                int(source_start), int(source_length), int(
                    target_start), int(target_length)
        else:
            return None

        changed_old_dfg = self.filter_dfg(old_dfg, old_index_to_code, (
            source_start - 1, source_start + source_length))  # get the dfg within the line scope
        changed_new_dfg = self.filter_dfg(
            new_dfg, new_index_to_code, (target_start - 1, target_start + target_length))
        if self.is_equal_dfg(changed_old_dfg, changed_new_dfg):
            return None
        old_dfg_normalized, old_var_mapping_anon, old_var_mapping = self.normalize_dataflow(
            changed_old_dfg)
        new_dfg_normalized, new_var_mapping_anon, new_var_mapping = self.normalize_dataflow(
            changed_new_dfg, old_var_mapping)

        diff_str = ""
        sep = "<extra_id_0>"

        for line in diff[1:]:
            if line[0] == '+':
                diff_str += "<add>" + line[1:]
            elif line[0] == '-':
                diff_str += "<del>" + line[1:]
            else:
                diff_str += "<keep>" + line[1:]
        
        tmp_dfg_str_list = []
        for edge in changed_old_dfg:
            for end_node in edge[3]:
                if edge[2] == 'comesFrom':
                    tmp_dfg_str_list.append(edge[0] + " " + end_node)
                elif edge[2] == 'computedFrom':
                    tmp_dfg_str_list.append(end_node + " " + edge[0])
                else:
                    raise("Node relationship wrong")
    
        old_dfg_str = sep.join(tmp_dfg_str_list)
        
        tmp_dfg_str_list = []
        for edge in changed_new_dfg:
            for end_node in edge[3]:
                if edge[2] == 'comesFrom':
                    tmp_dfg_str_list.append(edge[0] + " " + end_node)
                elif edge[2] == 'computedFrom':
                    tmp_dfg_str_list.append(end_node + " " + edge[0])
                else:
                    raise("Node relationship wrong")
        new_dfg_str = sep.join(tmp_dfg_str_list)
        
        # old data flow + new data flow -> code diff
        input_str = old_dfg_str + sep + new_dfg_str
        output_str = diff_str
        
        source_ids = self.encode_remove(tokenizer, input_str, args)
        target_ids = self.encode_remove(tokenizer, output_str, args)
        source_ids, target_ids = self.pad_assert(source_ids, target_ids, args, tokenizer)
        input_labels = [-100] * len(source_ids)
        return ReviewFeatures(js["idx"], source_ids, input_labels, target_ids, type="gendfg")
    
    def filter_dfg(self, dfg, index, scope):
        valid_dfg = []
        for edge in dfg:
            src_pos = index[edge[1]]
            if src_pos != -1:
                src_pos = src_pos[0][0]
            if scope[0] <= src_pos < scope[1]:
                valid_dfg.append(edge)
        return valid_dfg

    def is_equal_dfg(self, dfg_a, dfg_b):
        for edge_a, edge_b in zip(dfg_a, dfg_b):
            if edge_a[0] == edge_b[0] and edge_a[2] == edge_b[2] and edge_a[3] == edge_b[3]:
                continue
            else:
                return False
        return True

    def extract_dataflow(self, code, parser, lang):
        """
        remove comments, tokenize code and extract dataflow
        Args:
            code (_type_): _description_
            parser (_type_): _description_
            lang (_type_): _description_

        Returns:
            _type_: dataflow of input code
        """
        # remove comments
        try:
            code = remove_comments_and_docstrings(code, lang)
        except:
            pass
        # obtain dataflow
        if lang == "php":
            code = "<?php"+code+"?>"
        try:
            code_tokens = []
            code_to_index = defaultdict(lambda: -1)
            tree = parser[0].parse(bytes(code, 'utf8'))
            root_node = tree.root_node
            tokens_index = tree_to_token_index(root_node)
            code = code.split('\n')
            code_tokens = [index_to_code_token(x, code) for x in tokens_index]
            index_to_code = {}
            
            for idx, (index, code) in enumerate(zip(tokens_index, code_tokens)):
                index_to_code[index] = (idx, code)
                code_to_index[idx] = index
            try:
                DFG, _ = parser[1](root_node, index_to_code, {})
            except:
                DFG = []
            DFG = sorted(DFG, key=lambda x: x[1])
            indexs = set()
            for d in DFG:
                if len(d[-1]) != 0:
                    indexs.add(d[1])
                for x in d[-1]:
                    indexs.add(x)
            new_DFG = []
            for d in DFG:
                if d[1] in indexs:
                    new_DFG.append(d)
            dfg = new_DFG
        except:
            dfg = []
        return code_tokens, dfg, code_to_index

    def normalize_dataflow(self, dataflow, var_dict=None):

        if var_dict is None:
            var_dict = {}
            i = 1
        else:
            anon_var_list = [var_dict[x] for x in var_dict]
            var_ids = [int(re.findall('\d+', x)[0]) for x in anon_var_list]
            i = max(var_ids) + 1

        normalized_dataflow = []
        var_mapping = {}

        for item in dataflow:
            if i > 99:
                break
            var_name = item[0]
            relationship = item[2]
            par_vars_name_list = item[3]
            par_vars_idx_list = item[4]
            if var_name not in var_dict:
                var_dict[var_name] = f"<e{i}>"
                var_mapping[f"<e{i}>"] = item[1]
                i += 1
            elif var_name in var_dict and var_dict[var_name] not in var_mapping:
                var_mapping[var_dict[var_name]] = item[1]

        for item in dataflow:
            var_name = item[0]
            relationship = item[2]
            par_vars_name_list = item[3]
            par_vars_idx_list = item[4]
            for para_name, var_idx in zip(par_vars_name_list, par_vars_idx_list):
                if para_name not in var_dict:
                    var_dict[para_name] = f"<e{i}>"
                    var_mapping[f"<e{i}>"] = var_idx
                    i += 1
                elif para_name in var_dict and var_dict[para_name] not in var_mapping:
                    var_mapping[var_dict[para_name]] = var_idx
            if par_vars_name_list:
                normalized_dataflow.append((var_dict[var_name], relationship, tuple(
                    var_dict[x] for x in par_vars_name_list), item[1]))
            else:
                normalized_dataflow.append(
                    (var_dict[var_name], relationship, tuple(("<e0>", )), item[1]))

        return normalized_dataflow, var_mapping, var_dict


    def update_code(self, code, var_to_idx, idx_to_loc, scope):
        var_to_loc = {x: idx_to_loc[var_to_idx[x]] for x in var_to_idx}
        code = code.split('\n')
        updated_code = deepcopy(code)
        for var in var_to_loc:
            loc = var_to_loc[var]
            if not (scope[0] <= loc[0][0] < scope[1]) or not (scope[0] <= loc[1][0] < scope[1]):
                continue
            if loc[0][0] != loc[1][0]:
                continue
            true_var = code[loc[0][0]][loc[0][1]:loc[1][1]]

            tmp_rec = updated_code[scope[0]:scope[1]]
            updated_code[scope[0]:scope[1]] = [re.sub(
                '\\b' + re.escape(true_var) + '\\b', var, line) for line in updated_code[scope[0]:scope[1]]]
            if updated_code[scope[0]:scope[1]] == tmp_rec:
                updated_code[scope[0]:scope[1]] = [re.sub(
                    re.escape(true_var), var, line) for line in updated_code[scope[0]:scope[1]]]
        return "\n".join(updated_code)




    
class SimpleClsDataset(TextDataset):
    def __init__(self, tokenizer, pool, args, file_path, samplenum=-1):
        self.tokenizer = tokenizer
        if isinstance(tokenizer, T5Tokenizer):
            tokenizer_type = ""
        elif isinstance(tokenizer, RobertaTokenizer):
            tokenizer_type = "rb"
        else:
            tokenizer_type = "unk"
        savep = file_path.replace(".jsonl", tokenizer_type + ".simpexps")
        if os.path.exists(savep):
            logger.info("Loading examples from {}".format(savep))
            self.feats = torch.load(savep)
        else:
            logger.info("Reading examples from {}".format(file_path))
            examples = read_CC_examples(args, file_path, samplenum, tokenizer)
            logger.info(f"Tokenize examples: {file_path}")
            self.set_start_end_ids(examples)
            self.convert_examples_to_features((examples[7], tokenizer, args))
            self.feats = pool.map(self.convert_examples_to_features, \
                [(example, tokenizer, args) for example in examples])
            torch.save(self.feats, savep)

    def convert_examples_to_features(self, item):
        example, tokenizer, args = item
        # example.input_lines = example.input.split("<e0>")
        # labels_l = len(example.labels)
        # example.input_lines = example.input_lines[:labels_l]
        # for i in range(len(example.lines)):
        #     if example.labels[i] == 1:
        #         example.input_lines[i] = "<add>" + example.input_lines[i]
        #     elif example.labels[i] == 0:
        #         example.input_lines[i] = "<del>" + example.input_lines[i]
        # example.input = " ".join(example.input_lines)
        # input_ids = self.encode_remove(tokenizer, example.input, args)
        lines = example.lines
        labels = example.labels
        source_ids = []
        id_dict = {0: tokenizer.del_id,
                   1: tokenizer.add_id, 2: tokenizer.keep_id}
        for i, (line, label) in enumerate(zip(lines, labels)):
            if i == example.start_id:
                source_ids.append(tokenizer.start_id)
            if label == 0 or label == 1:
                source_ids.append(id_dict[label])
                source_ids.extend(line)
            if i == example.end_id:
                source_ids.append(tokenizer.end_id)
        
        exceed_l = len(source_ids) - args.max_source_length + 2
        if exceed_l > 0:
            halfexl = (exceed_l + 1) // 2
            source_ids = source_ids[halfexl:-halfexl]
        source_ids = source_ids[:args.max_source_length - 2]
        source_ids = [tokenizer.bos_id] + source_ids + [tokenizer.eos_id]
        pad_len = args.max_source_length - len(source_ids)
        source_ids += [tokenizer.pad_id] * pad_len
        example_id = example.idx
        y = example.y
        return ClsFeatures(example_id, source_ids, y)

    
class DQEClsDataset(TextDataset):
    def __init__(self, tokenizer, pool, args, file_path, samplenum=-1):
        self.tokenizer = tokenizer
        if isinstance(tokenizer, T5Tokenizer):
            tokenizer_type = ""
        elif isinstance(tokenizer, RobertaTokenizer):
            tokenizer_type = "rb"
        else:
            tokenizer_type = "unk"
        savep = file_path.replace(".jsonl", 'dqe_' + tokenizer_type + ".exps")
        if os.path.exists(savep):
            logger.info("Loading examples from {}".format(savep))
            examples = torch.load(savep)
        else:
            logger.info("Reading examples from {}".format(file_path))
            examples = read_review_examples(args, file_path, samplenum, tokenizer)
            logger.info(f"Tokenize examples: {file_path}")
            examples = pool.map(self.tokenize, \
                [(example, tokenizer, args) for example in examples])
            torch.save(examples, savep)
        logger.info("Convert examples to features...")
        self.set_start_end_ids(examples)
        self.feats = pool.map(self.convert_examples_to_features, \
            [(example, tokenizer, args) for example in examples])

    def convert_examples_to_features(self, item):
        example, tokenizer, args = item
        tmpfeature = self.gen_PL2NL_example(item)
        return ClsFeatures(tmpfeature.example_id, tmpfeature.source_ids, example.y)
    
    
class SimpleGenDataset(TextDataset):
    def __init__(self, tokenizer, pool, args, file_path, samplenum=-1):
        self.tokenizer = tokenizer
        if isinstance(tokenizer, T5Tokenizer):
            tokenizer_type = ""
        elif isinstance(tokenizer, RobertaTokenizer):
            tokenizer_type = "rb"
        else:
            tokenizer_type = "unk"
        savep = file_path.replace(".jsonl", tokenizer_type + ".simpgenexps")
        if os.path.exists(savep):
            logger.info("Loading examples from {}".format(savep))
            self.feats = torch.load(savep)
        else:
            logger.info("Reading examples from {}".format(file_path))
            data = read_jsonl(file_path)
            for i in range(len(data)):
                data[i]["idx"] = i
            logger.info(f"Tokenize examples: {file_path}")
            self.feats = pool.map(self.convert_examples_to_features, \
                [(dic, tokenizer, args) for dic in data])
            torch.save(self.feats, savep)


    def convert_examples_to_features(self, item):
        dic, tokenizer, args = item
        if "patch" in dic:
            diff= dic["patch"]
        elif "diff" in dic:
            diff = dic["diff"]
            
        if "msg" in dic:
            msg = dic["msg"]
        elif "nl" in dic:
            msg = dic["nl"]
        else:
            msg = ""

        regex = r"@@ -(\d+),(\d+) \+(\d+),(\d+) @@"

        difflines = diff.split("\n")
        matchres = re.match(regex, difflines[0])
        if matchres:
            difflines = difflines[1:]  # remove start @@

        difflines = [line for line in difflines if len(line.strip()) > 0]
        map_dic = {"-": 0, "+": 1, " ": 2}

        def f(s):
            if s in map_dic:
                return map_dic[s]
            else:
                return 2

        labels = [f(line[0]) for line in difflines]
        difflines = [line[1:].strip() for line in difflines]
        inputstr = ""
        for label, line in zip(labels, difflines):
            if label == 1:
                inputstr += "<add>" + line
            elif label == 0:
                inputstr += "<del>" + line
            else:
                inputstr += "<keep>" + line
        source_ids = self.encode_remove(tokenizer, inputstr, args)
        target_ids = []
        target_ids.append(tokenizer.msg_id)
        msg = self.encode_remove(tokenizer, msg, args)
        target_ids.extend(msg)
        source_ids, target_ids = self.pad_assert(
            source_ids, target_ids, args, tokenizer)
        input_labels = [-100] * len(source_ids)
        return ReviewFeatures(dic["idx"], source_ids, input_labels, target_ids, type="genmsg")

class SimpleCUPDataset(TextDataset):
    def __init__(self, tokenizer, pool, args, file_path, samplenum=-1):
        self.tokenizer = tokenizer
        if isinstance(tokenizer, T5Tokenizer):
            tokenizer_type = ""
        elif isinstance(tokenizer, RobertaTokenizer):
            tokenizer_type = "rb"
        else:
            tokenizer_type = "unk"
        savep = file_path.replace(".jsonl", tokenizer_type + ".simpcupexps")
        if os.path.exists(savep):
            logger.info("Loading examples from {}".format(savep))
            self.feats = torch.load(savep)
        else:
            logger.info("Reading examples from {}".format(file_path))
            data = read_jsonl(file_path)
            # data = [dic for dic in data if len(dic["patch"].split("\n")) <= 20]
            for i in range(len(data)):
                data[i]["idx"] = i
            logger.info(f"Tokenize examples: {file_path}")
            self.feats = pool.map(self.convert_examples_to_features, \
                [(dic, tokenizer, args) for dic in data])
            # self.feats = [self.convert_examples_to_features(
            #     (dic, tokenizer, args)) for dic in data]
            torch.save(self.feats, savep)


    def convert_examples_to_features(self, item):
        dic, tokenizer, args = item
        if "patch" in dic:
            diff= dic["patch"]
        elif "diff" in dic:
            diff = dic["diff"]
            
        if "msg" in dic:
            msg = dic["msg"]
        elif "nl" in dic:
            msg = dic["nl"]
        else:
            msg = ""
        old_msg = dic["old_nl"]
        
        regex = r"@@ -(\d+),(\d+) \+(\d+),(\d+) @@"

        difflines = diff.split("\n")
        matchres = re.match(regex, difflines[0])
        if matchres:
            difflines = difflines[1:]  # remove start @@

        difflines = [line for line in difflines if len(line.strip()) > 0]
        map_dic = {"-": 0, "+": 1, " ": 2}

        def f(s):
            if s in map_dic:
                return map_dic[s]
            else:
                return 2

        labels = [f(line[0]) for line in difflines]
        difflines = [line[1:].strip() for line in difflines]
        inputstr = ""
        inputstr += "<msg> " + old_msg + " "+ tokenizer.sep_token
        for label, line in zip(labels, difflines):
            if label == 1:
                inputstr += "<add> " + line
            elif label == 0:
                inputstr += "<del> " + line
                
        source_ids = self.encode_remove(tokenizer, inputstr, args)
        target_ids = []
        target_ids.append(tokenizer.msg_id)
        msg = self.encode_remove(tokenizer, msg, args)
        target_ids.extend(msg)
        source_ids, target_ids = self.pad_assert(
            source_ids, target_ids, args, tokenizer)
        input_labels = [-100] * len(source_ids)
        return ReviewFeatures(dic["idx"], source_ids, input_labels, target_ids, type="genmsg")


class SimpleJITDPDataset(TextDataset):
    def __init__(self, tokenizer, pool, args, file_path, samplenum=-1, oversample=False):
        self.tokenizer = tokenizer
        if isinstance(tokenizer, T5Tokenizer):
            tokenizer_type = ""
        elif isinstance(tokenizer, RobertaTokenizer):
            tokenizer_type = "rb"
        else:
            tokenizer_type = "unk"
        savep = file_path.replace(".jsonl", tokenizer_type + ".simpjitexps")
        if os.path.exists(savep):
            logger.info("Loading examples from {}".format(savep))
            self.feats = torch.load(savep)
            # print("")
        else:
            logger.info("Reading examples from {}".format(file_path))
            examples = read_jsonl(file_path)
            for i in range(examples.__len__()):
                examples[i]["idx"] = i 
                
            # features data
            features_filename = file_path.replace('changes', 'features')
            features_filename = features_filename.replace('.jsonl', '.pkl')
            features_data = pickle.load(open(features_filename, 'rb'))
            features_data = convert_dtype_dataframe(features_data, manual_features_columns)
            features_data = features_data[['commit_hash'] + manual_features_columns]
            manual_features = preprocessing.scale(features_data[manual_features_columns].to_numpy())
            assert len(manual_features) == len(examples), "The lengths of manual feautres and examples do not match"
            for i in range(examples.__len__()):
                examples[i]["MF"] = manual_features[i].tolist()
            
            logger.info(f"Tokenize examples: {file_path}")
            if args.debug:
                self.feats = [self.convert_examples_to_features((example, tokenizer, args)) \
                    for example in examples]
            else:
                self.feats = pool.map(self.convert_examples_to_features, \
                [(example, tokenizer, args) for example in examples])
            torch.save(self.feats, savep)
            

    def convert_examples_to_features(self, item):
        js, tokenizer, args = item
        msg_tokens = tokenizer.tokenize(js["msg"])
        msg_tokens = msg_tokens[:min(64, len(msg_tokens))]
        added_codes = [' '.join(line.split()) for line in js['added_code'].split('\n')]
        removed_codes = [' '.join(line.split()) for line in js['removed_code'].split('\n')]
        added_tokens, removed_tokens = [], []
        codes = '<add>'.join([line for line in added_codes if len(line)])
        added_tokens.extend(tokenizer.tokenize(codes))
        codes = '<del>'.join([line for line in removed_codes if len(line)])
        removed_tokens.extend(tokenizer.tokenize(codes))
        input_tokens = msg_tokens + ['<add>'] + added_tokens + ['<del>'] + removed_tokens
        input_tokens = input_tokens[:512 - 2]
        input_tokens = [tokenizer.cls_token] + input_tokens + [tokenizer.sep_token]
        source_ids = tokenizer.convert_tokens_to_ids(input_tokens)
        pad_len = args.max_source_length - len(source_ids)
        source_ids = source_ids + [tokenizer.pad_id] * pad_len
        example_id = js["idx"]
        manual_feature = js["MF"]
        y = int(js["y"])
        
        return JITDPFeatures(example_id, manual_feature, source_ids, y)


manual_features_columns = ['la', 'ld', 'nf', 'ns', 'nd', 'entropy', 'ndev',
                           'lt', 'nuc', 'age', 'exp', 'rexp', 'sexp', 'fix']


def convert_dtype_dataframe(df, feature_name):
    df['fix'] = df['fix'].apply(lambda x: float(bool(x)))
    df = df.astype({i: 'float32' for i in feature_name})
    return df


def read_jsonl(path):
    data = []
    with open(path) as f:
        for line in f:
            try:
                js = json.loads(line.strip())
            except:
                print("Error during reading json data.")
                continue
            data.append(js)
    return data


class ReviewExample(object):
    """A single training/test example."""

    def __init__(
            self, idx, oldf, diff, msg, cmtid, max_len, y, max_tgt_len, lang, tokenizer, skip_unavail=True):
        self.idx = idx      # idx is useless yet
        self.oldf = oldf
        self.diff = diff
        self.msg = msg
        self.cmtid = cmtid
        self.max_len = max_len
        self.y = y
        self.prevlines = []
        self.afterlines = []
        self.lines = []
        self.labels = []
        self.tokenized = False
        self.avail = False
        self.input = ""
        self.lang = lang
        self.max_tgt_len = max_tgt_len
        self.tokenizer = tokenizer
        self.align_and_clean(skip_unavail=True)
        self.postprocess()

    def tokenizer_encode(self, text, max_length=-1):
        if max_length == -1:
            text = self.tokenizer.encode(text)
        else:
            text = self.tokenizer.encode(
                text, max_length=max_length, truncation=True)

        if type(self.tokenizer) == T5Tokenizer:
            return text[:-1]
        elif type(self.tokenizer) == RobertaTokenizer:
            return text[1:-1]

        return None

    def postprocess(self):
        if not self.avail:
            return
        # Warning: lines is not self.lines
        # lines for rough length estimation (deprecated)
        # Since the tokenizer in encode_remove will limit the maximum length of the input, we deploy a more precise length calculation here
        lines = [self.tokenizer_encode(source_str, max_length=self.max_len - 2)
                 for source_str in self.lines]
        msg = self.tokenizer_encode(
            self.msg, max_length=self.max_tgt_len - 2)
        self.tokenized = True
        inputl = len(lines)  # line tag
        inputl += sum(map(len, lines))
        left, right = 0, len(lines)
        # compatibility for gen new code example.
        local_max_len = self.max_len - msg.__len__()
        while inputl > local_max_len:
            if left % 2 == 0:
                inputl -= len(lines[left]) + 1
                left += 1
            else:
                right -= 1
                inputl -= len(lines[right]) + 1
        lines = lines[left:right]
        self.lines = self.lines[left:right]
        self.labels = self.labels[left:right]
        prevlines = self.prevlines
        afterlines = self.afterlines
        prev_after_len = max(len(prevlines), len(afterlines))
        i = 0
        while inputl < local_max_len and i < prev_after_len:
            if i < len(prevlines):
                tokenized_prev_line = self.tokenizer_encode(
                    prevlines[-1-i], max_length=self.max_len)
                newl = inputl + len(tokenized_prev_line) + 1
                if newl > local_max_len:
                    break
                lines.insert(0, tokenized_prev_line)
                # self.lines.insert(0, prevlines[-1-i])
                self.labels.insert(0, -100)
                inputl = newl  # tag
            if i < len(afterlines):
                tokenized_after_line = self.tokenizer_encode(
                    afterlines[i], max_length=self.max_len)
                newl = inputl + len(tokenized_after_line) + 1
                if newl > local_max_len:
                    break
                lines.append(tokenized_after_line)
                self.labels.append(-100)
                inputl = newl  # tag
            i += 1
        assert inputl <= self.max_len, "Too long inputs."
        assert len(lines) == len(self.labels), "Not equal length."
        # self.input = "<e0>".join(self.lines)
        # self.input = "<e0>".join(self.lines)
        self.msg = msg
        self.lines = lines
        # self.prevlines, self.lines, self.afterlines, self.tokenizer = [], [], [], None  # save memory
        self.prevlines, self.input, self.afterlines, self.tokenizer = [
        ], "", [], None  # save memory

    def remove_space_clean(self, line):
        """
            Remove start and end empty chars.
        """
        rep = " \t\r"
        totallen = len(line)
        i = 0
        while i < totallen and line[i] in rep:
            i += 1
        j = totallen - 1
        while j >= 0 and line[j] in rep:
            j -= 1
        line = line[i: j + 1]
        return line

    def align_and_clean(self, skip_unavail=True):
        oldflines = self.oldf.split("\n")
        difflines = self.diff.split("\n")
        first_line = difflines[0]
        difflines = difflines[1:]
        difflines = [line for line in difflines if line !=
                     r"\ No newline at end of file"]
        regex = r"@@ -(\d+),(\d+) \+(\d+),(\d+) @@"
        matchres = re.match(regex, first_line)
        if matchres:
            startline, rangelen, startpos, endpos = matchres.groups()
            self.avail = True
        else:
            self.avail = False
            return
        startline, rangelen = int(startline) - 1, int(rangelen)
        endline = startline + rangelen
        self.prevlines = oldflines[:startline]
        self.afterlines = oldflines[endline:]
        for line in difflines:
            if line.startswith("-"):
                self.lines.append(line[1:])
                self.labels.append(0)
            elif line.startswith("+"):
                self.lines.append(line[1:])
                self.labels.append(1)
            else:
                self.lines.append(line)
                self.labels.append(2)
        self.prevlines = [self.remove_space_clean(
            line) for line in self.prevlines]
        self.afterlines = [self.remove_space_clean(
            line) for line in self.afterlines]
        self.lines = [self.remove_space_clean(
            line) for line in self.lines]  # diff lines
        self.msg = self.remove_space_clean(self.msg)
        self.prevlines = [line for line in self.prevlines if len(line) > 0]
        self.afterlines = [line for line in self.afterlines if len(line) > 0]
        # print("\n".join(self.prevlines))
        # print("\n\n\n\n")
        # print("\n".join(self.lines))
        # print("\n\n\n\n")
        # print("\n".join(self.afterlines))
        # print("\n\n\n\n")
        assert len(self.lines) == len(
            self.labels), "Not equal length in align."
        topack = list(
            zip(
                *[
                    (line, label)
                    for line, label in zip(self.lines, self.labels)
                    if len(line) > 0
                ]
            )
        )
        if topack == []:
            self.avail = False
            return
        else:
            self.lines, self.labels = topack
        # tuple->list, convenient for later operation
        self.lines = list(self.lines)
        self.labels = list(self.labels)


def read_review_examples(args, filename, data_num=-1, tokenizer=None, skip_unavail=True):
    """Read examples from filename."""
    examples = []
    idx = 0
    with open(filename, 'r', encoding='utf8') as f:
        for i, line in enumerate(f):
            # print(i)
            if args.debug and i > 100:
                break
            try:
                js = json.loads(line.strip())
            except:
                print("Error during reading json data.")
                continue
            # maxl = 200  # original
            maxl = args.max_source_length  # TEST by Bo
            if "y" not in js:
                js["y"] = 0
            if "msg" in js and len(js["msg"]) > 0:
                js["y"] = 1
            if "lang" not in js:
                js["lang"] = ""
            example = ReviewExample(
                idx=idx,
                oldf=js["oldf"],
                diff=js["patch"],
                msg=js["msg"] if "msg" in js else "",
                cmtid=js["cmtid"] if "cmtid" in js else "",
                max_len=maxl,
                y=int(js["y"]),
                max_tgt_len=args.max_target_length,
                lang=js["lang"],
                tokenizer=tokenizer,
                skip_unavail=True
                
            )
            if example.avail:
                examples.append(example)
                idx += 1
                if idx == data_num:
                    break
            else:
                # print(f"Passing {idx} because of invalid diff.")
                if skip_unavail is False:
                    examples.append(example)
                idx += 1
                if idx == data_num:
                    break
    return examples


def read_CC_examples(args, filename, data_num=-1, tokenizer=None):
    """Read examples from filename."""
    examples = []
    idx = 0
    with open(filename) as f:
        for line in f:
            try:
                js = json.loads(line.strip())
            except:
                print("Error during reading json data.")
                continue
            # maxl = 200  # original
            maxl = args.max_source_length  # TEST by Bo
            if "y" not in js:
                js["y"] = 0
            if ("nl" in js and len(js["nl"]) > 0) or ("msg" in js and len(js["msg"]) > 0):
                js["y"] = 1
            if "lang" not in js:
                js["lang"] = ""
            if "old_file" in js:
                example = ReviewExample(
                    idx=idx,
                    oldf=js["old_file"] if "old_file" in js else "",
                    diff=js["diff"],
                    msg=js["nl"] if "nl" in js else "",
                    cmtid=js["cmtid"] if "cmtid" in js else "",
                    max_len=maxl,
                    y=js["y"],
                    max_tgt_len=args.max_target_length,
                    lang=js["lang"],
                    tokenizer=tokenizer
                )
            elif "oldf" in js:
                example = ReviewExample(
                    idx=idx,
                    oldf=js["oldf"] if "oldf" in js else "",
                    diff=js["patch"],
                    msg=js["msg"] if "msg" in js else "",
                    cmtid=js["cmtid"] if "cmtid" in js else "",
                    max_len=maxl,
                    y=js["y"],
                    max_tgt_len=args.max_target_length,
                    lang=js["lang"],
                    tokenizer=tokenizer
                )
            if example.avail:
                examples.append(example)
                idx += 1
                if idx == data_num:
                    break
            else:
                idx += 1
                if idx == data_num:
                    break
    return examples



