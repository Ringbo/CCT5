# encoding=utf-8

import re
import nltk
from typing import List


class Tokenizer:
    @classmethod
    def camel_case_split(cls, identifier):
        return re.sub(r'([A-Z][a-z])', r' \1', re.sub(r'([A-Z]+)', r' \1', identifier)).strip().split()

    @classmethod
    def tokenize_identifier_raw(cls, token, keep_underscore=True):
        regex = r'(_+)' if keep_underscore else r'_+'
        id_tokens = []
        for t in re.split(regex, token):
            if t:
                id_tokens += cls.camel_case_split(t)
        return list(filter(lambda x: len(x) > 0, id_tokens))

    @classmethod
    def tokenize_desc_with_con(cls, desc: str) -> List[str]:
        def _tokenize_word(word):
            new_word = re.sub(r'([-!"#$%&\'()*+,./:;<=>?@\[\\\]^`{|}~])', r' \1 ', word)
            subwords = nltk.word_tokenize(new_word)
            new_subwords = []
            for w in subwords:
                new_subwords += cls.tokenize_identifier_raw(w, keep_underscore=True)
            return new_subwords

        tokens = []
        for word in desc.split():
            if not word:
                continue
            tokens += " <con> ".join(_tokenize_word(word)).split()
        return tokens