import argparse
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
#os.environ['HF_HOME'] = '/root/autodl-tmp/cache/'   # AutoDL数据盘
# os.environ['HF_HOME'] = 'D:/tmp/cache'
os.environ["HF_HOME"] = "/mnt/d/tmp/cache"
from collections import defaultdict
import re
import math
import numpy
import json
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("roberta-large")

class FindNgrams:
    def __init__(self, min_count=0, min_pmi=0, language='en'):
        self.min_count = min_count
        self.min_pmi = min_pmi
        self.words = defaultdict(int)
        self.ngrams, self.pairs = defaultdict(int), defaultdict(int)
        self.total = 0.
        self.language = language

    def text_filter(self, sentence):
        cleaned_text = []
        index = 0
        for i, w in enumerate(sentence):
            if re.match(u'[^\u0600-\u06FF\u0750-\u077F\u4e00-\u9fa50-9a-zA-Z]+', w):
                if i > index:
                    cleaned_text.append([w.lower() for w in sentence[index:i]])
                index = 1 + i
        if index < len(sentence):
            cleaned_text.append([w.lower() for w in sentence[index:]])
        return cleaned_text

    def count_ngram(self, texts, n):
        self.ngrams = defaultdict(int)
        for sentence in texts:
            sub_sentence = sentence.split()
            for i in range(n):
                n_len = i + 1
                for j in range(len(sub_sentence) - i):
                    ngram = tuple([w for w in sub_sentence[j: j+n_len]])
                    self.ngrams[ngram] += 1
        self.ngrams = {i:j for i, j in self.ngrams.items() if j > self.min_count}

    def find_ngrams_pmi(self, texts, n, freq_threshold):
        for sentence in texts:
            # 确保句子不是空字符串或只包含空白字符
            if not sentence or not sentence.strip():
                continue
            sub_sentence = sentence.split()
            # 检查 sub_sentence 是否为空
            if len(sub_sentence) > 0:
                self.words[sub_sentence[0]] += 1
            else:
                continue
            for i in range(len(sub_sentence)-1):
                self.words[sub_sentence[i + 1]] += 1
                self.pairs[(sub_sentence[i], sub_sentence[i+1])] += 1
                self.total += 1
        self.words = {i: j for i, j in self.words.items() if j > self.min_count}
        self.pairs = {i: j for i, j in self.pairs.items() if j > self.min_count}

        min_mi = math.inf
        max_mi = -math.inf

        self.strong_segments = set()
        for i, j in self.pairs.items():
            if i[0] in self.words and i[1] in self.words:
                mi = math.log(self.total * j / (self.words[i[0]] * self.words[i[1]]))
                if mi > max_mi:
                    max_mi = mi
                if mi < min_mi:
                    min_mi = mi
                if mi >= self.min_pmi:
                    self.strong_segments.add(i)

        self.ngrams = defaultdict(int)
        for sentence in texts:
            if not sentence or not sentence.strip():
                continue
            sub_sentence = sentence.split()
            # 检查 sub_sentence 是否为空
            s = [sub_sentence[0]]
            for i in range(len(sub_sentence)-1):
                if (sub_sentence[i], sub_sentence[i+1]) in self.strong_segments:
                    s.append(sub_sentence[i+1])
                else:
                    self.ngrams[tuple(s)] += 1
                    s = [sub_sentence[i+1]]
        self.ngrams = {i:j for i, j in self.ngrams.items() if j > self.min_count and len(i) <= n}

        self.renew_ngram_by_freq(texts, freq_threshold, n)

    def renew_ngram_by_freq(self, all_sentences, min_feq, ngram_len=10):
        new_ngram2count = {}
        new_all_sentences = []

        for sentence in all_sentences:
            sentence = sentence.split()
            sen = sentence
            for i in range(len(sen)):
                for n in range(1, ngram_len + 1):
                    if i + n > len(sentence):
                        break
                    n_gram = tuple(sentence[i: i + n])
                    if n_gram not in self.ngrams:
                        continue
                    if n_gram not in new_ngram2count:
                        new_ngram2count[n_gram] = 1
                    else:
                        new_ngram2count[n_gram] += 1
        self.ngrams = {gram: c for gram, c in new_ngram2count.items() if c > min_feq}

if __name__ == '__main__':
    ngram_list = []
    dataset = 'data/data2text_data_all.json'
    output_dir = 'data/vocabulary/vocab_data2text.txt'
    ngram = 5
    min_count = 5
    min_pmi = 1
    ngram_freq_threshold = 5
    delete_special_symbol = True

    print('dataset: ', dataset)

    data = json.load(open(dataset, 'r', encoding='utf-8'))

    sentence_list = [item['input_context'] for item in data]

    ngram_finder = FindNgrams(min_count=min_count, min_pmi=min_pmi)
    ngram_finder.find_ngrams_pmi(sentence_list, ngram, ngram_freq_threshold)

    ngram_type_count = [0 for _ in range(ngram)]
    ngram_finder.ngrams = dict(sorted(ngram_finder.ngrams.items(), key = lambda kv:(kv[1], kv[0]), reverse=True))  # sort

    count = 0

    with open(output_dir, 'w', encoding='utf-8') as f_write:
        for w, c in ngram_finder.ngrams.items():
            count += 1
            s = ""
            for word_index in range(len(w)):
                s += w[word_index]+" "
            s = s.strip()
            i = len(s)
            if delete_special_symbol:
                while i > 0:
                    if s[i-1].isalnum():
                        break
                    i -= 1
            s = s[0:i]
            if s not in ngram_list and len(s) > 0:
                if s not in ngram_list:
                    ngram_list.append(s)
                f_write.write(s + '\n')
        print(str(ngram_type_count))