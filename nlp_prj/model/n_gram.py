#!/usr/bin/env py_kp
# -*- coding: utf-8 -*-
# @Time    : 2019/9/23 15:05
# @Author  : song
# @File    : n_gram.py
import kenlm


class NGram(object):
    def __init__(self, model_path):
        self.model = kenlm.LanguageModel(model_path)

    def train(self):
        """
        text.txt :分词后
        训练 bin/lmplz -o 5  < text.txt  > text.arpa
        压缩 bin/build_binary -s text.arpa text.bin
        """

    def infer_perplexity(self, input_str=None):
        """
        困惑度
        :param input: "结构   脑部   什么  "
        :return:
        """
        sc = self.model.perplexity(input_str)
        return sc

    def infer_score(self, input_str=None):
        """
        bos=True, eos=True 属性，让 score 返回输入字符串的 log10 概率，即得分越高，句子的组合方式越好
        ngram value
        :param input_str:
        :return:
        """
        sc = self.model.score(input_str)
        return sc


if __name__ == '__main__':
    model_path = '/mnt/hgfs/share_ubuntu/resources/clean.zh.arpa.bin'
    model_path = "/home/ys/sample_ngram_kenlm.txt.arpa"
    model_path = "/home/ys/sample_ngram_kenlm.txt.bin"
    ngram = NGram(model_path=model_path)
    res = ngram.infer_perplexity("结构   脑部   什么  ")
    print(res)
    res = ngram.infer_score("结构   脑部   什么  ")
    print(res)
