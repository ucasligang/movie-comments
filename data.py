# -*- coding: utf-8 -*-
# @Time : 2020/6/24 2:22 下午
# @Author : ligang
# @FileName: data.py
# @Email   : ucasligang@163.com
# @Software: PyCharm
from collections import Counter

import gensim
import numpy as np


def build_word2id(file, save_to_path=None):
    """
    :param file: word2id保存地址
    :param save_to_path: 保存训练语料库中的词组对应的word2vec到本地
    :return: None
    """
    word2id = {'_PAD_': 0}
    path = ['./dataset/train.txt', './dataset/validation.txt', './dataset/test.txt']

    for _path in path:
        with open(_path, encoding='utf-8') as f:
            for line in f.readlines():
                sp = line.strip().split()
                for word in sp[1:]:
                    if word not in word2id.keys():
                        word2id[word] = len(word2id)
    if save_to_path:
        with open(file, 'w', encoding='utf-8') as f:
            for w in word2id:
                f.write(w+'\t')
                f.write(str(word2id[w]))
                f.write('\n')
    return word2id


def build_word2vec(fname, word2id, save_to_path=None):
    """
    :param fname: 预训练的word2vec
    :param word2id: 语料文本中包含的词汇id
    :param save_to_path: 保存训练语料库中的词组对应的word2vec到本地
    :return: 语料文本中词汇集对应的word2vec向量{id:word2vec}
    """
    n_words = max(word2id.values())+1
    model = gensim.models.KeyedVectors.load_word2vec_format(fname, binary=True)
    word_vecs = np.array(np.random.uniform(-1., 1., [n_words, model.vector_size]))
    for word in word2id.keys():
        try:
            word_vecs[word2id[word]] = model[word]
        except KeyError:
            pass
    if save_to_path:
        with open(save_to_path, 'w', encoding='utf-8') as f:
            for vec in word_vecs:
                # vec = [str[w] for w in vec]
                f.write(' '.join(str(vec)))
                f.write('\n')
    return word_vecs


def cat_to_id(classes=None):
    """
    :param classes:分类标签；默认为0：pos，1：neg
    :return: {人类标签：id}
    """
    if not classes:
        classes = ['0', '1']
    cat2id = {cat: idx for (idx, cat) in enumerate(classes)}
    return classes, cat2id


def load_corpus(path, word2id, max_sen_len=50):
    """
    :param path: 样本语料库文件路径
    :param word2id: word to id的映射表
    :param max_sen_len: 映射向量最大长度
    :return: 文本映射后的向量contents 以及分类标签labels
    """
    _, cat2id = cat_to_id()
    contents, labels = [], []
    with open(path, encoding='utf-8') as f:
        for line in f.readlines():
            sp = line.strip().split()
            label = sp[0]
            content = [word2id.get(w, 0) for w in sp[1:]]
            content = content[:max_sen_len]
            if len(content) < max_sen_len:
                content += [word2id['_PAD_']] * (max_sen_len-len(content))
            labels.append(label)
            contents.append(content)
    couter = Counter(labels)
    print('总样本个数为:%d' % (len(labels)))
    print('各个类别样本数如下：')
    for w in couter:
        print(w, couter[w])

    contents = np.asarray(contents)
    labels = np.array([cat2id[l] for l in labels])

    return contents, labels


if __name__ == '__main__':
    word2id = build_word2id(file='./dataset/word2id.txt', save_to_path=True)
    word2vec = build_word2vec('./dataset/wiki_word2vec_50.bin', word2id, save_to_path='./dataset/word2vec.txt')
    print(word2vec.shape)
    assert word2vec.shape == (59290, 50)
    print(word2vec)
    print("train corpus load:")
    train_contents, train_labels = load_corpus('./dataset/train.txt', word2id, max_sen_len=50)
    print('\nvalidation corpus load:')
    val_contents, val_labels = load_corpus('./dataset/validation.txt', word2id, max_sen_len=50)
    print('\ntest corpus load:')
    test_contents, test_labels = load_corpus('./dataset/test.txt', word2id, max_sen_len=50)  # contents为id字符串的向量
    print(test_contents)
    print("labels:\n")
    print(test_labels)
