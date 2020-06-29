# -*- coding: utf-8 -*-
# @Time : 2020/6/24 2:18 下午
# @Author : ligang
# @FileName: Config.py
# @Email   : ucasligang@163.com
# @Software: PyCharm
from data import build_word2id, build_word2vec, load_corpus

word2id = build_word2id(file='./dataset/word2id.txt', save_to_path=True)
word2vec = build_word2vec('./dataset/wiki_word2vec_50.bin', word2id, save_to_path='./dataset/word2vec.txt')


class Config():
    update_w2v = True  # 是否在训练中更新w2v
    vocab_size = 59290  # 词汇量，与word2id中的词汇量一致
    n_class = 2  # 分类数：分别为pos和neg
    embedding_dim = 50  # 词向量维度
    drop_keep_prob = 0.5  # dropout层，参数keep的比例
    num_filters = 256  # 卷积层filter的数量
    kernel_size = 3  # 卷积核的尺寸
    pretrained_embed = word2vec  # 预训练的词嵌入模型

    learning_rate = 0.001  # 学习率
    batch_size = 128  # 训练批量
    epochs = 10  # 训练轮数
    model_path = "./models/model.pth"  # 预训练模型路径
    verbose = True  # 打印训练过程
