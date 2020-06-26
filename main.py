# -*- coding: utf-8 -*-
# @Time : 2020/6/24 11:27 下午
# @Author : ligang
# @FileName: main.py
# @Email   : ucasligang@163.com
# @Software: PyCharm
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from Config import Config
from data import build_word2id, build_word2vec, load_corpus
from model import TextCNN

word2id = build_word2id(file='./dataset/word2id.txt', save_to_path=True)
word2vec = build_word2vec('./dataset/wiki_word2vec_50.bin', word2id, save_to_path='./dataset/word2vec.txt')
config = Config()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def train():
    train_contents, train_labels = load_corpus('./dataset/train.txt', word2id, max_sen_len=50)
    val_contents, val_labels = load_corpus('./dataset/validation.txt', word2id, max_sen_len=50)
    # 混合训练集和验证集
    contents = np.vstack([train_contents, val_contents])
    labels = np.concatenate([train_labels, val_labels])
    # 加载训练用的数据
    train_dataset = TensorDataset(torch.from_numpy(contents).type(torch.float),
                                  torch.from_numpy(labels).type(torch.long))
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=config.batch_size,
                                  shuffle=True, num_workers=2)
    model = TextCNN(config)
    if config.model_path:
        model.load_state_dict(torch.load(config.model_path))
    model.to(device)
    # 设置优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    # 设置损失函数
    criterion = nn.CrossEntropyLoss()
    # 定义训练过程
    for epoch in range(config.epochs):
        for batch_idx, (batch_x, batch_y) in enumerate(train_dataloader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            output = model(batch_x)
            loss = criterion(output, batch_y)
            if batch_idx % 200 == 0 & config.verbose:
                print("Train Epoch:{}[{}/{} ({:.0f}%)]\tLoss:{:.6f}".format(
                    epoch+1, batch_idx*len(batch_x), len(train_dataloader.dataset),
                    100.*batch_idx/len(train_dataloader), loss.item()
                ))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    # 保存模型
    torch.save(model.state_dict(), './models/model.pth')

def predict():
    test_contents, test_labels = load_corpus('./dataset/test.txt', word2id, max_sen_len=50)
    # 加载测试集
    test_dataset = TensorDataset(torch.from_numpy(test_contents).type(torch.float),
                                 torch.from_numpy(test_labels).type(torch.long))
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=config.batch_size,
                                 shuffle=False, num_workers=2)
    # 读取模型
    model = TextCNN(config)
    model.load_state_dict(torch.load(config.model_path))
    model.eval()
    model.to(device)

    # 测试过程
    count, correct = 0, 0
    for _, (batch_x, batch_y) in enumerate(test_dataloader):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        output = model(batch_x)
        # correct += (output.argmax(1) == batch_y).float().sum().item()
        correct += (output.argmax(1) == batch_y).sum().item()
        count += len(batch_x)

    # 打印准确率
    print('test accuracy is {:.2f}%.'.format(100 * correct / count))

if __name__ == '__main__':
    # train()
    predict()