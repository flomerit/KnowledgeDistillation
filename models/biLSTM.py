# -*- coding: utf-8 -*-
"""
# @Time    : 2020/5/18 下午5:27
# @Author  : HOY
# @Email   : huangouyan@changingedu.com
# @File    : biLSTM.py
# @Software: PyCharm
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class biLSTM(nn.Module):  # nn.Module 是 PyTorch 中的一个重要基类，也是biLSTM子类的父类必须指明，用于构建神经网络模型，不可省略

    def __init__(self, config):  # 参数 config 用于传递模型的超参数等配置信息
        super(biLSTM, self).__init__()  # 调用了 biLSTM 类的父类 nn.Module 的 __init__ 方法（2.x的写法;3.x中super().__init__() 即可调用父类的初始化方法，而不需要显式地传递类名和 self 参数。）
        self.Embedding = nn.Embedding(21128,300)   # 将离散的词汇（比如单词或者字符）映射到连续的低维向量空间中。Embedding 层的作用是将一个大小为 21128 的词汇表中的每个单词（或者标记）映射成一个 300 维的稠密向量
        self.lstm = nn.LSTM(input_size=300, hidden_size=300,
                            num_layers=1, batch_first=True, dropout=0, bidirectional=True)
        # self.linear = nn.Linear(in_features=256, out_features=2)
        self.fc1 = nn.Linear(300*2, 192)
        self.fc2 = nn.Linear(192, config.num_classes)

    def forward(self, x, hidden=None):
        x = self.Embedding(x)
        lstm_out, hidden = self.lstm(x, hidden)     # LSTM 的返回很多
        out = self.fc1(lstm_out)
        activated_t = F.relu(out)
        linear_out = self.fc2(activated_t)
        linear_out = torch.max(linear_out, dim=1)[0]

        return linear_out
