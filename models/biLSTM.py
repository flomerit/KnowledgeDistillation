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

    def __init__(self, config):  # 参数 config 用于传递模型的超参数等配置信息。定义了模型的各个层，包括嵌入层（Embedding）、双向 LSTM 层（lstm）、两个全连接层（fc1 和 fc2）
        super(biLSTM, self).__init__()  # 调用了 biLSTM 类的父类 nn.Module 的 __init__ 方法（2.x的写法;3.x中super().__init__() 即可调用父类的初始化方法，而不需要显式地传递类名和 self 参数。）
        self.Embedding = nn.Embedding(21128,300)   
    # 将一个索引数组或张量转换为一个表示固定大小的稠密向量的嵌入。生成词汇列表，21128个词汇，每个词汇300维。将输入的张量数组映射成词汇表的向量
        self.lstm = nn.LSTM(input_size=300, hidden_size=300,
                            num_layers=1, batch_first=True, dropout=0, bidirectional=True)
    '''
    参数：
    input_size：输入数据的特征维度。
    hidden_size：隐藏状态的特征维度。
    num_layers：LSTM 层的层数。默认值是 1。
    bias：如果为 False，那么 LSTM 层将不会使用偏置。默认值是 True。
    batch_first：如果为 True，则输入和输出张量的形状应该是 (batch_size, seq_len, feature_dim)，否则形状应该是 (seq_len, batch_size, feature_dim)。默认值是 False。
    dropout：如果非零，将在 LSTM 层的输出上应用丢弃。默认值是 0。正则化技术，在LSTM层的输出上应用丢弃意味着在每次前向传播中，以概率 dropout随机将一些神经元的输出设置为零来防止过拟合.
    bidirectional：如果为 True，则 LSTM 将是双向的。默认值是 False。
    '''
        self.linear = nn.Linear(in_features=256, out_features=2) # 输入特征的维度从 256 维降到 2 维
        self.fc1 = nn.Linear(300*2, 192)
        self.fc2 = nn.Linear(192, config.num_classes)  # 将模型的最后一层线性层的输出维度设置为类别数量
    def forward(self, x, hidden=None):  # 定义了数据在模型中的正向传播过程
        x = self.Embedding(x)   # .Embedding() 方法的输入通常是表示离散标签或索引的张量
        lstm_out, hidden = self.lstm(x, hidden)     # LSTM 的返回很多，拼接？
        out = self.fc1(lstm_out)
        activated_t = F.relu(out)
        linear_out = self.fc2(activated_t)
        linear_out = torch.max(linear_out, dim=1)[0] # 每行最大值和它的索引，[0]取第一个

        return linear_out
