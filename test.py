import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from torch import nn
from torch.nn import Module
from torch.autograd import backward, Variable
from torch.optim import lr_scheduler, Adam
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms

#
#
# class EncoderRNN(nn.Module):
#     def __init__(self, input_size, hidden_size):
#         super(EncoderRNN, self).__init__()
#         self.hidden_size = hidden_size
#
#         self.embedding = nn.Embedding(input_size, hidden_size)  # embeding成词向量
#         self.gru = nn.GRU(hidden_size, hidden_size)  # 注意，这里GRU的hide layer维度和embeding维度一样，但并不是必须的
#
#     def forward(self, input, hidden):
#         embedded = self.embedding(input).view(1, 1,
#                                               -1)  # (seq_len, batch, input_size)这是RNN的输入数据格式，这里只有1个时间步，但是为什么batch也是1？
#         output = embedded
#         output, hidden = self.gru(output, hidden)
#         return output, hidden
#
#     def initHidden(self):
#         result = Variable(torch.zeros(1, 1, self.hidden_size))
#         return result
#
#
# class AttnDecoderRNN(nn.Module):
#     def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
#         super(AttnDecoderRNN, self).__init__()
#         self.hidden_size = hidden_size
#         self.output_size = output_size
#         self.dropout_p = dropout_p
#         self.max_length = max_length
#
#         self.embedding = nn.Embedding(self.output_size, self.hidden_size)
#         self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
#         self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
#         self.dropout = nn.Dropout(self.dropout_p)
#         self.gru = nn.GRU(self.hidden_size, self.hidden_size)
#         self.out = nn.Linear(self.hidden_size, self.output_size)
#
#     def forward(self, input, hidden, encoder_outputs):
#         embedded = self.embedding(input).view(1, 1, -1)
#         embedded = self.dropout(embedded)
#
#         attn_weights = F.softmax(
#             self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)  # 连接输入的词向量和上一步的hide state并建立bp训练，他们决定了attention权重
#         attn_applied = torch.bmm(attn_weights.unsqueeze(0),
#                                  encoder_outputs.unsqueeze(0))  # 施加权重到所有的语义向量上
#
#         output = torch.cat((embedded[0], attn_applied[0]), 1)  # 加了attention的语义向量和输入的词向量共同作为输入，此处对应解码方式三+attention
#         output = self.attn_combine(output).unsqueeze(0)  # 进入RNN之前，先过了一个全连接层
#
#         output = F.relu(output)
#         output, hidden = self.gru(output, hidden)
#
#         output = F.log_softmax(self.out(output[0]), dim=1)  # 输出分类结果
#         return output, hidden, attn_weights
#
#     def initHidden(self):
#         result = Variable(torch.zeros(1, 1, self.hidden_size))
#         return result


if __name__ == '__main__':
    a = torch.rand((1, 50))
    out = nn.LSTMCell(input_size=50, hidden_size=30)(a)
    print()