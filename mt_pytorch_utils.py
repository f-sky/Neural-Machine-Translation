import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from torch import nn
from torch.nn import Module, Parameter
from torch.autograd import backward, Variable
from torch.nn.functional import softmax
from torch.optim import lr_scheduler, Adam
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms

from cfg import train_cfg
from nmt_utils import *

n_a = 32
n_s = 64
Tx = 30
Ty = 10


class MTLib:

    def __init__(self) -> None:
        super().__init__()
        m = 10000
        dataset, human_vocab, machine_vocab, inv_machine_vocab = load_dataset(m)
        Tx = 30
        Ty = 10
        X, Y, Xoh, Yoh = preprocess_data(dataset, human_vocab, machine_vocab, Tx, Ty)
        self.X = X
        self.Y = Y
        self.Xoh = Xoh
        self.Yoh = Yoh
        # print(X.shape)
        # print(Y.shape)
        # print(Xoh.shape)
        # print(Yoh.shape)


class MTDataset(Dataset):

    def __init__(self, lib: MTLib, train) -> None:
        super().__init__()
        self.lib = lib
        self.train = train
        self.num_total = self.lib.X.shape[0]
        self.num_train = int(0.8 * self.num_total)
        self.num_dev = self.num_total - self.num_train

    def __getitem__(self, index):
        idx = index if self.train else self.num_train + index
        x = self.lib.Xoh[idx]
        y = self.lib.Y[idx]
        return torch.FloatTensor(x), torch.LongTensor(y)

    def __len__(self):
        return self.num_train if self.train else self.num_dev


def new_parameter(*size):
    out = Parameter(torch.FloatTensor(*size))
    torch.nn.init.xavier_normal(out)
    return out


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.dense1 = nn.Linear(in_features=2 * n_a + n_s, out_features=10)
        self.tanh = nn.Tanh()
        self.dense2 = nn.Linear(in_features=10, out_features=1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, s: torch.Tensor, a):
        # after this, we have (batch, dim1) with a diff weight per each cell
        srep = s.reshape((-1, 1, n_s)).repeat([1, Tx, 1])
        concat = torch.cat((srep, a), dim=2)
        z = self.dense1(concat)
        z = self.tanh(z)
        z = self.dense2(z)
        z = self.relu(z)
        # z = torch.cat(z, dim=1)
        z = self.softmax(z)
        # z = z.unsqueeze(-1)
        return (z * a).sum(dim=1)


class Decoder(Module):

    def __init__(self):
        super().__init__()
        self.attention = Attention()
        if train_cfg['use_gpu']:
            self.attention = self.attention.cuda()
        self.post_lstm = nn.LSTMCell(input_size=n_s, hidden_size=n_s)
        self.linear = nn.Linear(in_features=n_s, out_features=11)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, s, c, a):
        context = self.attention(s, a)
        s, c = self.post_lstm(context, (s, c))
        output = self.linear(s)
        output = self.softmax(output)
        return s, c, output


class MTModel(Module):

    def __init__(self):
        super().__init__()
        self.pre_lstm = nn.LSTM(input_size=37, hidden_size=n_a, bidirectional=True, batch_first=True)
        self.decoder = Decoder()
        if train_cfg['use_gpu']:
            self.decoder = self.decoder.cuda()

    def forward(self, x):
        a, _ = self.pre_lstm(x)
        batch_size = a.shape[0]
        s = torch.zeros((a.shape[0], n_s), requires_grad=True)
        c = torch.zeros((a.shape[0], n_s), requires_grad=True)
        if train_cfg['use_gpu']:
            s = s.cuda()
            c = c.cuda()
        outputs = torch.zeros((batch_size, Ty, 11))
        outputs = outputs.cuda() if train_cfg['use_gpu'] else outputs
        for i in range(Ty):
            s, c, output = self.decoder(s, c, a)
            outputs[:, i, :] = output
        return outputs


if __name__ == '__main__':
    lib = MTLib()
    trainset = MTDataset(lib, True)
    model = MTModel()
    x, y = trainset[0]
    x = x.unsqueeze(0)
    # x = x.unsqueeze(0).repeat([2, 1, 1])
    print(x.shape,y.shape)
    outputs = model(x)
    print(outputs.shape)
    print(y.shape)
