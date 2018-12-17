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
        y = self.lib.Yoh[idx]
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
        self.denses1 = [nn.Linear(in_features=2 * n_a + n_s, out_features=10) for _ in range(Tx)]
        self.tanh = nn.Tanh()
        self.denses2 = [nn.Linear(in_features=10, out_features=1) for _ in range(Tx)]
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, s: torch.Tensor, a):
        # after this, we have (batch, dim1) with a diff weight per each cell
        srep = s.reshape((1, n_s, 1)).repeat([1, 1, Tx])
        concat = torch.cat((srep, a))
        z = [self.denses1[i](concat[i]) for i in range(Tx)]
        z = [self.tanh(z[i]) for i in range(Tx)]
        z = [self.denses2[i](z[i]) for i in range(Tx)]
        z = [self.relu(z[i]) for i in range(Tx)]
        z = torch.Tensor(z)
        z = self.softmax(z)
        return z @ a


class MTModel(Module):

    def __init__(self):
        super().__init__()
        self.pre_lstm = nn.LSTM(input_size=37, hidden_size=n_a, bidirectional=True, batch_first=True)
        self.attentions = [Attention() for _ in range(Ty)]
        self.post_lstms = [nn.LSTMCell(input_size=n_a, hidden_size=n_s)]
        self.linears = [nn.Linear(in_features=n_s, out_features=11) for _ in range(Ty)]
        self.softmax = nn.Softmax()

    def forward(self, x):
        a, _ = self.pre_lstm(x)
        s = torch.zeros((1, n_s, 1))
        c = None
        outputs = []
        for i in range(Ty):
            context = self.attentions[i](s, a)
            s, c = self.post_lstms[i](context, (s, c))
            output = self.linears[i](s)
            output = self.softmax(output)
            outputs.append(output)
        return outputs


if __name__ == '__main__':
    lib = MTLib()
    trainset = MTDataset(lib, True)
    model = MTModel()
    x, y = trainset[0]
    x = x.unsqueeze(0)
    model(x)
