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
from torchvision.models import resnet18
from torchvision.transforms import transforms

from base_utils import load_model
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
        self.human_vocab = human_vocab
        self.inv_machine_vocab = inv_machine_vocab


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
        self.return_alphas = False

    def forward(self, s: torch.Tensor, a):
        # after this, we have (batch, dim1) with a diff weight per each cell
        srep = s.reshape((-1, 1, n_s)).repeat([1, Tx, 1])
        concat = torch.cat((srep, a), dim=2)
        z = self.dense1(concat)
        z = self.tanh(z)
        z = self.dense2(z)
        z = self.relu(z)
        z = self.softmax(z)
        context = (z * a).sum(dim=1)
        if not self.return_alphas:
            return context
        else:
            return context, z


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
        result = self.attention(s, a)
        if self.attention.return_alphas:
            context, alphas = result
        else:
            context = result
            alphas = None
        s, c = self.post_lstm(context, (s, c))
        output = self.linear(s)
        output = self.softmax(output)
        return s, c, output, alphas


class MTModel(Module):

    def __init__(self):
        super().__init__()
        self.pre_lstm = nn.LSTM(input_size=37, hidden_size=n_a, bidirectional=True, batch_first=True)
        self.decoder = Decoder()
        if train_cfg['use_gpu']:
            self.decoder = self.decoder.cuda()
        self.alphas = []

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
            s, c, output, alpha = self.decoder(s, c, a)
            self.alphas.append(alpha)
            outputs[:, i, :] = output
        return outputs

    def predict(self, lib: MTLib, sentence):
        human_vocab = lib.human_vocab
        source = string_to_int(sentence, Tx, lib.human_vocab)

        source = np.array(list(map(lambda x: to_categorical(x, num_classes=len(human_vocab)), source)))
        source = source[np.newaxis, :]
        source = torch.from_numpy(source).cuda()
        prediction = self(source)
        prediction = np.argmax(prediction.detach().cpu().numpy(), axis=-1).squeeze()
        output = [lib.inv_machine_vocab[int(i)] for i in prediction]
        return output


def plot_attention_map(model: MTModel, lib, text, n_s=128, Tx=30, Ty=10):
    """
    Plot the attention map. not work for pytorch yet.

    """
    attention_map = np.zeros((10, 30))
    Ty, Tx = attention_map.shape

    # s0 = np.zeros((1, n_s))
    # c0 = np.zeros((1, n_s))

    encoded = np.array(string_to_int(text, Tx, lib.human_vocab)).reshape((1, 30))
    encoded = np.array(list(map(lambda x: to_categorical(x, num_classes=len(lib.human_vocab)), encoded)))
    model.alphas = []
    model.decoder.attention.return_alphas = True
    model(torch.from_numpy(encoded).cuda())
    r = model.alphas

    for t in range(Ty):
        for t_prime in range(Tx):
            attention_map[t][t_prime] = r[t][0, t_prime, 0]

    # Normalize attention map
    #     row_max = attention_map.max(axis=1)
    #     attention_map = attention_map / row_max[:, None]

    predicted_text = model.predict(lib, text)

    text_ = list(text)

    # get the lengths of the string
    input_length = len(text)
    output_length = Ty

    # Plot the attention_map
    plt.clf()
    f = plt.figure(figsize=(8, 8.5))
    ax = f.add_subplot(1, 1, 1)

    # add image
    i = ax.imshow(attention_map, interpolation='nearest', cmap='Blues')

    # add colorbar
    cbaxes = f.add_axes([0.2, 0, 0.6, 0.03])
    cbar = f.colorbar(i, cax=cbaxes, orientation='horizontal')
    cbar.ax.set_xlabel('Alpha value (Probability output of the "softmax")', labelpad=2)

    # add labels
    ax.set_yticks(range(output_length))
    ax.set_yticklabels(predicted_text[:output_length])

    ax.set_xticks(range(input_length))
    ax.set_xticklabels(text_[:input_length], rotation=45)

    ax.set_xlabel('Input Sequence')
    ax.set_ylabel('Output Sequence')

    # add grid and legend
    ax.grid()

    f.show()

    return attention_map


if __name__ == '__main__':
    lib = MTLib()
    model = MTModel().cuda() if train_cfg['use_gpu'] else MTModel()
    load_model(model, Adam(model.parameters()), 'data/models')
    model.eval()
    plot_attention_map(model, lib,
                       "Tuesday 09 Oct 1993", n_s=64)
