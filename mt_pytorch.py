from progressbar import progressbar

import os

import numpy as np
from progressbar import progressbar
from torch.autograd import Variable

from base_utils import History, AverageMeter, save_model
from mt_pytorch_utils import *

use_gpu = False


def fit(model, loss_fn, optimizer, dataloaders, metrics_functions=None, num_epochs=1, scheduler=None, begin_epoch=0,
        save=True,
        save_model_dir='data/models', history=None, use_progressbar=False, plot_every_epoch=False):
    if metrics_functions is None:
        metrics_functions = {}
    if save and save_model_dir is None:
        raise Exception('save_model is True but no directory is specified.')
    if save:
        os.system('mkdir -p ' + save_model_dir)
    num_epochs += begin_epoch
    if history is None:
        history = History(['loss', *metrics_functions.keys()])
    for epoch in range(begin_epoch, num_epochs):
        print('Starting epoch %d / %d' % (epoch + 1, num_epochs))
        for phase in ['train', 'dev']:
            meters = {'loss': AverageMeter()}
            for k in metrics_functions.keys():
                meters[k] = AverageMeter()
            if phase == 'train':
                if scheduler is not None:
                    scheduler.step()
                model.train()
            else:
                model.eval()
            loaders = progressbar(dataloaders[phase]) if use_progressbar else dataloaders[phase]
            for data in loaders:
                x, y = data
                # print(x.shape, y.shape)
                nsamples = x.shape[0]
                x_var = Variable(x.cuda()) if use_gpu else Variable(x)
                y_var = Variable(y.cuda()) if use_gpu else Variable(y)
                optimizer.zero_grad()
                scores = model(x_var)
                # print(scores.shape)
                loss = loss_fn(scores.reshape((-1, 11)), y_var.reshape((-1, 1)).squeeze())

                meters['loss'].update(loss.item(), nsamples)
                for k, f in metrics_functions.items():
                    result = f(y.detach().cpu().numpy().astype(np.int64),
                               scores.detach().cpu().numpy())
                    meters[k].update(result, nsamples)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            s = 'Epoch {}/{}, {}, loss = {:.4f}'.format(epoch + 1, num_epochs, phase, meters['loss'].avg)

            for k in metrics_functions.keys():
                s += ', {} = {:.4f}'.format(k, meters[k].avg)
            print(s)
            history.records['loss'][phase].append(meters['loss'].avg)
            for k in metrics_functions.keys():
                history.records[k][phase].append(meters[k].avg)
        if save:
            save_model(model, optimizer, epoch, save_model_dir)
        if plot_every_epoch:
            history.plot()
    if not plot_every_epoch:
        history.plot()


def compute_accuracy(y_true, y_pred):
    return np.sum(np.argmax(y_pred, axis=1) == y_true) / y_true.shape[0]


if __name__ == '__main__':
    lib = MTLib()
    trainset = MTDataset(lib, True)
    devset = MTDataset(lib, False)
    dataloaders = {'train': DataLoader(trainset, batch_size=100, shuffle=True),
                   'dev': DataLoader(devset, batch_size=100, shuffle=False)}
    model = MTModel().cuda() if use_gpu else MTModel()
    print(list(model.named_parameters()))
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters())

    fit(model, loss_fn, optimizer, dataloaders,
        metrics_functions={'accuracy': compute_accuracy}, num_epochs=1)
