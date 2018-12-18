from progressbar import progressbar

from base_utils import History, AverageMeter, save_model, load_model
from mt_pytorch_utils import *


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
                x_var = Variable(x.cuda()) if train_cfg['use_gpu'] else Variable(x)
                y_var = Variable(y.cuda()) if train_cfg['use_gpu'] else Variable(y)
                optimizer.zero_grad()
                scores = model(x_var)
                loss = loss_fn(scores.reshape((-1, 11)), y_var.reshape((-1, 1)).squeeze())

                meters['loss'].update(loss.item())
                for k, f in metrics_functions.items():
                    result = f(y.detach().cpu().numpy().astype(np.int64),
                               scores.detach().cpu().numpy())
                    meters[k].update(result)
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
    return np.sum(np.argmax(y_pred, axis=2) == y_true) / (y_true.shape[0] * y_true.shape[1])


def train():
    lib = MTLib()
    trainset = MTDataset(lib, True)
    devset = MTDataset(lib, False)
    dataloaders = {'train': DataLoader(trainset, batch_size=100, shuffle=True),
                   'dev': DataLoader(devset, batch_size=100, shuffle=False)}
    model = MTModel().cuda() if train_cfg['use_gpu'] else MTModel()
    loss_fn = nn.CrossEntropyLoss().cuda() if train_cfg['use_gpu'] else nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.01)
    scheduler = lr_scheduler.StepLR(optimizer, 30)
    fit(model, loss_fn, optimizer, dataloaders, scheduler=scheduler,
        metrics_functions={'accuracy': compute_accuracy}, num_epochs=50)


def test():
    lib = MTLib()
    model = MTModel().cuda() if train_cfg['use_gpu'] else MTModel()
    load_model(model, Adam(model.parameters()), 'data/models')
    model.eval()
    EXAMPLES = ['3 May 1979', '5 April 09', '21th of August 2016', 'Tue 10 Jul 2007', 'Saturday May 9 2018',
                'March 3 2001', 'March 3rd 2001', '1 March 2001']
    for example in EXAMPLES:
        output = model.predict(lib, example)
        print("source:", example)
        print("output:", ''.join(output))


if __name__ == '__main__':
    # train()
    test()

