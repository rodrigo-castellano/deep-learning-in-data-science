"""Assignment 2 -- two-layer network on CIFAR-10, cyclical learning rate.

    python main.py train        lambda=0.1, eta=0.001, 20 epochs
    python main.py grad-check   analytic vs numerical gradients
"""
import argparse

from functions import LoadBatch, model, zero_mean


def _data():
    train, val, test = LoadBatch('data_batch_1.mat', 'data_batch_2.mat', 'test_batch.mat')
    X = zero_mean(train['data'], val['data'], test['data'])
    return (*X, train['one_hot'], val['one_hot'], test['one_hot'])


def train():
    model(*_data()).train_model(landa=.1, eta=.001, n_epochs=20)


def grad_check():
    m = model(*_data())
    m.ComputeGradients(m.X_train, m.Y_train, m.W, m.b, landa=0)
    m.check_grads()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('task', choices=['train', 'grad-check'], nargs='?', default='train')
    {'train': train, 'grad-check': grad_check}[p.parse_args().task]()
