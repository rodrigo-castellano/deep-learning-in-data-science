"""Assignment 1 -- single-layer classifier on CIFAR-10, trained with mini-batch GD.

    python main.py train        one model, lambda=0.1, eta=0.001, decaying lr
    python main.py grad-check   analytic vs numerical gradients
    python main.py grid         lambda / eta grid search
"""
import argparse
import time

from functions import LoadData, model


def train():
    m = model(*LoadData())
    W, b, *_ = m.train_model(landa=.1, eta=.001, lr_epochs=30, lr_decay=True)
    m.plot_prob(time.strftime('%Y-%m-%d_%H-%M-%S'), m.X_test, m.Y_test, W, b)


def grad_check():
    m = model(*LoadData())
    m.zero_mean()
    m.initialise_weights()
    m.ComputeGradients(m.X_train, m.Y_train, m.W, m.b, landa=0)
    m.check_grads()


def grid():
    m = model(*LoadData())
    m.zero_mean()
    m.initialise_weights()
    param_grid = {'n_batch': [100], 'eta': [1e-3, 1e-4],
                  'n_epochs': [40], 'landa': [1, .1, .001]}
    m.GridSearch(param_grid, m.X_train, m.Y_train, m.X_val, m.Y_val,
                 m.X_test, m.Y_test, m.W, m.b, lr_epochs=30, lr_decay=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('task', choices=['train', 'grad-check', 'grid'], nargs='?', default='train')
    {'train': train, 'grad-check': grad_check, 'grid': grid}[p.parse_args().task]()
