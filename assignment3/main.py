"""Assignment 3 -- k-layer network on CIFAR-10 with batch normalisation.

    python main.py baseline      2x50 hidden, cyclical lr, batch norm
    python main.py adam          Adam + exponential lr decay
    python main.py adaptive-bn   batch-norm statistics learned during training
    python main.py wide          2x300 hidden
    python main.py he-init       He weight initialisation
    python main.py grad-check    analytic vs numerical gradients
    python main.py search --coarse | --fine    lambda random search
"""
import argparse
import random
import time

import numpy as np

from model import model
from utils import (GridSearch, LoadBatch_1_2, LoadData, lr_scheduler, optimizer,
                   save_state, zero_mean)

LANDA = 0.006655          # best value found by the random search below
HIDDEN = [50, 50]


def _data():
    X_train, X_val, X_test, Y_train, Y_val, Y_test = LoadData()
    X_train, X_val, X_test = zero_mean(X_train, X_val, X_test)
    return X_train, X_val, X_test, Y_train, Y_val, Y_test


def _run(opt, sched, n_hidden=HIDDEN, **kw):
    X_train, X_val, X_test, Y_train, Y_val, Y_test = _data()
    m = model(n_hidden, X_train.shape[0], Y_train.shape[0])
    history = m.train_model(X_train, Y_train, X_val, Y_val, X_test, Y_test,
                            optimizer=opt, lr_scheduler=sched, landa=LANDA,
                            dir='./results/' + time.strftime('%Y%m%d-%H%M%S') + '/',
                            batchnorm=True, **kw)
    save_state(history)


def _cyclic(n_cycles=1):
    return lr_scheduler('cyclic', eta_min=1e-5, eta_max=.1, n_cycles=n_cycles)


def baseline():     _run(optimizer('SGD'), _cyclic(1))
def adaptive_bn():  _run(optimizer('SGD'), _cyclic(1), train_BN=True)
def he_init():      _run(optimizer('SGD'), _cyclic(2), weight_ini='He')
def wide():         _run(optimizer('SGD'), _cyclic(2), n_hidden=[300, 300], n_epochs=20)


def adam():
    _run(optimizer('Adam', beta1=.9, beta2=.999),
         lr_scheduler('decay', eta=.001, decay=.9, decay_epochs_freq=1), n_epochs=20)


def grad_check():
    X_train, _, _, Y_train, _, _ = LoadBatch_1_2()
    X, Y = X_train[:10, :5], Y_train[:10, :5]
    model(HIDDEN, X.shape[0], Y.shape[0]).check_grads(X, Y, batchnorm=True)


def search(fine=False):
    lo, hi = (-3, -2) if fine else (-5, -1)
    landas = [np.round(10 ** (lo + (hi - lo) * random.random()), 6) for _ in range(8)]
    GridSearch({'n_epochs': [40], 'n_batch': [100], 'landa': landas,
                'n_hidden': [HIDDEN], 'batchnorm': [True]},
               *_data_for_search(), optimizer('SGD'), _cyclic(2))


def _data_for_search():
    X_train, X_val, X_test, Y_train, Y_val, Y_test = _data()
    return X_train, Y_train, X_val, Y_val, X_test, Y_test


if __name__ == '__main__':
    np.random.seed(300)
    random.seed(300)
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('task', nargs='?', default='baseline',
                   choices=['baseline', 'adam', 'adaptive-bn', 'wide', 'he-init',
                            'grad-check', 'search'])
    p.add_argument('--fine', action='store_true', help='narrow lambda range (search only)')
    a = p.parse_args()
    if a.task == 'search':
        search(a.fine)
    else:
        {'baseline': baseline, 'adam': adam, 'adaptive-bn': adaptive_bn,
         'wide': wide, 'he-init': he_init, 'grad-check': grad_check}[a.task]()
