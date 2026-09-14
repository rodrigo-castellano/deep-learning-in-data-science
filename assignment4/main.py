"""Assignment 4 -- vanilla RNN trained character by character on a text corpus.

    python main.py train        100 hidden units, seq_length 25, Adam, 7 epochs
    python main.py grad-check   analytic vs numerical gradients
    python main.py prepare      build the one-hot corpus from the raw text
"""
import argparse
import time

from model import model
from utils import load_data, lr_scheduler, optimizer, process_save_raw_data, save_state


def train():
    one_hot, _, ind_to_char, char_to_ind = load_data()
    rnn = model(100, one_hot.shape[0], ind_to_char, char_to_ind, seq_length=25)
    history = rnn.train_model(one_hot, optimizer=optimizer('Adam'),
                              lr_scheduler=lr_scheduler('decay', eta=.001, decay=1.0,
                                                        decay_epochs_freq=1),
                              dir='./results/' + time.strftime('%Y%m%d-%H%M%S') + '/',
                              n_epochs=7, n_batch=1)
    save_state(history)


def grad_check():
    one_hot, _, ind_to_char, char_to_ind = load_data()
    seq_length = 5
    X, Y = one_hot[:, :seq_length], one_hot[:, 1:seq_length + 1]
    rnn = model(m=5, K=X.shape[0], ind_to_char=ind_to_char,
                char_to_ind=char_to_ind, seq_length=seq_length)
    rnn.initialise_weights()
    rnn.check_grads(X, Y)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('task', choices=['train', 'grad-check', 'prepare'],
                   nargs='?', default='train')
    {'train': train, 'grad-check': grad_check,
     'prepare': process_save_raw_data}[p.parse_args().task]()
