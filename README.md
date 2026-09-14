# Deep Learning in Data Science

Neural networks written from scratch in NumPy — no autograd, no deep learning
framework. Every gradient is derived by hand and checked numerically against a
finite-difference approximation before training.

Coursework for **DD2424 Deep Learning in Data Science**, KTH Royal Institute of
Technology, spring 2022.

| | Model | Trained on | Key ideas |
|---|---|---|---|
| [1](assignment1) | Single-layer classifier | CIFAR-10 | softmax + cross-entropy, mini-batch gradient descent, L2 regularisation, learning-rate decay |
| [2](assignment2) | Two-layer network | CIFAR-10 | cyclical learning rates, random search over λ, data augmentation by horizontal flips |
| [3](assignment3) | k-layer network | CIFAR-10 | batch normalisation, He initialisation, Adam, learned BN statistics |
| [4](assignment4) | Vanilla RNN | *Harry Potter and the Goblet of Fire* | backpropagation through time, character-level text synthesis |

Each assignment's brief and the report written for it are in [`assignments/`](assignments).

## Running

```bash
pip install numpy scipy matplotlib scikit-learn tqdm

cd assignment3
python main.py grad-check     # verify the analytic gradients first
python main.py baseline       # then train
```

Every assignment exposes its experiments as named tasks; `python main.py --help`
lists them. Results are written to `results/<timestamp>/` as loss curves,
accuracy curves and a montage of the learnt weights, alongside an `info.txt`
recording the hyperparameters that produced them.

## Data

Not included here — download it into a `Dataset/` folder inside the assignment
you want to run.

- **Assignments 1–3** expect the MATLAB version of CIFAR-10 in
  `Dataset/cifar-10-batches-mat/` ([cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)).
- **Assignment 4** expects `Dataset/goblet_book.txt`. Run `python main.py prepare`
  once to build the one-hot corpus from it.

## Gradient checking

Each model implements `check_grads()`, comparing the analytic gradients against
centred finite differences using the relative error

$$\frac{|g_a - g_n|}{\max(\varepsilon,\ |g_a| + |g_n|)}$$

Run it before trusting any training run — it is the only thing standing between
a correct implementation and one that silently learns the wrong function.

## Course project

The project, a BERT-based classifier for the *Natural Language Processing with
Disaster Tweets* Kaggle competition, lives in its own repository:
[Disaster_tweets](https://github.com/rodrigo-castellano/Disaster_tweets).
