# coding=utf-8
"""
"""
import math
import os
import zipfile

from mxnet import nd


def sigmoid(x):
    return math.log(1 + math.exp(-x))


def load_data_jay_lyrics():
    """Load the Jay Chou lyric data set (available in the Chinese book)."""
    cwd = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_path = os.path.join(cwd, '../data')
    lyrics_file = os.path.join(data_path, 'jaychou_lyrics.txt.zip')

    with zipfile.ZipFile(lyrics_file) as zin:
        with zin.open('jaychou_lyrics.txt') as f:
            corpus_chars = f.read().decode('utf-8')
    corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
    corpus_chars = corpus_chars[0:10000]
    idx_to_char = list(set(corpus_chars))
    char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
    vocab_size = len(char_to_idx)
    corpus_indices = [char_to_idx[char] for char in corpus_chars]
    return corpus_indices, char_to_idx, idx_to_char, vocab_size


def to_onehot(X, size):
    """Represent inputs with one-hot encoding."""
    return [nd.one_hot(x, size) for x in X.T]


def grad_clipping(params, theta, ctx):
    """Clip the gradient."""
    if theta is not None:
        norm = nd.array([0], ctx)
        for param in params:
            norm += (param.grad ** 2).sum()
        norm = norm.sqrt().asscalar()
        if norm > theta:
            for param in params:
                param.grad[:] *= theta / norm


