# coding=utf-8

from mxnet import nd
import random
import zipfile


def data_iter_random(corpus_indices, batch_size, num_steps, ctx=None):
    """每次从数据里随机采样一个小批量，在随机采样中，每个样本是原始序列上任意截取的一段序列。
    相邻的两个随机小批量在原始序列上的位置不一定相毗邻。因此，我们无法用一个小批量最终时间步的隐藏状态来初始化下一个小批量的隐藏状态。
    在训练模型时，每次随机采样前都需要重新初始化隐藏状态。

    :param corpus_indices: 下标
    :param batch_size: 每个小批量的样本数
    :param num_steps: 每个样本所包含的时间步数。 在随机采样中，每个样本是原始序列上任意截取的一段序列
    """
    num_examples = (len(corpus_indices) - 1) // num_steps
    epoch_size = num_examples // batch_size
    example_indices = list(range(num_examples))
    random.shuffle(example_indices)


    def _data(pos):
        return corpus_indices[pos: pos + num_steps]


    for i in range(epoch_size):
        i = i * batch_size
        batch_indices = example_indices[i: i+batch_size]
        X = [_data(j * num_steps) for j in batch_indices]
        Y = [_data(j * num_steps + 1) for j in batch_indices]
        yield nd.array(X, ctx), nd.array(Y, ctx)


def data_iter_consecurtive(corpus_indices, batch_size, num_steps, ctx=None):
    corpus_indices = nd.array(corpus_indices, ctx=ctx)
    data_len = len(corpus_indices)
    batch_len = data_len // batch_size
    indices = corpus_indices[0: batch_size * batch_len].reshape((batch_size, batch_len))
    epoch_size = (batch_len - 1) // num_steps
    for i in range(epoch_size):
        i = i * num_steps
        X = indices[:, i: i + num_steps]
        Y = indices[:, i + 1: i + num_steps + 1]
        yield X, Y


def main():
    with zipfile.ZipFile('../../data/jaychou_lyrics.txt.zip') as zin:
        with zin.open('jaychou_lyrics.txt') as f:
            corpus_chars = f.read().decode('utf-8')
    corpus_chars[:40]

    # 建立字符索引
    idx_to_char = list(set(corpus_chars))
    char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
    vocab_size = len(char_to_idx)

    # 训练数据集中每个字符转化为索引
    corpus_indices = [char_to_idx[char] for char in corpus_chars]
    sample = corpus_indices[:20]
    print('chars:', ''.join([idx_to_char[idx] for idx in sample]))
    print('indices:', sample)

    my_seq = list(range(30))
    for X, Y in data_iter_random(my_seq, 2, 6):
        print('X:' X, '\nY:', Y, '\n')
