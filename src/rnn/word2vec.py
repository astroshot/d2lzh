# coding=utf-8

import collections
import math
import os
import random
import sys
import time
import zipfile

import d2l
from mxnet import autograd, gluon, nd
from mxnet.gluon import data as gdata, loss as gloss, nn


class Word2Vector(object):

    def __init__(self):
        self.raw_data = None
        self.counter = None
        self.idx_to_token = None
        self.token_to_idx = None
        self.dataset = None
        self.subsampled_dataset = None
        self.centers = None
        self.contexts = None
        self.negatives = None
        self.data_iter = None
        self.batch_size = None
        self.embed_size = 100
        self.net = None

    def load_data(self):
        cwd = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        data_path = os.path.join(cwd, '../data')
        ptb_file = os.path.join(data_path, 'ptb.zip')

        with zipfile.ZipFile(ptb_file, 'r') as zin:
            zin.extractall(data_path)

        in_file = os.path.join(data_path, 'ptb', 'ptb.train.txt')
        with open(in_file, 'r') as f:
            lines = f.readlines()
            # st means sentence
            raw_data = [st.split() for st in lines]
            self.raw_data = raw_data
            return raw_data

    def make_index(self):
        raw_dataset = self.raw_data
        # tk means token
        counter = collections.Counter([tk for st in raw_dataset for tk in st])
        # 保留在数据集中出现次数超过 5 次的单词
        counter = dict(filter(lambda x: x[1] >= 5, counter.items()))
        self.counter = counter

        self.idx_to_token = [tk for tk, _ in counter.items()]
        self.token_to_idx = {tk: idx for idx, tk in enumerate(self.idx_to_token)}
        self.dataset = [[self.token_to_idx[tk] for tk in st if tk in self.token_to_idx] for st in raw_dataset]
        num_tokens = sum([len(st) for st in self.dataset])

        def discard(index):
            return random.uniform(0, 1) < 1 - math.sqrt(1e-4 / counter[self.idx_to_token[index]] * num_tokens)

        self.subsampled_dataset = [[tk for tk in st if not discard(tk)] for st in self.dataset]

    def compare_counts(self, token):
        return '# {}: before={}, after={}'.format(
            token,
            sum([st.count(self.token_to_idx[token]) for st in self.dataset]),
            sum([st.count(self.token_to_idx[token]) for st in self.subsampled_dataset]))

    def get_centers_and_contexts(self, max_window_size):
        centers, contexts = [], []
        for st in self.dataset:
            if len(st) < 2:
                continue
            centers += st

            for center_i in range(len(st)):
                window_size = random.randint(1, max_window_size)
                indices = list(range(max(0, center_i - max_window_size), min(len(st), center_i + 1 + window_size)))
                indices.remove(center_i)
                contexts.append([st[idx] for idx in indices])
        self.centers = centers
        self.contexts = contexts

    def get_negatives(self, sampling_weights, K):
        """负采样"""
        all_negatives, negative_cadidates, i = [], [], 0
        population = list(range(len(sampling_weights)))
        for contexts in self.contexts:
            negatives = []
            while len(negatives) < len(contexts) * K:
                if i == len(negative_cadidates):
                    i, negative_cadidates = 0, random.choices(population, sampling_weights, k=int(1e5))
                neg, i = negative_cadidates[i], i + 1
                if neg not in set(contexts):
                    negatives.append(neg)
            all_negatives.append(negatives)
        self.negatives = all_negatives

    @staticmethod
    def batchify(data):
        max_len = max(len(c) + len(n) for _, c, n in data)
        centers, contexts_negatives, masks, labels = [], [], [], []
        for center, context, negative in data:
            cur_len = len(context) + len(negative)
            centers += [center]
            contexts_negatives += [context + negative + [0] * (max_len - cur_len)]
            masks += [[1] * cur_len + [0] * (max_len - cur_len)]
            labels += [[1] * len(context) + [0] * (max_len - len(context))]
        return nd.array(centers).reshape((-1, 1)), nd.array(contexts_negatives), nd.array(masks), nd.array(labels)

    def prepare(self):
        self.load_data()
        self.make_index()
        self.get_centers_and_contexts(5)
        sampling_weights = [self.counter[w] for w in self.idx_to_token]
        self.get_negatives(sampling_weights, 5)

        batch_size = 512
        worker_num = 0 if sys.platform.startswith('win32') else 4
        dataset = gdata.ArrayDataset(self.centers, self.contexts, self.negatives)
        data_iter = gdata.DataLoader(
            dataset, batch_size, shuffle=True, batchify_fn=self.batchify, num_workers=worker_num)

        for batch in data_iter:
            for name, data in zip(['centers', 'contexts_negatives', 'masks',
                                   'labels'], batch):
                print(name, 'shape:', data.shape)
            break
        self.data_iter = data_iter
        self.batch_size = batch_size

    @staticmethod
    def skip_gram(center, contexts_and_negatives, embed_u, embed_v):
        v = embed_v(center)
        u = embed_u(contexts_and_negatives)
        pred = nd.batch_dot(v, u.swapaxes(1, 2))
        return pred
        # embed = nn.Embedding(input_dim=20, output_dim=4)
        # embed.initialize()
        # print(embed.weight)

    def train(self, lr, num_epochs):
        # import ipdb; ipdb.set_trace()
        ctx = d2l.try_gpu()
        embed_size = self.embed_size
        self.net = nn.Sequential()
        self.net.add(nn.Embedding(input_dim=len(self.idx_to_token), output_dim=embed_size),
                     nn.Embedding(input_dim=len(self.idx_to_token), output_dim=embed_size))

        self.net.initialize(ctx=ctx, force_reinit=True)
        trainer = gluon.Trainer(self.net.collect_params(), 'adam', {'learning_rate': lr})

        loss = gloss.SigmoidBinaryCrossEntropyLoss()

        for epoch in range(num_epochs):
            start, l_sum, n = time.time(), 0.0, 0
            for batch in self.data_iter:
                center, context_negative, mask, label = [data.as_in_context(ctx) for data in batch]
                with autograd.record():
                    pred = self.skip_gram(center, context_negative, self.net[0], self.net[1])
                    l = (loss(pred.reshape(label.shape), label, mask) * mask.shape[1] / mask.sum(axis=1))

                l.backward()
                trainer.step(self.batch_size)
                # l_sum += l_sum().asscalar()
                l_sum += l_sum
                n += l.size
            print('epoch %d, loss %.2f, time %.2fs'
                  % (epoch + 1, l_sum / n, time.time() - start))

    def predict(self, query_token, k, embed):
        word = embed.weight.data()
        x = word[self.token_to_idx[query_token]]

        cos = nd.dot(word, x) / (nd.sum(word * word, axis=1) * nd.sum(x * x) + 1e-9).sqrt()
        topk = nd.topk(cos, k=k + 1, ret_typ='indices').asnumpy().astype('int32')
        for i in topk[1:]:
            print('cosine sim=%.3f: %s' % (cos[i].asscalar(), (self.idx_to_token[i])))


if __name__ == '__main__':
    w = Word2Vector()
    w.prepare()
    w.train(0.005, 5)
    w.predict('chip', 3, w.net[0])
