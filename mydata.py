import sys
import random
import pickle
import numpy as np
import torch
import torch.nn as nn

data_dir = './data/'
with open(data_dir+'dataset.pkl', 'rb') as f:
    train_set = pickle.load(f, encoding='latin1')
    test_set = pickle.load(f, encoding='latin1')
    cate_list = pickle.load(f, encoding='latin1')
    cate_list = torch.Tensor(cate_list).int()
    user_count, item_count, cate_count = pickle.load(f)


class DataLoader:
    def __init__(self, batch_size, data):
        self.batch_size = batch_size
        self.data = data
        self.epoch_size = len(self.data) // self.batch_size
        if self.epoch_size * self.batch_size < len(self.data):
            self.epoch_size += 1
        self.i = 0

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        if self.i == self.epoch_size:
            raise StopIteration
        ts = self.data[self.i * self.batch_size : min((self.i+1) * self.batch_size,
                                                      len(self.data))]
        self.i += 1

        u, i, y, sl = [], [], [], []
        for t in ts:
            #reviewerID
            u.append(t[0])
            #product to predict
            i.append(t[2])
            #access or not
            y.append(t[3])
            sl.append(len(t[1]))
        max_sl = max(sl)
        #access history
        hist_i = np.zeros([len(ts), max_sl], np.int64)

        k = 0
        for t in ts:
            for l in range(len(t[1])):
                hist_i[k][l] = t[1][l]
            k += 1

        return torch.Tensor(u), torch.Tensor(i), \
               torch.Tensor(y), torch.Tensor(hist_i), \
               sl


class DataLoaderTest:
    def __init__(self, batch_size, data):

        self.batch_size = batch_size
        self.data = data
        self.epoch_size = len(self.data) // self.batch_size
        if self.epoch_size * self.batch_size < len(self.data):
            self.epoch_size += 1
        self.i = 0

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):

        if self.i == self.epoch_size:
            raise StopIteration

        ts = self.data[self.i * self.batch_size : min((self.i+1) * self.batch_size,
                                                      len(self.data))]
        self.i += 1

        u, i, j, sl = [], [], [], []
        for t in ts:
            u.append(t[0])
            i.append(t[2][0])
            j.append(t[2][1])
            sl.append(len(t[1]))
        max_sl = max(sl)

        hist_i = np.zeros([len(ts), max_sl], np.int64)

        k = 0
        for t in ts:
            for l in range(len(t[1])):
                hist_i[k][l] = t[1][l]
            k += 1

        return torch.Tensor(u), torch.Tensor(i), \
               torch.Tensor(j), torch.Tensor(hist_i), \
               sl

    def __len__(self):
        return len(self.data)


def get_dataloader(train_batch_size, test_batch_size):
    return DataLoader(train_batch_size, train_set), DataLoaderTest(test_batch_size, test_set), \
           user_count, item_count, cate_count, cate_list
