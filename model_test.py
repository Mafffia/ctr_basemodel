import torch
from mydata import get_dataloader
import matplotlib.pyplot as plt
from mymodel import Net
import torch.nn as nn
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_batch_size = 32
test_batch_size = 8
user_dim = 128
item_dim = 64
cate_dim = 64


def same(a, b, c, d):
    return a == b and a == c and b == c and b == d and c == d


def eval(net, test_data):
    losssum = 0.0
    accsum = 0.0
    step = 0
    with torch.no_grad():
        score_arr = []

        for u, i, j, hist_i, sl in test_data:
            step += 1
            if (not same(u.size()[0], hist_i.size()[0], i.size()[0], j.size()[0])):
                print("found error")
                continue
            if (u.shape[0] < test_batch_size):
                continue

            # his_hot = torch.zeros(test_batch_size,item_count)
            # for ii in range(hist_i.size()[0]):
            #     for jj in range(len(hist_i[ii])):
            #         # print(int(hist_i[ii][j]),end='')

            #         his_hot[ii][int(hist_i[ii][jj])] = 1

            # print("hot:")
            # print(his_hot.shape)

            out_positive = net(u.long().to(device), i.long().to(device), hist_i.long().to(device), sl).to(device)
            out_negative = net(u.long().to(device), j.long().to(device), hist_i.long().to(device), sl).to(device)
            criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
            test_y_p = torch.ones((test_batch_size)).to(device)
            test_y_n = torch.zeros((test_batch_size)).to(device)

            test_y = torch.cat([test_y_p, test_y_n], 0).long().to(device)
            # print(test_y.shape)
            # test_y=test_y.squeeze(1).to(device)
            out_y = torch.cat([out_positive, out_negative], 0).to(device)

            _, pred = torch.max(out_y, axis=1)

            acc = torch.sum(test_y == pred).to(device) / out_y.shape[0]


            # print(out_y.shape)
            loss = criterion(out_y.to(device), test_y.to(device))

            accsum += acc
            losssum += loss
            torch.cuda.empty_cache()

        losssum = losssum / step
        accsum = accsum / step
    return losssum, accsum


if (__name__ == '__main__'):
    net = torch.load('_model_epoch_5.pkl', map_location=device)
    train_data, test_data, \
    user_count, item_count, cate_count, \
    cate_list = get_dataloader(train_batch_size, test_batch_size)
    # print(eval(model,test_data))
    total = {}
    tru = {}
    fal = {}
    for u, i, j, hist_i, sl in test_data:

        if not same(u.size()[0], hist_i.size()[0], i.size()[0], j.size()[0]):
            print("found error")
            continue
        if u.shape[0] < test_batch_size:
            continue

        # count the length of history
        lengths = torch.count_nonzero(hist_i, dim=1)
        out_positive = net(u.long().to(device), i.long().to(device), hist_i.long().to(device), sl).to(device)
        out_negative = net(u.long().to(device), j.long().to(device), hist_i.long().to(device), sl).to(device)
        test_y_p = torch.ones(test_batch_size).to(device)
        test_y_n = torch.zeros(test_batch_size).to(device)
        _, pred_pos = torch.max(out_positive, axis=1)
        _, pred_neg = torch.max(out_positive, axis=1)
        test_y = torch.cat([test_y_p, test_y_n], 0).long().to(device)
        # print(test_y.shape)
        # test_y=test_y.squeeze(1).to(device)
        out_y = torch.cat([out_positive, out_negative], 0).to(device)
        acc_true =  lengths[pred_pos==test_y_p]
        acc_false = lengths[pred_neg==test_y_n]
        for i in lengths:
            le = int(i)
            if(le not in total.keys()):
                total[le] = 0
            if(le not in tru.keys()):
                tru[le] = 0
            if(le not in fal.keys()):
                fal[le] = 0
            total[le] +=1
        for i in acc_true:
            le = int(i)
            tru[le] +=1
        for i in acc_false:
            le = int(i)
            fal[le]+=1

    for key in tru.keys():
        tru[key]/=total[key]
    for key in fal.keys():
        fal[key]/=total[key]

    # print(tru)
    # print(fal)
    tru_pairs = sorted(tru.items(),key = lambda item:int(item[0]))
    fal_pairs = sorted(fal.items(),key = lambda item:int(item[0]))
    tru_x = []
    tru_y = []
    fal_x = []
    fal_y = []

    for pair in tru_pairs:
        if(total[pair[0]] < 20):
            continue
        tru_x.append(int(pair[0]))
        tru_y.append(pair[1])

    for pair in fal_pairs:
        if (total[pair[0]] < 20):
            continue
        fal_x.append(int(pair[0]))
        fal_y.append(pair[1])

    plt.plot(tru_x[:150],tru_y[:150])
    plt.show()
    plt.plot(fal_x[:150],fal_y[:150])
    plt.show()
    print('pos mean',np.mean(tru_y[:150]))
    print('neg mean',np.mean(fal_y[:150]))