import os
from numpy import sign, sin, single
import torch
from mydata import get_dataloader
import torch.optim as optim
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import matplotlib.pyplot as plt
from mymodel import Net
import torch.nn as nn
import os
import pandas as pd
os.environ["CUDA_VISIBLE_DEVICES"] = "0" #GPU编号
train_batch_size = 32
test_batch_size = 8
user_dim = 128
item_dim = 64
cate_dim = 64
def same(a,b,c,d):
    return a==b and a==c and b==c and b==d and c==d
# Data Load
# parser.add_argument('--user_count', default=192403, help='number of users', type=int)
#     parser.add_argument('--item_count', default=63001, help='number of items', type=int)
#     parser.add_argument('--cate_count', default=801, help='number of categories', type=int)

#     parser.add_argument('--user_dim', default=128, help='dimension of user', type=int)
#     parser.add_argument('--item_dim', default=64, help='dimension of item', type=int)
#     parser.add_argument('--cate_dim', default=64, help='dimension of category', type=int)

#     parser.add_argument('--dim_layers', default=[80,40,2], type=int)
def eval(model, test_data):
    losssum = 0.0
    accsum = 0.0
    step = 0
    with torch.no_grad():
        score_arr = []



        for u, i, j, hist_i, sl in test_data:
            step+=1
            if(not same(u.size()[0],hist_i.size()[0],i.size()[0],j.size()[0])):
                print("found error")
                continue
            if(u.shape[0]<test_batch_size):
                continue
            
            
            # his_hot = torch.zeros(test_batch_size,item_count)
            # for ii in range(hist_i.size()[0]):
            #     for jj in range(len(hist_i[ii])):
            #         # print(int(hist_i[ii][j]),end='')

            #         his_hot[ii][int(hist_i[ii][jj])] = 1
                
            # print("hot:")
            # print(his_hot.shape)
            
            out_positive = net(u.long().to(device),i.long().to(device),hist_i.long().to(device),sl).to(device)
            out_negative = net(u.long().to(device),j.long().to(device),hist_i.long().to(device),sl).to(device)
            criterion = nn.CrossEntropyLoss()  #交叉熵损失函数
            test_y_p = torch.ones((test_batch_size)).to(device)
            test_y_n = torch.zeros((test_batch_size)).to(device)

            test_y = torch.cat([test_y_p,test_y_n],0).long().to(device)
            # print(test_y.shape)
            # test_y=test_y.squeeze(1).to(device)
            out_y = torch.cat([out_positive,out_negative],0).to(device)

            acc = 0.0
            total = 0.0
            for zz in range(out_y.shape[0]):
                total+=1
                flag = 0
                if(float(out_y[zz][0]) < float(out_y[zz][1])):
                    flag = 1
                if(int(test_y[zz]) == flag):
                    acc +=1
            
            acc/=total
            # print(out_y.shape)
            loss = criterion(out_y.to(device),test_y.to(device))
            
            accsum += acc
            losssum+=loss
            torch.cuda.empty_cache()
            
           
        losssum = losssum/step
        accsum = accsum/step
    return losssum,accsum

if(__name__=="__main__"):
    #gpu
    testlosses = []
    trainlosses = []
    accs = []
    test_accs = []
    # Data Load
    train_data, test_data, \
    user_count, item_count, cate_count, \
    cate_list = get_dataloader(train_batch_size, test_batch_size)
    criterion = nn.CrossEntropyLoss()  #交叉熵损失函数
    net = Net(user_count, item_count, cate_count, cate_list,user_dim, item_dim, cate_dim,
                       [80,40,2]).to(device)
    print(net)
    optimizer = optim.SGD(net.parameters(), lr=0.05, momentum=0.9)
    # optimizer = optim.Adam(net.parameters(), lr=0.1)
    for epoch in range(5):
        running_loss = 0.0
        running_acc = 0.0
        for step, (u, i, y, hist_i, sl) in enumerate(train_data, start=1):
            
            if(u.shape[0]<train_batch_size):
                continue
            his_hot = torch.zeros(train_batch_size,item_count)
            for ii in range(hist_i.size()[0]):
                for j in range(len(hist_i[ii])):
                    # print(int(hist_i[ii][j]),end='')

                    his_hot[ii][int(hist_i[ii][j])] = 1
               
            # print("hot:")
            # print(his_hot.shape)
            optimizer.zero_grad()
            out = net(u.long().to(device),i.long().to(device),hist_i.long().to(device),sl)
            # print(out)
            # print(out.shape)
            train_y = torch.randint(1,10,(train_batch_size,1))
            for i in range(train_batch_size):
                if(int(y[i])==1):
                    train_y[i][0] = 1
                else:
                    train_y[i][0] = 0
                # if(float(out[i][0] > 0.5)):
                #     out[i] = 1
                # else:
                #     out[i] = 0
            
            train_y = train_y.squeeze(1) .to(device)
            # # out = out.squeeze(1) 
            
            # print(out)
            # print(train_y)
            # print(out.shape)
            # print(train_y.shape)

            loss = criterion(out.to(device),train_y.to(device))
            acc = 0.0
            total = 0.0
            for zz in range(out.shape[0]):
                total+=1
                flag = 0
                if(float(out[zz][0]) < float(out[zz][1])):
                    flag = 1
                if(int(train_y[zz]) == flag):
                    acc +=1
            acc/=total
            accs.append(acc)
            # print(loss)
            trainlosses.append(float(loss))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # print statistics
            running_loss += loss.item()
            running_acc+=acc
            
            # if True:    # print every 2000 mini-batches
            #     print('[%d, %5d] loss: %.3f train_acc: %5f' %
            #     (epoch + 1, step + 1, running_loss / 1, running_acc/1.0))
            # running_loss = 0.0
            # running_acc = 0.0
                
            if step%3000 == 2999:    # print every 3000 mini-batches
                print('[%d, %5d] loss: %.3f train_acc: %5f' %
                    (epoch + 1, step + 1, running_loss / 3000, running_acc/3000))
                running_loss = 0.0
                running_acc = 0.0
                # test_loss = eval(net,test_data)
                # testlosses.append(test_loss)
                # print("eval_loss: %.3f " % test_loss)



            # if step%10000 == 9999:
                
            if step%10000 == 9999: 
               
                test_loss,test_acc = eval(net,test_data)
                testlosses.append(float(test_loss))
                test_accs.append(float(test_acc))
                print("eval_loss: %.3f eval_acc: %.3f" % (test_loss,test_acc))
            
        test_loss,test_acc = eval(net,test_data)
        testlosses.append(float(test_loss))
        test_accs.append(float(test_acc))
        print("eval_loss: %.3f  eval_acc: %.3f" % (test_loss,test_acc))
    plt.plot(trainlosses)
    plt.savefig('train_loss.png')
    plt.show()
    plt.plot(testlosses)
    plt.savefig('test_loss.png')
    plt.show()
    plt.plot(accs)
    plt.savefig('train_acc.png')
    plt.show()
    plt.plot(test_accs)
    plt.savefig('test_acc.png')
    plt.show()
    dflist = []            
    for it in range(len(trainlosses)):
        dflist.append([trainlosses[it],accs[it]])
    df = pd.DataFrame(dflist,columns=['train loss','train acc'])
    df.to_csv('train.csv')

    dflist = []            
    for it in range(len(test_accs)):
        dflist.append([testlosses[it],test_accs[it]])
    df = pd.DataFrame(dflist,columns=['test loss','test acc'])
    df.to_csv('test.csv')

        # print(out)