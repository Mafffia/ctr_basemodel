from numpy import int64
import numpy as np
from numpy.lib.function_base import append
import torch
import torch.nn as nn
from mydata import get_dataloader
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def auc_arr(score_p, score_n):
    score_arr = []
    for s in score_p.numpy():
        score_arr.append([0, 1, s])
    for s in score_n.numpy():
        score_arr.append([1, 0, s])
    return score_arr

def sequence_mask(lengths, maxlen=None, dtype=torch.bool):
    lengths = torch.tensor(lengths)
    if maxlen is None:
        maxlen = lengths.max()
    row_vector = torch.arange(0, maxlen, 1)
    matrix = torch.unsqueeze(lengths, dim=-1)
    mask = row_vector < matrix

    mask.type(dtype)
    return mask

def calc_auc(raw_arr):
    """Summary
    Args:
        raw_arr (TYPE): Description
    Returns:
        TYPE: Description
    """
    # sort by pred value, from small to big
    arr = sorted(raw_arr, key=lambda d:d[2])

    auc = 0.0
    fp1, tp1, fp2, tp2 = 0.0, 0.0, 0.0, 0.0
    for record in arr:
        fp2 += record[0] # noclick
        tp2 += record[1] # click
        auc += (fp2 - fp1) * (tp2 + tp1)
        fp1, tp1 = fp2, tp2

    # if all nonclick or click, disgard
    threshold = len(arr) - 1e-3
    if tp2 > threshold or fp2 > threshold:
        return -0.5

    if tp2 * fp2 > 0.0:  # normal auc
        return (1.0 - auc / (2.0 * tp2 * fp2))
    else:
        return None

def eval(model, test_data):
    auc_sum = 0.0
    score_arr = []
    for u, i, j, hist_i, sl in test_data:
        out = model(u.long(),i.long(),hist_i.long(),sl)
        
        
        mf_auc =0
        for t in out:
            if(t[0]>t[1]):
                mf_auc += 1
            elif (t[1]<t[0]):
                mf_auc += 0
            else:
                mf_auc+=0.5

       
        auc_sum += mf_auc
        break
    test_gauc = auc_sum / len(test_data)
    
    return test_gauc

class Net(nn.Module):
    def __init__(self, user_count, item_count, cate_count, cate_list,
                       user_dim, item_dim, cate_dim,
                       dim_layers):
        super(Net, self).__init__()
        self.item_dim = item_dim
        self.cate_dim = cate_dim
        self.item_count = item_count
        self.user_emb = nn.Embedding(user_count, user_dim).to(device)
        self.item_emb = nn.Embedding(item_count, item_dim).to(device)
        #embeding of category
        self.cate_emb = nn.Embedding(cate_count, cate_dim).to(device)
        self.cate_list = cate_list
        self.fc = nn.Sequential().to(device)
        self.fc.add_module('norm',nn.BatchNorm1d(384).to(device))
        self.fc.add_module('linear',nn.Linear(384,200).to(device))
        self.fc.add_module('relu1',nn.PReLU().to(device))
        self.fc.add_module('linear2',nn.Linear(200,80).to(device))
        self.fc.add_module('relu2',nn.PReLU().to(device))
        self.fc.add_module('line3',nn.Linear(80,2).to(device))
        # self.fc.add_module('relu3',nn.PReLU().to(device))
        self.fc.add_module('softmax',nn.Softmax().to(device))
       
    def get_emb(self, user, item, history):
        user_emb = self.user_emb(user).to(device)

        item_emb = self.item_emb(item).to(device)
        #embedding of category
        
        # item_cate_emb = self.cate_emb(torch.gather(self.cate_list, item))
        item_cate_emb = self.cate_emb(self.cate_list[item].to(device)).to(device)
        #concat of item embedding and item category embedding
        item_join_emb = torch.cat([item_emb.to(device), item_cate_emb.to(device)], -1).to(device)

        # drop = nn.Dropout(p=0.5)
        # history = drop(history.float()).long().to(device)
        
        # print('history :' , history.shape)
        
        batch_pool = torch.zeros((history.shape[0],128)).to(device)
        # print('eval batch in total:',history.shape[0])
        for batch in range(history.shape[0]):
            # print(history.shape)
            his_items = []
            his_cat = []
            if(int(history[batch][0]) == 0):
                break
            for jj in range(history.shape[1]):
                # print(history[batch][jj])
                # if(not int(history[batch][jj])==0):
                    
                    his_items.append(int(history[batch][jj]))
                    his_cat.append(int(self.cate_list[int(history[batch][jj])]))
                    if(jj == history.shape[1] -1):
                        break
                    if(jj < history.shape[1] and int(history[batch][jj+1]) == 0):
                        break
            # print(his_items)
            # print(his_cat)

            batch_sum = torch.zeros((1,128)).to(device)
            concat_sum = 0
            # print('his item in batch: ', his_items)
            for iii in range(len(his_items)):
                # print('III 1')
                batch_item = self.item_emb(torch.Tensor([his_items[iii]]).long().to(device)).to(device)
                batch_cate = self.cate_emb(torch.Tensor([his_cat[iii]]).long().to(device)).to(device)
                # print(batch_item.shape)
                # print(batch_cate.shape)
                batch_sum+=torch.cat([batch_item,batch_cate],-1).to(device)
                # print(batch_sum.shape)
                concat_sum+=1
            # print('III end')
            
            
            
            # print('III 2')
            
            if(concat_sum==0):
                concat_sum=1
            #perform sum pooling    
            batch_sum/=concat_sum
            #save into ..
            batch_pool[batch,:] = batch_sum
        
        # hist_emb = self.item_emb(history).to(device)
        
        # # print("hist_emb size: " , hist_emb.shape)
        # # hist_cate_emb = self.cate_emb(torch.gather(self.cate_list, history))
        # hist_cate_emb = self.cate_emb(self.cate_list[history].to(device)).to(device)
        # hist_join_emb = torch.cat([hist_emb, hist_cate_emb], -1).to(device)
        # # hist_emb = self.item_emb(history)
        # # using dropout to reduce overfit
        # drop = nn.Dropout(p=0.5)
        # hist_join_emb = drop(hist_join_emb)
        # # hist_cate_emb = self.cate_emb(self.cate_list[history])
        # # hist_join_emb = torch.cat([hist_emb, hist_cate_emb], -1)
        
        return user_emb, item_join_emb, batch_pool
    def forward(self, user, item, history, length):
        # print(list(history[0].cpu().numpy()) ==list(history[1].cpu().numpy()))
        # print(len(history[0].cpu().numpy()))
        # print(user)
        # print(history)
        # print(item)
        user_emb, item_join_emb, hist_join_emb = self.get_emb(user.to(device), item.to(device), history.to(device))
        # print(hist_join_emb.shape)
        # print(user_emb.shape)
        # print(item_join_emb.shape)
        # print(hist_join_emb.shape)
        # hist_attn_emb = self.hist_at(item_join_emb, hist_join_emb, length)
        # hist_attn_emb = self.hist_fc(self.hist_bn(hist_attn_emb))
        #final concat layer
        # print(user_emb.shape)
        # print(item_join_emb.shape)
        # print(hist_join_emb.shape)
        # hist_mask = sequence_mask(length, max(length), dtype=torch.long)
        # hist_mask = torch.tile(torch.unsqueeze(hist_mask, -1), (1,1,self.item_dim+self.cate_dim))
        # hist_join_emb = torch.multiply(hist_join_emb, hist_mask)
        # hist_join_emb = torch.sum(hist_join_emb, -1)
        # hist_hid_emb = self.hist_fc(self.hist_bn(hist_join_emb))
        
        # user_emb = user_emb.reshape(-1,32)
        # print(user_emb.shape)
        # print("debug")
        # print(item_join_emb.shape)
        # print(hist_join_emb.shape)
        # hist_join_emb = torch.cat([hist_join_emb], dim=-1) 
        # hist_join_emb = hist_join_emb.reshape((32,-1))
        
        join_emb = torch.cat([user_emb.to(device), item_join_emb.to(device), hist_join_emb.to(device)], 1).to(device)
        

        # print(join_emb.shape)
        output = self.fc(join_emb).to(device)


        return output
