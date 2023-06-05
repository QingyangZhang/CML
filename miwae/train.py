import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.distributions as td
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
import os
from dataset import Multi_view_data
from pictures import *
from model import MIWAE 
import argparse
from numpy.random import randint
import random

# calculate mean and std for imputation
def get_mean(loader):
    mean_list = []
    for batch_idx, (data, target) in enumerate(loader):
        mean = torch.mean(data, axis = 0).unsqueeze(0)
        print(mean.shape)
        var = torch.std(data, axis = 0)
        mean_list.append(mean)
        
    total_mean = torch.cat(mean_list, 0)
    print(total_mean.shape)
    total_mean_mean = torch.mean(total_mean, axis = 0)
    
    return total_mean_mean
        


def set_seed(seed):
    torch.cuda.manual_seed(seed) 
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def save_model(args, model):
    save_path=os.path.join('pts',args.dataset, str(args.seed))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(model.state_dict(),os.path.join(save_path,f'MIWAE_{str(args.beta)}.pt'))
    #torch.save(model, os.path.join(save_path,f'{str(args.beta)}.pt'))

       
def load_model(args):
    model = MIWAE(args.h, args.d, args.K, args.p, args.classes).cuda()
    # load weights
    model_state_dict=torch.load(os.path.join('pts', args.dataset, str(args.seed), f'{str(args.beta)}.pt'))
    model.load_state_dict(model_state_dict,strict=True)
    #model = torch.load(os.path.join('pts', args.dataset, str(args.seed), f'{str(args.beta)}.pt'))
    
    return model


# Training
def weights_init(layer):
  if type(layer) == nn.Linear: torch.nn.init.orthogonal_(layer.weight)

# mask for multi inputs，mask_list[i] is 0-1 matrix with dim n*l，
# where n is data size，l is num of attributions of all modality
def get_multi_mask(input_num, n, p, feature_per_view):
    mask_list = []
    view_num = len(feature_per_view)
    miss_order = []
    for i in range(n):
        tmp = list(range(view_num))
        random.shuffle(tmp)
        miss_order.append(tmp)
        
    mask = np.ones((n,p))
    mask_list.append(torch.from_numpy(mask).bool().cuda())
    for num in range(input_num-1):
        for i in range(n):
            miss_which_view = miss_order[i][num]
            begin = sum(feature_per_view[0:miss_which_view])
            end = sum(feature_per_view[0:miss_which_view+1])
            mask[i,begin:end] = 0
        mask_list.append(torch.from_numpy(mask).bool().cuda())
               
    return mask_list

# sample mask    
def sample_multi_mask(n, p, feature_per_view, miss_rate, train):
    view_num = len(feature_per_view)
    mask_list = []
    for m in miss_rate:
        mask_list.append(np.random.choice([0,1], n*view_num, True, [1-m, m]).reshape(n,view_num))
    
    
    if train:
        for i in range(1, len(miss_rate)):
            mask_list[i] = mask_list[i]&mask_list[i-1]
        
    for i in range(0, len(miss_rate)):
        miss_all = mask_list[i].sum(axis=1)==0
        for index in range(n):
            if miss_all[index]:
                mask_list[i][index,random.randint(0,view_num-1)]=1
    
    multi_mask_list = []
    for mask in mask_list:
        tmp = np.ones((n, p))
        for i in range(n):
            for v in range(view_num):
                if mask[i,v] == 0:
                    begin = sum(feature_per_view[0:v])
                    end = sum(feature_per_view[0:v+1])
                    tmp[i,begin:end] = 0
        multi_mask_list.append(torch.from_numpy(tmp).bool().cuda())
    
    return multi_mask_list
  
            
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=200, metavar='N',
                        help='input batch size for training [default: 100]')
    parser.add_argument('--dataset', type=str, default='handwritten0', metavar='N',
                        help='input batch size for training [default: 100]')
    parser.add_argument('--epochs', type=int, default=60, metavar='N',
                        help='number of epochs to train [default: 500]')
    parser.add_argument('--lr', type=float, default=0.003, metavar='LR',
                        help='learning rate [default: 1e-3]')
    parser.add_argument('--beta', type=int, default=0, metavar='N',
                        help='trade-off parameter for rank_loss [default: 0/100]')
    parser.add_argument('--alpha', type=int, default=1, metavar='N',
                        help='trade-off parameter for cls_loss [default: 1]')
    parser.add_argument('--seed', type=int, default=2022, metavar='N',
                        help='random seed [default: 2022]')
    parser.add_argument('--epsilon', type=int, default=0.0, metavar='N',
                        help='Tolerance Margin [default: 0.0]')

    
    args = parser.parse_args()
    
    args.data_path = 'data/' + args.dataset
    
    set_seed(args.seed)

    # get dataloader
    train_dataset = Multi_view_data(args.data_path, train=True)
    test_dataset = Multi_view_data(args.data_path, train=False)
    
    #loader_kwargs = {'num_workers': 0, 'pin_memory': False}
    loader_kwargs = {'num_workers': 4, 'pin_memory': True}
        
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        **loader_kwargs
        )
        
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=args.batch_size,
        shuffle=False,
        **loader_kwargs
        )
    
    data_mean = get_mean(train_loader)
       
    view_num = train_dataset.view_number
    train_input_num = view_num
    feature_per_view = train_dataset.feature_per_view
    
    confidence_list = []
    
    # Hyperparameters
    args.h = 128 # number of hidden units in (same for all MLPs)
    args.d = 64 # dimension of the latent space
    args.K = 3 # number of IS during training
    args.p = sum(feature_per_view)
    args.classes = train_dataset.classes
    
    # Model building
    model = MIWAE(args.h, args.d, args.K, args.p, args.classes)
    model.cuda()
    model.apply(weights_init)
    
    if args.dataset == 'yaleB_mtv':
        optimizer = optim.Adam(list(model.parameters()),lr=1e-3)
    else:
        optimizer = optim.Adam(list(model.parameters()),lr=1e-2)
    
    best_acc = 0.0
          
    for ep in range(1,args.epochs):
        # two training stage
        if ep < 30:
            step = [5, 0, 0] # stage 1, train model with reconstruction loss only
        else:
            step = [0, 0, 5] # stage 2，train the whole model with rec_loss/cls_loss/MCCA
            
        # learning rate decay 
        if ep > 20:
            if args.dataset == 'yaleB_mtv':
                optimizer = optim.Adam(list(model.parameters()),lr=1e-4)
            else:
                optimizer = optim.Adam(list(model.parameters()),lr=1e-3)
        '''
            # param freeze
            for name, param in model.named_parameters():
                if "encoder" in name or "decoder" in name:
                    param.requires_grad = False
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-2)
        '''
            
        train_total_cls_loss = 0
        train_total_rec_loss = 0
        train_total_rank_loss = 0
        
        # training
        for batch_idx, (data, target) in enumerate(train_loader):
            #mask_list = sample_multi_mask_pair(data.shape[0], p, feature_per_view, [0.75,0.75,0.75,0.75,0.75])
            #mask_list = sample_multi_mask(data.shape[0], p, feature_per_view, [0.5,0.5], True)
            mask_list = get_multi_mask(train_input_num, data.shape[0], args.p, feature_per_view)
            data_partial_list = []
            data_imputed_list = []
            data, target = data.cuda(), target.long().cuda()
            
            # impute data with means
            for num in range(train_input_num):
                data_partial = data.clone()
                data_partial[~mask_list[num]] = data_mean.expand(data.shape[0],-1).cuda()[~mask_list[num]]
                data_partial = data_partial.cuda()
                data_partial_list.append(data_partial)
            
            # stage 1
            for i in range(step[0]):
                rec_loss = 0
                for num in range(train_input_num):
                    rec_loss += model.miwae_loss(\
                        data_partial_list[num], \
                        mask_list[num]\
                        )

                rec_loss = rec_loss/train_input_num
                optimizer.zero_grad()
                rec_loss.backward()
                optimizer.step()
                train_total_rec_loss += rec_loss
            
            # no use
            '''
            for i in range(step[1]):
                cls_loss = 0
                rec_loss = 0
                for num in range(0,train_input_num):
                    rec_loss += model.miwae_loss(\
                        data_partial_list[num], \
                        mask_list[num]\
                        )
                    data_imputed, logits = model(data_partial_list[num], mask_list[num])
                    cls_loss = nn.CrossEntropyLoss()(logits, target)
                    
                cls_loss = cls_loss/train_input_num
                rec_loss = rec_loss/train_input_num
                loss = rec_loss + cls_loss*args.alpha
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_total_rec_loss += rec_loss
                train_total_cls_loss += cls_loss
            '''
            # stage 2
            for i in range(step[2]):
                rec_loss = 0
                cls_loss = 0
                rank_loss = 0
                confidence_list = []
                predict_list = []
                for num in range(0,train_input_num):
                    rec_loss += model.miwae_loss(\
                        data_partial_list[num], \
                        mask_list[num]\
                        )
                    data_imputed, logits = model(data_partial_list[num], mask_list[num])
                    cls_loss += nn.CrossEntropyLoss()(logits, target)
                    probability = F.softmax(logits)
                    _, predict = torch.max(probability, 1)
                    confidence = torch.max(probability, axis=1)[0]
                    confidence_list.append(confidence)
                    predict_list.append(predict)
                cls_loss /= train_input_num
                
                for num in range(train_input_num-1):
                    #sign = (~(predict_list[num+1]==target)&(predict_list[num]!=target)).long() #trick 1
                    sign = (~(predict_list[num]!=target)).long() #trick 1
                    rank_loss += torch.nn.ReLU()(torch.sub(confidence_list[num+1],confidence_list[num])*sign-args.epsilon).sum() 
                    # trick 2
                    sign1 = ((predict_list[num+1]!=target)&(predict_list[num]!=target))
                    conf1 = confidence_list[num+1].clone()
                    conf2 = confidence_list[num].clone().detach()
                    rank_loss += torch.nn.ReLU()(torch.sub(conf1,conf2)*sign1-args.epsilon).sum()
                    
                loss = rec_loss + cls_loss*args.alpha + rank_loss*args.beta/100
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_total_cls_loss += cls_loss
                train_total_rank_loss +=rank_loss
                
        
        test_input_num = train_input_num
        imputed_mse_list = [0 for i in range(test_input_num)]
        acc_list = [0 for i in range(test_input_num)]
        
        set_seed(args.seed) #reset random seed
        
        # testing
        for batch_idx, (data, target) in enumerate(test_loader):
            #mask_list = sample_multi_mask(data.shape[0], p, feature_per_view, [1, 0.8, 0.6, 0.4, 0.2], False)
            mask_list = get_multi_mask(test_input_num, data.shape[0], args.p, feature_per_view)
            data_partial_list = []
            data, target = data.cuda(), target.long().cuda()
            
            confidence_list = []
            for num in range(test_input_num):
                data_partial = data.detach().clone()
                data_partial[~mask_list[num]] = data_mean.expand(data.shape[0],-1).cuda()[~mask_list[num]]
                data_partial = data_partial.cuda()
                data_partial_list.append(data_partial)
                
                with torch.no_grad():
                    data_imputed, logits = model(data_partial_list[num], mask_list[num])
                    imputed_mse_list[num] = imputed_mse_list[num] + model.miwae_mse(data_imputed, data)
                    probability = F.softmax(logits)
                    _, predict = torch.max(probability, 1)
                    confidence = torch.max(probability, axis=1)[0]
                    confidence = confidence.cpu().data.numpy()
                    confidence_list.append(confidence)
                    acc_list[num] = acc_list[num] + torch.sum(predict==target.squeeze())/predict.shape[0]

        print('Epoch %g' %(ep))
        #print(train_total_rec_loss.cpu().data.numpy())
        #print(train_total_cls_loss)
        #print(train_total_rank_loss.cpu().data.numpy())
        acc_list = [acc/len(test_loader) for acc in acc_list]
        imputed_mse_list = [imputed_mse/len(test_loader) for imputed_mse in imputed_mse_list]
        for num in range(test_input_num):
            print('Input %g, Imputation MSE  %g, Classify Acc %g' %(num, imputed_mse_list[num], acc_list[num]))
        print('-----')
        #print(best_acc)
        #print(sum(acc_list))
        if sum(acc_list) > best_acc:
            best_acc = sum(acc_list)
            best_acc_list = acc_list
            print("save model")
            save_model(args,model)
            
    print(best_acc_list)
    
    # visualize
    #for v in range(1,test_input_num):
    #    draw_views_gap(confidence_list[v-1], confidence_list[v], str(test_input_num-v+1)+'v'+str(test_input_num-v)+'_'+str(args.beta))
    #    draw_gap_hist(confidence_list[v-1], confidence_list[v], str(test_input_num-v+1)+'v'+str(test_input_num-v)+'_'+str(args.beta)+'_Hist')
            
            