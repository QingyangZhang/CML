import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.distributions as td
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
import os
from dataset import Multi_view_data, Multi_view_data_with_noise
from pictures import *
from model import MIWAE 
import argparse
from numpy.random import randint
import random
from metrics import calc_metrics_for_CPM
import logging
from itertools import combinations,permutations


def reset_logger(logger):
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

def get_mean(loader):
    
    mean_list = []
    
    for batch_idx, (data, target) in enumerate(loader):
        mean = torch.mean(data, axis = 0).unsqueeze(0)
        #print(mean.shape)
        
        var = torch.std(data, axis = 0)
        mean_list.append(mean)
        
    total_mean = torch.cat(mean_list, 0)
    #print(total_mean.shape)
    total_mean_mean = torch.mean(total_mean, axis = 0)
    
    return total_mean_mean
        
#set logger
def get_logger(name):
    logger = logging.getLogger(__name__)
    logger.setLevel(level = logging.INFO)
    handler = logging.FileHandler("{}.txt".format(name))
    handler.setLevel(logging.INFO)
    #formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    #handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.addHandler(console)
    
    return logger


def save_confidence(args, confidence_list):
    save_path=os.path.join('confidence',args.dataset, str(args.seed),str(args.beta))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for i, confidence in enumerate(confidence_list):
        txt_path = os.path.join(save_path, str(i)+'.txt')
        fileObject = open(txt_path, 'w+') 
        for c in confidence:  
            fileObject.write(str(c))
            fileObject.write('\n')
        fileObject.close()

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
    #model_state_dict=torch.load(os.path.join('pts', args.dataset, str(args.seed), f'MIWAE_{str(args.beta)}.pt'))
    model_state_dict=torch.load(os.path.join('pts', args.dataset, str(2022), f'MIWAE_{str(args.beta)}.pt'))
    model.load_state_dict(model_state_dict,strict=True)
    #model = torch.load(os.path.join('pts', args.dataset, str(args.seed), f'{str(args.beta)}.pt'))
    
    return model
    
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
    noise_sens = [0, 0.5, 1.0, 1.5, 2, 2.5]
    #if args.dataset in ['animal']:
    #    noise_sens = [0, 1.0, 2.0, 3.0, 4.0, 5.0]
    #noise_sens = [0]
    
    if args.dataset in ['animal','cub']:
        noise_mods = [[0],[1],[0,1]]
    elif args.dataset in ['yaleB_mtv', 'PIE_face_10']:
        noise_mods = [[0],[1],[2], [0,1], [1,2], [0,2], [0,1,2]]
    elif args.dataset in ['handwritten0']:
        noise_mods = [[0],[1],[2],[3],[4],[5]]
        noise_mods.extend(list(list(i) for i in permutations([0,1,2,3,4,5],2)))
        noise_mods.extend(list(list(i) for i in permutations([0,1,2,3,4,5],3)))
        
    #noise_mods = [[1]]
    
    args.data_path = 'data/' + args.dataset
    
    #loader_kwargs = {'num_workers': 0, 'pin_memory': False}
    loader_kwargs = {'num_workers': 4, 'pin_memory': True}
    
    for noise_mod in noise_mods:
        
        noise_mod_str_list = [str(i) for i in noise_mod]
        logger = get_logger(args.dataset+"_noise_on_"+''.join(noise_mod_str_list))
        
        for noise_sen in noise_sens:
            # get dataloader
            args.noise_sen = noise_sen
            set_seed(args.seed)
            train_dataset = Multi_view_data_with_noise(args.data_path, True, noise_mod, noise_sen)
            test_dataset = Multi_view_data_with_noise(args.data_path, False, noise_mod, noise_sen)
            #test_dataset = Multi_view_data(args.data_path, False)
            
            train_loader = torch.utils.data.DataLoader(
                train_dataset, 
                batch_size=args.batch_size,
                shuffle=True,
                **loader_kwargs
                )
        
            test_loader = torch.utils.data.DataLoader(
                test_dataset, 
                #batch_size=args.batch_size,
                batch_size = len(test_dataset),
                shuffle=False,
                **loader_kwargs
                )
    
            data_mean = get_mean(train_loader)
       
            view_num = train_dataset.view_number
            test_input_num = 1
            feature_per_view = train_dataset.feature_per_view
    
            confidence_list = []
    
            # Hyperparameters
            args.h = 128 # number of hidden units in (same for all MLPs)
            args.d = 64 # dimension of the latent space
            args.K = 3 # number of IS during training
            args.p = sum(feature_per_view)
            args.classes = train_dataset.classes
      
            set_seed(args.seed) #reset random seed
    
            # Model building
            model = load_model(args)
            model.cuda()
    
            imputed_mse_list = [0 for i in range(test_input_num)]
            acc_list = [0 for i in range(test_input_num)]
    
            # predict, softmax, logit, label
            total_predict_list = {i:[] for i in range(test_input_num)}
            total_softmax_list = {i:[] for i in range(test_input_num)}
            total_logits_list = {i:[] for i in range(test_input_num)}
            total_label_list = {i:[] for i in range(test_input_num)}
    
            
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
                
                    total_label_list[num].extend(target.cpu().data.numpy())
                    total_logits_list[num].extend(logits.cpu().data.numpy())
                    total_softmax_list[num].extend(probability.cpu().data.numpy())
                    total_predict_list[num].extend(predict.cpu().data.numpy())
    
                    #save_confidence(args, confidence_list)
    
            acc_list = [acc/len(test_loader) for acc in acc_list]
            imputed_mse_list = [imputed_mse/len(test_loader) for imputed_mse in imputed_mse_list]
            
            for num in range(test_input_num):
                print('Input %g, Imputation MSE  %g, Classify Acc %g' %(num, imputed_mse_list[num], acc_list[num]))
            print('-----')
    
            for num in range(test_input_num):
                total_label_list[num] = np.array(total_label_list[num])
                total_predict_list[num] = np.array(total_predict_list[num])
                total_softmax_list[num] = np.array(total_softmax_list[num])
                total_logits_list[num] = np.array(total_logits_list[num])

        
            for num in range(test_input_num):
                acc, aurc, eaurc, aupr, fpr, ece, nll, brier = calc_metrics_for_CPM(
                total_predict_list[num],\
                total_softmax_list[num],\
                total_logits_list[num],\
                total_label_list[num]
                )
        
            logger.info('{},{},{},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}'\
                .format(args.seed, args.beta, args.noise_sen, acc, nll, aurc, eaurc, aupr, fpr))
    
        reset_logger(logger)
