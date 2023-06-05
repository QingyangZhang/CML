import random
import numpy as np
import torch
from util.util import read_data_noise
from util.get_sn import get_sn
from util.get_sn import get_sn_for_multi_input, get_sn_for_multi_input_noise
from CPM_Nets import CPMNet_Works
import util.classfiy as classfiy
from sklearn.metrics import accuracy_score
import os
import warnings
from pictures import *
from metrics import calc_metrics_for_CPM
import logging
from itertools import combinations,permutations

warnings.filterwarnings("ignore")
device = torch.device('cuda:0')
matplotlib.use('Agg')


#set logger
def get_logger(name):
    logger = logging.getLogger()
    logger.setLevel(level = logging.INFO)
    handler = logging.FileHandler("{}.txt".format(name))
    handler.setLevel(logging.INFO)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.addHandler(console)
    
    return logger
    
def reset_logger(logger):
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    
def save_confidence(args, confidence_list):
    save_path=os.path.join('confidence_with_noise',args.dataset, str(args.seed),str(args.beta), str(args.noise_sen))
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
    save_path = os.path.join('pts', args.dataset, str(args.seed))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(model.state_dict(), os.path.join(save_path, f'{str(args.beta)}.pt'))
    # torch.save(model, os.path.join(save_path,f'{str(args.beta)}.pt'))


def load_model(args):
    model = CPMNet_Works(device, view_num, input_num, trainData.num_examples, testData.num_examples, layer_size,
                         args.lsd_dim, learning_rate,
                         args.lamb, args.beta).cuda()
    # load weights
    model_state_dict = torch.load(os.path.join('pts', args.dataset, str(args.seed), f'{str(args.beta)}.pt'))
    model = model.to(device)
    model.load_state_dict(model_state_dict, strict=True)
    # model = torch.load(os.path.join('pts', args.dataset, str(args.seed), f'{str(args.beta)}.pt'))

    return model


def save_h(args, h, train):
    if train:
        if not os.path.exists('./h_train_with_noise/{dataset}/{seed}'.format(dataset=args.dataset, seed=str(args.seed))):
            os.makedirs('./h_train_with_noise/{dataset}/{seed}'.format(dataset=args.dataset, seed=str(args.seed)))
        for num in range(len(h)):
            torch.save(h[num], './h_train_with_noise/{dataset}/{seed}/{beta}_{num}.pt'.format( \
                dataset=args.dataset, \
                seed=str(args.seed), \
                beta=str(args.beta), \
                num=num) \
                       )
    else:
        if not os.path.exists('./h_test_with_noise/{dataset}/{seed}'.format(dataset=args.dataset, seed=str(args.seed))):
            os.makedirs('./h_test/{dataset}/{seed}'.format(dataset=args.dataset, seed=str(args.seed)))
        for num in range(len(h)):
            torch.save(h[num], './h_test_with_noise/{dataset}/{seed}/{beta}_{num}.pt'.format( \
                dataset=args.dataset, \
                seed=str(args.seed), \
                beta=str(args.beta), \
                num=num) \
                       )


def load_h(args, num, train):
    h_list = []
    if train:
        for num in range(num):
            tmp = torch.load('./h_train/{dataset}/{seed}/{beta}_{num}.pt'.format( \
                dataset=args.dataset, \
                seed=str(args.seed), \
                beta=str(args.beta), \
                num=num) \
                )
            h_list.append(tmp)
    else:
        for num in range(num):
            tmp = torch.load('./h_test/{dataset}/{seed}/{beta}_{num}.pt'.format( \
                dataset=args.dataset, \
                seed=str(args.seed), \
                beta=str(args.beta), \
                num=num) \
                )
            h_list.append(tmp)
    return h_list


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--lsd-dim', type=int, default=512,
                        help='dimensionality of the latent space data [default: 150]')
    parser.add_argument('--epochs-train', type=int, default=30, metavar='N',
                        help='number of epochs to train [default: 30]')
    parser.add_argument('--epochs-test', type=int, default=30, metavar='N',
                        help='number of epochs to test [default: 30]')
    parser.add_argument('--lamb', type=float, default=1.0,
                        help='trade off parameter [default: 1]')
    parser.add_argument('--missing-rate', type=float, default=0,
                        help='view missing rate [default: 0]')
    parser.add_argument('--lam', type=int, default=0,
                        help='weight of rank_loss [default: 1]')
    parser.add_argument('--dataset', type=str, default='animal',
                        help='dataset [default: handwritten0]')
    parser.add_argument('--seed', type=int, default='2022',
                        help='dataset [default: handwritten0]')
    parser.add_argument('--beta', type=int, default=0,
                        help='weight of rank_loss [default: 1]')
    args = parser.parse_args()
    
    noise_sens = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    
    if args.dataset in ['animal','cub','tuandromd']:
        noise_mods = [[0],[1],[0,1]]
    elif args.dataset in ['yaleB_mtv', 'PIE_face_10']:
        noise_mods = [[0],[1],[2], [0,1], [1,2], [0,2], [0,1,2]]
    elif args.dataset in ['handwritten0']:
        noise_mods = [[0],[1],[2],[3],[4],[5]]
        noise_mods.extend(list(list(i) for i in permutations([0,1,2,3,4,5],2)))
        noise_mods.extend(list(list(i) for i in permutations([0,1,2,3,4,5],3)))
    
    for noise_mod in noise_mods:
        
        noise_mod_str_list = [str(i) for i in noise_mod]
        logger = get_logger(args.dataset+"_noise_on_"+''.join(noise_mod_str_list))
        
        for noise_sen in noise_sens:
            set_seed(args.seed)
            args.noise_sen = noise_sen
            noise_flag = [0, 1]
            # read data
            # trainData, testData, view_num = read_data('/home/zhangqingyang/data/'+args.dataset+'.mat', 0.8, 1)
            trainData, testData, view_num = read_data_noise('./data/' + args.dataset + '.mat', 0.8, 1, noise=noise_mod, noise_sen=noise_sen)
            outdim_size = [trainData.data[str(i)].shape[1] for i in range(view_num)]
            # set layer size
            layer_size = [[512, 512, outdim_size[i]] for i in range(view_num)]
            # set parameter
            epoch = [args.epochs_train, args.epochs_test]
            learning_rate = [0.01, 0.01]
            # Randomly generated missing matrix
            input_num = 1
        
            Sn = get_sn_for_multi_input_noise(view_num, [trainData.num_examples, testData.num_examples], input_num, noise_flag)
            Sn_train = Sn[:, np.arange(trainData.num_examples), :]
            Sn_test = Sn[:, np.arange(testData.num_examples) + trainData.num_examples, :]

            Sn = torch.LongTensor(Sn).cuda()
            Sn_train = torch.LongTensor(Sn_train).cuda()
            Sn_test = torch.LongTensor(Sn_test).cuda()

            # train
            gt1 = trainData.labels.reshape(trainData.num_examples)
            gt1 = gt1.reshape([gt1.shape[0], 1])
            gt1 = torch.LongTensor(gt1)
            class_num = (torch.max(gt1) - torch.min(gt1) + 1).cpu()
            batch_size = torch.tensor(gt1.shape[0])
            # gt1 begin from 1 so we need to set the minimum of it to 0
            label_onehot = (torch.zeros(batch_size, class_num).scatter_(1, gt1 - 1,1)) 
                                                                          
            H_train = load_h(args, input_num, True)
            H_test = load_h(args, input_num, False)
        
            model = load_model(args)
            model.h_train = H_train
            model.h_test = H_test
        
            # test
            gt2 = testData.labels.reshape(testData.num_examples)
            gt2 = gt2.reshape([gt2.shape[0], 1])
            gt2 = torch.LongTensor(gt2)
            H_test = model.test(testData.data, Sn_test, epoch[1])
        
            predict_list, confidence_list, logits_list, probability_list = classfiy.ave(H_train, H_test, label_onehot.cuda(),
                                                                                        testData.num_examples)
            # print('Accuracy on the test set is {:.4f}'.format(accuracy_score(testData.labels, label_pre[num])))
            acc, aurc, eaurc, aupr, fpr, ece, nll, brier = calc_metrics_for_CPM(
                predict_list[0] - 1, \
                probability_list[0], \
                logits_list[0], \
                gt2.cpu().detach().long().numpy() - 1
                )
            
            logger.info('{},{},{},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}'\
                .format(args.seed, args.beta, args.noise_sen, acc, nll, aurc, eaurc, aupr, fpr))
                
        reset_logger(logger)