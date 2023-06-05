import random
import numpy as np
import torch
from util.util import read_data
from util.get_sn import get_sn
from util.get_sn import get_sn_for_multi_input
from CPM_Nets import CPMNet_Works
import util.classfiy as classfiy
from sklearn.metrics import accuracy_score
import os
import warnings
from pictures import *
from metrics import calc_metrics_for_CPM
warnings.filterwarnings("ignore")
device = torch.device('cuda:0')

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
    torch.save(model.state_dict(),os.path.join(save_path,f'{str(args.beta)}.pt'))
    #torch.save(model, os.path.join(save_path,f'{str(args.beta)}.pt'))

       
def load_model(args):
    model = CPMNet_Works(device, view_num, input_num, trainData.num_examples, testData.num_examples, layer_size, args.lsd_dim, learning_rate,
                args.lamb, args.beta).cuda()
    # load weights
    model_state_dict=torch.load(os.path.join('pts', args.dataset, str(args.seed), f'{str(args.beta)}.pt'))
    model=model.to(device)
    model.load_state_dict(model_state_dict,strict=True)
    #model = torch.load(os.path.join('pts', args.dataset, str(args.seed), f'{str(args.beta)}.pt'))
    
    return model

# save representation h
def save_h(args, h, train):
    if train:
        if not os.path.exists('./h_train/{dataset}/{seed}'.format(dataset = args.dataset,seed = str(args.seed))):
            os.makedirs('./h_train/{dataset}/{seed}'.format(dataset = args.dataset,seed = str(args.seed)))
        for num in range(len(h)):
            torch.save(h[num], './h_train/{dataset}/{seed}/{beta}_{num}.pt'.format(\
                dataset = args.dataset,\
                seed = str(args.seed),\
                beta = str(args.beta), \
                num=num)\
                )
    else:
        if not os.path.exists('./h_test/{dataset}/{seed}'.format(dataset = args.dataset,seed = str(args.seed))):
            os.makedirs('./h_test/{dataset}/{seed}'.format(dataset = args.dataset,seed = str(args.seed)))
        for num in range(len(h)):
            torch.save(h[num], './h_test/{dataset}/{seed}/{beta}_{num}.pt'.format(\
                dataset = args.dataset,\
                seed = str(args.seed),\
                beta = str(args.beta), \
                num=num)\
                )
                
        
def load_h(args, num, train):
    h_list = []
    if train:
        for num in range(num):
            tmp = torch.load('./h_train/{dataset}/{seed}/{beta}_{num}.pt'.format(\
                dataset = args.dataset,\
                seed = str(args.seed),\
                beta = str(args.beta), \
                num=num)\
                )
            h_list.append(tmp)
    else:
        for num in range(num):
            tmp = torch.load('./h_test/{dataset}/{seed}/{beta}_{num}.pt'.format(\
                dataset = args.dataset,\
                seed = str(args.seed),\
                beta = str(args.beta), \
                num=num)\
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
    parser.add_argument('--beta', type=int, default=0,
                        help='weight of rank_loss [default: 1]')
    parser.add_argument('--dataset', type=str, default='animal',
                        help='dataset [default: handwritten0]')
    parser.add_argument('--seed', type=int, default='2022',
                        help='dataset [default: handwritten0]')
    args = parser.parse_args()

    set_seed(args.seed)

    # read data
    #trainData, testData, view_num = read_data('/home/zhangqingyang/data/'+args.dataset+'.mat', 0.8, 1)
    #trainData, testData, view_num = read_data('./data/'+args.dataset+'_c.mat', 0.8, 1)
    trainData, testData, view_num = read_data('./data/'+args.dataset+'.mat', 0.8, 1)
    #testData, _, view_num = read_data('./data/'+args.dataset+'_test.mat', 0.99, 1)
    outdim_size = [trainData.data[str(i)].shape[1] for i in range(view_num)]
    # set layer size
    layer_size = [[512, 512, outdim_size[i]] for i in range(view_num)]
    # set parameter
    epoch = [args.epochs_train, args.epochs_test]
    learning_rate = [0.01, 0.01]
    # Randomly generated missing matrix
    input_num = view_num
    Sn = get_sn_for_multi_input(view_num, [trainData.num_examples, testData.num_examples], input_num)

    #Sn_train = Sn[np.arange(trainData.num_examples)]
    #Sn_test = Sn[np.arange(testData.num_examples) + trainData.num_examples]
    Sn_train = Sn[:,np.arange(trainData.num_examples),:]
    Sn_test = Sn[:,np.arange(testData.num_examples) + trainData.num_examples,:]

    Sn = torch.LongTensor(Sn).cuda()

    Sn_train = torch.LongTensor(Sn_train).cuda()
    Sn_test = torch.LongTensor(Sn_test).cuda()

    # Model building
    model = CPMNet_Works(device, view_num, input_num, trainData.num_examples, testData.num_examples, layer_size, args.lsd_dim, learning_rate,
                    args.lamb, args.beta).cuda()
    # train
    gt1 = trainData.labels.reshape(trainData.num_examples)
    gt1 = gt1.reshape([gt1.shape[0],1])
    gt1 = torch.LongTensor(gt1)
    class_num = (torch.max(gt1) - torch.min(gt1) + 1).cpu()
    batch_size = torch.tensor(gt1.shape[0])
    label_onehot = (torch.zeros(batch_size,class_num).scatter_(1,gt1 - 1,1)) # gt1 begin from 1 so we need to set the minimum of it to 0
    H_train = model.train(trainData.data, Sn_train, label_onehot, gt1, epoch[0])
    save_model(args, model)
    save_h(args, H_train, True)
    save_h(args, model.h_test, False)

    set_seed(args.seed) #reset random seed

    H_train = load_h(args, input_num, True)
    H_test = load_h(args, input_num, False)

    model = load_model(args)
    model.h_train = H_train
    model.h_test = H_test

    # test
    gt2 = testData.labels.reshape(testData.num_examples)
    gt2 = gt2.reshape([gt2.shape[0],1])
    gt2 = torch.LongTensor(gt2)
    H_test = model.test(testData.data, Sn_test, epoch[1])

    predict_list, confidence_list, logits_list, probability_list = classfiy.ave(H_train, H_test, label_onehot.cuda(), testData.num_examples)
    for num in range(input_num):
        #print('Accuracy on the test set is {:.4f}'.format(accuracy_score(testData.labels, label_pre[num])))
        acc, aurc, eaurc, aupr, fpr, ece, nll, brier = calc_metrics_for_CPM(
                predict_list[num]-1,\
                probability_list[num],\
                logits_list[num],\
                gt2.cpu().detach().long().numpy()-1
                )

        print('Acc {:.4f}, AURC {:.4f}, NLL {:.4f}'.format(acc, aurc, nll))

    for v in range(1,input_num):
        draw_views_gap(confidence_list[v-1], confidence_list[v], args.dataset+'_'+str(input_num-v+1)+'v'+str(input_num-v)+'_'+str(args.beta))
    
    
