import util.classfiy as classfiy
import torch
import numpy as np
from numpy.random import shuffle
from util.util import xavier_init
from util.CPM import CPMNets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class CPMNet_Works(nn.Module): # Main parts of the test code
    """build model
    """
    def __init__(self, device, view_num, input_num, trainLen, testLen, layer_size, lsd_dim=128, learning_rate=[0.001, 0.001], lamb=1, belta=1, seed=2022):
        """
        :param learning_rate:learning rate of network and h
        :param view_num:view number
        :param layer_size:node of each net
        :param lsd_dim:latent space dimensionality
        :param trainLen:training dataset samples
        :param testLen:testing dataset samples
        """
        super(CPMNet_Works, self).__init__()
        # initialize parameter
        self.view_num = view_num
        self.input_num = input_num
        self.layer_size = layer_size
        self.lsd_dim = lsd_dim
        self.trainLen = trainLen
        self.testLen = testLen
        self.lamb = lamb
        self.belta = belta
        self.learning_rate = learning_rate
        self.device = device
        self.seed = seed
        # initialize latent space data
        self.h_train = self.H_init('train')
        self.h_test = self.H_init('test')
        # initialize nets for different views
        self.net, self.train_net_op = self.bulid_model()
        
    def H_init(self, a):
        #h = nn.ModuleList()
        h = []
        if a == 'train':
            for num in range(self.input_num):
                #tmp = nn.Module()
                #tmp.value = nn.Parameter(xavier_init(self.trainLen, self.lsd_dim))
                #h.append(tmp)
                h.append(Variable(xavier_init(self.trainLen, self.lsd_dim), requires_grad = True))
                #h.append(xavier_init(self.trainLen, self.lsd_dim))
        elif a == 'test':
            for num in range(self.input_num):
                torch.manual_seed(self.seed)
                np.random.seed(self.seed)
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
                #tmp = nn.Module()
                #tmp.value = nn.Parameter(xavier_init(self.testLen, self.lsd_dim))
                #h.append(tmp)
                h.append(Variable(xavier_init(self.testLen, self.lsd_dim), requires_grad = True))
                #h.append(xavier_init(self.testLen, self.lsd_dim))
        return h

    def reconstruction_loss(self,h,x,sn):
        loss = 0
        x_pred = self.calculate(h.cuda())
        for i in range(self.view_num):
                loss = loss + (torch.pow((x_pred[str(i)].cpu() - x[str(i)].cpu())
                        , 2.0) * sn[str(i)].cpu()
                ).sum()
        return loss

    def classification_loss(self,label_onehot, gt, h_temp):
        h_temp = h_temp.float()
        h_temp = h_temp.cuda()
        F_h_h = torch.mm(h_temp, (h_temp.T))
        F_hn_hn = torch.eye(F_h_h.shape[0],F_h_h.shape[1])
        F_h_h = F_h_h - F_h_h * (F_hn_hn.cuda())
        label_num = label_onehot.sum(0, keepdim=True)  # should sub 1.Avoid numerical errors; the number of samples of per label
        label_onehot = label_onehot.float()
        F_h_h_sum = torch.mm(F_h_h, label_onehot)
        F_h_h_mean = F_h_h_sum / label_num
        gt1 = torch.max(F_h_h_mean, axis=1)[1]  # gt begin from 1
        gt_ = gt1.type(torch.IntTensor) + 1
        F_h_h_mean_max = torch.max(F_h_h_mean, axis=1, keepdim=False)[0]
        gt_ = gt_.cuda()
        gt_ = gt_.reshape([gt_.shape[0],1])
        theta = torch.ne(gt, gt_).type(torch.FloatTensor)
        F_h_hn_mean_ = F_h_h_mean * label_onehot
        F_h_hn_mean = F_h_hn_mean_.sum(axis=1)
        F_h_h_mean_max = F_h_h_mean_max.reshape([F_h_h_mean_max.shape[0],1])
        F_h_hn_mean = F_h_hn_mean.reshape([F_h_hn_mean.shape[0],1])
        theta = theta.cuda()
        return (torch.nn.functional.relu(theta + F_h_h_mean_max - F_h_hn_mean)).sum()
  
    def rank_loss(self,label_onehot, gt, h_temp_list):
        confidence_list = []
        predict_list = []
        for num in range(self.input_num):
            h_temp = h_temp_list[num]
            h_temp = h_temp.float()
            h_temp = h_temp.cuda()
            F_h_h = torch.mm(h_temp, (h_temp.T))
            F_hn_hn = torch.eye(F_h_h.shape[0],F_h_h.shape[1])
            F_h_h = F_h_h - F_h_h * (F_hn_hn.cuda())
            label_num = label_onehot.sum(0, keepdim=True)  # should sub 1.Avoid numerical errors; the number of samples of per label
            label_onehot = label_onehot.float()
            F_h_h_sum = torch.mm(F_h_h, label_onehot)
            F_h_h_mean = F_h_h_sum / label_num
            probability = F.softmax(F_h_h_mean)
            confidence, predict = torch.max(probability, axis=1)
            #print(predict.shape)
            predict_list.append(predict)
            confidence_list.append(confidence)
        loss = 0
        #print(torch.max(predict_list[0]))
        #print(torch.max(gt))
        target = gt.squeeze()-1
        for num in range(self.input_num-1):
            sign = (~((predict_list[num+1]==target)&(predict_list[num]!=target))).long() #trick 1
            loss = loss + torch.nn.ReLU()(torch.sub(confidence_list[num+1],confidence_list[num])*sign-0.00).sum()
        return loss
        

    def train(self, data, sn, label_onehot, gt, epoch, step=[5, 5]):
        
        index = np.array([x for x in range(self.trainLen)])
        shuffle(index)
        gt = gt.cuda()
        label_onehot = label_onehot.cuda()
        #sn1[j][str(i)] is missing vector for i_th view of j_th input
        sn1 = []
        data1 = dict()
        for v_num in range(self.view_num):
            data1[str(v_num)] = torch.from_numpy(data[str(v_num)]).cuda()
        for j in range(self.input_num):
            tmp = dict()
            for i in range(self.view_num):
                tmp[str(i)] = sn[j, :, i].reshape(self.trainLen, 1).cuda()
            sn1.append(tmp)
        train_hn_op = []
        for num in range(self.input_num):
            train_hn_op.append(torch.optim.Adam([self.h_train[num]], self.learning_rate[1]))
        # start training
        for iter in range(epoch):
            for i in range(step[0]):
                Reconstruction_LOSS = 0
                for num in range(self.input_num):
                    Reconstruction_LOSS = Reconstruction_LOSS + self.reconstruction_loss(self.h_train[num],data1,sn1[num]).float()
                Reconstruction_LOSS = Reconstruction_LOSS/(self.input_num)
                for v_num in range(self.view_num):
                    self.train_net_op[v_num].zero_grad()
                    Reconstruction_LOSS.backward(retain_graph=True)
                for v_num in range(self.view_num):
                    self.train_net_op[v_num].step()
            
            for i in range(step[1]):
                rec_loss = 0
                cls_loss = 0
                for num in range(self.input_num):
                    rec_loss = rec_loss + self.reconstruction_loss(self.h_train[num],data1,sn1[num]).float().cuda() 
                    cls_loss = cls_loss + self.lamb * self.classification_loss(label_onehot, gt, self.h_train[num]).float().cuda()
                rec_loss = rec_loss/self.input_num
                cls_loss = cls_loss/self.input_num
                rank_loss = self.rank_loss(label_onehot, gt, self.h_train).float().cuda()
                total_loss = rec_loss + cls_loss + self.belta*rank_loss

                for num in range(self.input_num):
                    train_hn_op[num].zero_grad()
                total_loss.backward()
                for num in range(self.input_num):
                    train_hn_op[num].step()
            #Classification_LOSS = self.classification_loss(label_onehot,gt,self.h_train)
            #Reconstruction_LOSS = self.reconstruction_loss(self.h_train,data1,sn1)
            output = "Epoch:{:.0f} ===> Reconstruction Loss = {:.4f}, Classification Loss = {:.4f}, Rank Loss = {:.4f}" \
                .format((iter + 1), rec_loss, cls_loss, rank_loss)
            print(output)
            
        return (self.h_train)

    def bulid_model(self):
        # initialize network
        net = nn.ModuleDict()
        train_net_op = []
        for v_num in range(self.view_num):
            net[str(v_num)] = CPMNets(self.view_num, self.trainLen, self.testLen, self.layer_size, v_num,
            self.lsd_dim, self.learning_rate, self.lamb).cuda()
            train_net_op.append(torch.optim.Adam([{"params":net[str(v_num)].parameters()}], self.learning_rate[0]))
        return net,train_net_op

    def calculate(self,h):
        h_views = dict()
        for v_num in range(self.view_num):
            h_views[str(v_num)] = self.net[str(v_num)](h.cuda())
        return h_views

    def test(self, data, sn, epoch):
        sn1 = []
        for num in range(self.input_num):
            tmp = dict()
            for i in range(self.view_num):
                tmp[str(i)] = sn[num, :, i].reshape(self.testLen, 1).cuda()
            sn1.append(tmp)
            
        data1 = dict()
        for v_num in range(self.view_num):
            data1[str(v_num)] = torch.from_numpy(data[str(v_num)]).cuda() 
        adj_hn_op = []
        for num in range(self.input_num):
            adj_hn_op.append(torch.optim.Adam([self.h_test[num]], self.learning_rate[0]))
        for num in range(self.input_num):
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            for iter in range(epoch):
                # update the h
                for i in range(5):
                    Reconstruction_LOSS = 0
                    Reconstruction_LOSS = Reconstruction_LOSS + self.reconstruction_loss(self.h_test[num], data1, sn1[num]).float()
                    adj_hn_op[num].zero_grad()
                    Reconstruction_LOSS.backward()
                    adj_hn_op[num].step()
                output = "Epoch : {:.0f}  ===> Reconstruction Loss = {:.4f}" \
                    .format((iter + 1), Reconstruction_LOSS)
                print(output)

        return self.h_test
        
    def get_h_train(self):
        return self.h_train
        
    def get_h_test(self):
        return self.h_test

    

