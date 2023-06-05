import util.classfiy as classfiy
import numpy as np
from numpy.random import shuffle
from util.util import xavier_init
import torch
import torch.nn as nn
from torch.autograd import Variable

class CPMNets(nn.Module): # The architecture of the CPM
    """build model
    """
    def __init__(self, view_num, trainLen, testLen, layer_size, v, lsd_dim=128, learning_rate=[0.001, 0.001], lamb=1):
        """
        :param learning_rate:learning rate of network and h
        :param view_num:view number
        :param layer_size:node of each net
        :param lsd_dim:latent space dimensionality
        :param trainLen:training dataset samples
        :param testLen:testing dataset samples
        """
        super(CPMNets, self).__init__()
        # initialize parameter
        self.view_num = view_num
        self.layer_size = layer_size
        self.lsd_dim = lsd_dim
        self.trainLen = trainLen
        self.testLen = testLen
        self.lamb = lamb
        #initialize forward methods 
        self.net = self._make_view(v).cuda()

    def forward(self,h):
        h_views = self.net(h.cuda())
        return h_views
    '''
    def initialize_weight(self, dims_net):
        all_weight = dict()
        all_weight['w0'] = Variable(xavier_init(self.lsd_dim, dims_net[0]),requires_grad = True)
        all_weight['b0'] = Variable(torch.zeros([dims_net[0]]),requires_grad = True)
        for num in range(1, len(dims_net)):
            all_weight['w' + str(num)] = Variable(xavier_init(dims_net[num - 1], dims_net[num]),requires_grad = True)
            all_weight['b' + str(num)] = Variable(torch.zeros([dims_net[num]]),requires_grad = True)
        return all_weight
    '''
    def _make_view(self, v):
        dims_net = self.layer_size[v]
        net1 = nn.Sequential()
        w = torch.nn.Linear(self.lsd_dim, dims_net[0])
        nn.init.xavier_normal_(w.weight)
        nn.init.constant_(w.bias, 0.0)
        net1.add_module('lin'+str(0), w)
        for num in range(1, len(dims_net)):
            w = torch.nn.Linear(dims_net[num - 1], dims_net[num])
            nn.init.xavier_normal_(w.weight)
            nn.init.constant_(w.bias, 0.0)
            net1.add_module('lin'+str(num), w)
            net1.add_module('drop'+str(num), torch.nn.Dropout(p=0.1))
        return net1
    
