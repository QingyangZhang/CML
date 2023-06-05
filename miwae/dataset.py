import numpy as np
import scipy.io
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import math


class Multi_view_data(Dataset):
    """
    load multi-view data
    """

    def __init__(self, root, train=True, Normal=1):
        """
        :param root: data name and path
        :param train: load training set or test set
        :param noise: noise level in test set
        """
        super(Multi_view_data, self).__init__()
        self.root = root
        self.train = train
        self.data_path = self.root + '.mat'
        self.data = scipy.io.loadmat(self.data_path)
        self.view_number = self.data['X'].shape[1]
        self.feature_per_view = []
        ratio = 0.8
        
        X = np.split(self.data['X'], self.view_number, axis=1)
        X_train = []
        X_test = []
        labels_train = []
        labels_test = []
        
        self.X_train_mean = {}
        self.X_train_std = {}
        
        if min(self.data['gt']) == 0:
            labels = self.data['gt'] + 1
        else:
            labels = self.data['gt']
        labels = self.data['gt']
        classes = max(labels)[0]+1
        self.classes = max(labels)[0]+1
        
        all_length = 0
        for c_num in range(classes):
            c_length = np.sum(labels == c_num)
            index = np.arange(c_length)
            #shuffle(index)
            labels_train.extend(labels[all_length + index][0:math.floor(c_length * ratio)])
            labels_test.extend(labels[all_length + index][math.floor(c_length * ratio):])
            X_train_temp = []
            X_test_temp = []
            for v_num in range(self.view_number):
                X_train_temp.append(X[v_num][0][0].transpose()[all_length + index][0:math.floor(c_length * ratio)])
                X_test_temp.append(X[v_num][0][0].transpose()[all_length + index][math.floor(c_length * ratio):])
            if c_num == 0:
                X_train = X_train_temp;
                X_test = X_test_temp
            else:
                for v_num in range(self.view_number):
                    X_train[v_num] = np.r_[X_train[v_num], X_train_temp[v_num]]
                    X_test[v_num] = np.r_[X_test[v_num], X_test_temp[v_num]]
            all_length = all_length + c_length
        if (Normal == 1):
            sign = 0
            if self.root.endswith("animal"):
                sign = 1
            if self.root.endswith("yaleB"):
                sign = 2
            if self.root.endswith("nyud2"):
                sign = 2
            for v_num in range(self.view_number):
                X_train[v_num] = Normalize(X_train[v_num], sign)
                X_test[v_num] = Normalize(X_test[v_num], sign)
                
            
        
        if self.train:
            self.X = X_train
            self.y = labels_train
        else:
            self.X = X_test
            self.y = labels_test
        
        for v in range(self.view_number):
            self.feature_per_view.append(self.X[v].shape[1])
            
        

    def __getitem__(self, index):
        data = dict()
        d = []
        for v_num in range(len(self.X)):
            data[v_num] = (self.X[v_num][index]).astype(np.float32)
            d.append((self.X[v_num][index]).astype(np.float32))
        target = self.y[index]
        batch_size = len(target)
        view_num = len(self.X)
        
        full_data = np.concatenate(d, axis=0)
        target = target.squeeze()
        
        return full_data, target

    def __len__(self):
        return len(self.X[0])


def Normalize(x, min=0):
    if min == 0:
        #scaler = StandardScaler(with_mean=False,with_std=False)
        scaler = StandardScaler()
    elif min ==1:
        scaler = StandardScaler(with_mean=False,with_std=False)
    elif min ==2:
        scaler = StandardScaler(with_mean=False)
    else:
        print('error')
        exit()
    
    norm_x = scaler.fit_transform(x)
    
    return norm_x