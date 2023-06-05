import numpy as np
from sklearn.preprocessing import OneHotEncoder
import torch
import torch.nn.functional as F

def convert_to_one_hot(y, C):
    return np.eye(C)[y.reshape(-1)]


def vote(lsd1, lsd2, label, n=1): 
    """Sometimes the prediction accuracy will be higher in this way.
    :param lsd1: train set's latent space data
    :param lsd2: test set's latent space data
    :param label: label of train set
    :param n: Similar to K in k-nearest neighbors algorithm
    :return: Predicted label
    """
    F_h_h = np.dot(lsd2, np.transpose(lsd1))
    gt_list = []
    label = label.reshape(len(label), 1)
    for num in range(n):
        F_h_h_argmax = np.argmax(F_h_h, axis=1)
        F_h_h_onehot = convert_to_one_hot(F_h_h_argmax, len(label))
        F_h_h = F_h_h - np.multiply(F_h_h, F_h_h_onehot)
        gt_list.append(np.dot(F_h_h_onehot, label))
    gt_ = np.array(gt_list).transpose(2, 1, 0)[0].astype(np.int64)
    count_list = []
    count_list.append([np.argmax(np.bincount(gt_[i])) for i in range(lsd2.shape[0])])
    gt_pre = np.array(count_list)
    return gt_pre.transpose()

def ave(lsd1_list, lsd2_list, label_onehot, trainLen): 
    """In most cases, this method is used to predict the highest accuracy.
    :param lsd1: train set's latent space data
    :param lsd2: test set's latent space data
    :param label: label of train set
    :return: Predicted label
    """
    gt_list = []
    confidence_list = []
    logits_list = []
    probability_list = []
    for num in range(len(lsd1_list)):
        lsd1 = lsd1_list[num]
        lsd2 = lsd2_list[num]
        F_h_h = torch.mm(lsd2, (lsd1.T))
        label_num = label_onehot.sum(0, keepdim=True)  # should sub 1.Avoid numerical errors; the number of samples of per label
        label_onehot = label_onehot.float()
        F_h_h_sum = torch.mm(F_h_h, label_onehot)
        F_h_h_mean = F_h_h_sum / label_num
        probability = F.softmax(F_h_h_mean)
        confidence = torch.max(probability, axis=1)[0]
        confidence_list.append(confidence.cpu().detach().numpy())
        gt1 = torch.max(F_h_h_mean, axis=1)[1]  # gt begin from 1
        gt_ = gt1.type(torch.IntTensor) + 1
        gt_ = gt_.cuda()
        gt_ = gt_.reshape([gt_.shape[0],1])
        gt_list.append(gt_.cpu().detach().numpy())
        logits_list.append(F_h_h_mean.cpu().detach().numpy())
        probability_list.append(probability.cpu().detach().numpy())

    return gt_list, confidence_list, logits_list, probability_list
