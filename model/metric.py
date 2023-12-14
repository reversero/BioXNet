#!/usr/bin/env python3
import numpy as np
from sklearn import metrics
from scipy.stats import pearsonr

def accuracy(y_pred, y_true):
    """
    Note that the input y_pred and y_true are both numpy array.
    Args:
        y_pred (numpy.array): 
        y_true (numpy.array): [description]
    """
    return metrics.accuracy_score(y_pred=y_pred.round(), y_true=y_true)

def precision(y_pred, y_true):
    """
    Note that the input y_pred and y_true are both numpy array.
    Args:
        y_pred (numpy.array): 
        y_true (numpy.array): [description]
    """
    return metrics.precision_score(y_pred=y_pred.round(), y_true=y_true)

def recall(y_pred, y_true):
    """
    Note that the input y_pred and y_true are both numpy array.
    Args:
        y_pred ([type]): [description]
        y_true ([type]): [description]
    Returns:
        [type]: [description]
    """
    return metrics.recall_score(y_pred=y_pred.round(), y_true=y_true)

def roc_auc(y_pred, y_true):
    """
    The values of y_pred can be decimicals, within 0 and 1.
    Args:
        y_pred ([type]): [description]
        y_true ([type]): [description]
    Returns:
        [type]: [description]
    """
    return metrics.roc_auc_score(y_score=y_pred, y_true=y_true)

def pr_auc(y_pred, y_true):
    return metrics.average_precision_score(y_score=y_pred, y_true=y_true)

def f1_score(y_pred, y_true):
    return metrics.f1_score(y_pred=y_pred.round(), y_true=y_true)

def rmse(y_pred, y_true):
    return metrics.mean_squared_error(y_pred=y_pred, y_true=y_true, squared=False)

def pearson_roi(y_pred, y_true):
    return pearsonr(x=y_pred, y=y_true)[0]

def pearson_pval(y_pred, y_true):
    return pearsonr(x=y_pred, y=y_true)[1]

def r_squared(y_pred, y_true):
    return metrics.r2_score(y_pred=y_pred, y_true=y_true)

def multi_acc(y_pred, y_true):
    # y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    y_pred_tags = np.argmax(y_pred, axis=1)    
    acc = (y_pred_tags == y_true).sum() / len(y_true)
    
    return acc

def multi_precision(y_pred, y_true):
    # y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    y_pred_tags = np.argmax(y_pred, axis=1)    
    precision = metrics.precision_score(y_pred=y_pred_tags, y_true=y_true, average='weighted')
    
    return precision

def multi_recall(y_pred, y_true):
    # y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    y_pred_tags = np.argmax(y_pred, axis=1)    
    recall = metrics.recall_score(y_pred=y_pred_tags, y_true=y_true, average='weighted')
    
    return recall

def multi_f1_score(y_pred, y_true):
    # y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    y_pred_tags = np.argmax(y_pred, axis=1)    
    f1 = metrics.f1_score(y_pred=y_pred_tags, y_true=y_true, average='weighted')
    
    return f1