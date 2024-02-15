import torch
import torch.nn as nn
import torch.nn.functional as F

'''
    负对数似然损失
'''
def nll_loss(output, target):
    return F.nll_loss(output, target)


'''
    二元交叉熵损失
'''
def bce_loss(output, target, class_weight=None):
    if class_weight:
        weight = torch.zeros_like(target)
        weight[target==0] = class_weight[0]
        weight[target==1] = class_weight[1]
        return F.binary_cross_entropy(input=output, target=target,
                                      weight=weight)
    else:
        return F.binary_cross_entropy(input=output, target=target)
    
    
'''
    带权重的带logits的二元交叉熵损失
'''
def bce_withlogits_loss_weighted(output, target, class_weight=None):
    if class_weight is not None:
        weight = torch.zeros_like(target)
        weight[target==0] = class_weight[0]
        weight[target==1] = class_weight[1]
        return F.binary_cross_entropy_with_logits(input=output, target=target,
                                                  weight=weight)
    else:
        return F.binary_cross_entropy_with_logits(input=output, target=target)


'''
    带权重的交叉熵损失
'''
def cross_entropy_loss_weighted(output, target, class_weight=None):
    target = torch.squeeze(target.long())
    if class_weight is not None:
        return F.cross_entropy(output, target, weight=class_weight)
    else:
        return F.cross_entropy(output, target)


'''
    均方根误差（RMSE）损失
'''
def rmse_loss(output, target):
    return torch.sqrt(F.mse_loss(output, target))