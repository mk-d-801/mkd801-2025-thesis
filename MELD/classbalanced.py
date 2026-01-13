import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import random

class ClassBalancedLoss(nn.Module):
    def __init__(self, samples_per_cls, beta=0.99, device='cuda'):
        super(ClassBalancedLoss, self).__init__()

        effective_num = 1.0 - np.power(beta, samples_per_cls)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * len(samples_per_cls)  
        self.weights = torch.tensor(weights, dtype=torch.float32).to(device)
        self.loss_fct = nn.CrossEntropyLoss(weight=self.weights)

    def forward(self, pred_outs, labels):
        return self.loss_fct(pred_outs, labels)


samples_per_cls = [345,68,50,402,1256,208,281]
CBLoss = ClassBalancedLoss(samples_per_cls, beta=0.99,device='cuda')
