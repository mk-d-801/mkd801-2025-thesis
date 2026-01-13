import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import random

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0,device='cuda'):
        super(TripletLoss,self).__init__()
        self.margin = margin
        self.device = device
        self.loss_fn = nn.TripletMarginLoss(margin=margin,p=2)

    def forward(self,embeddings,labels):
        anchor,positive,negative = self._get_triplets(embeddings,labels)
        if anchor is None:
            return torch.tensor(0.0,device=self.device,requires_grad=True)
        return self.loss_fn(anchor,positive,negative)
    
    def _get_triplets(self,embeddings,labels):
        anchor,positive,negative = [],[],[]
        labels = labels.detach().cpu().numpy()

        for i in range(len(labels)):
            pos = np.where(labels == labels[i])[0]
            neg = np.where(labels != labels[i])[0]
            if len(pos) < 2 or len(neg) <1:
                continue

            pos_idx = np.random.choice(pos[pos != i])
            neg_idx = np.random.choice(neg)

            anchor.append(embeddings[i])
            positive.append(embeddings[pos_idx])
            negative.append(embeddings[neg_idx])

        if len(anchor) == 0:
                return None,None,None
            
        return (
                torch.stack(anchor).to(self.device),
                torch.stack(positive).to(self.device),
                torch.stack(negative).to(self.device)
            )

triplet = TripletLoss(margin=1.2,device='cuda')