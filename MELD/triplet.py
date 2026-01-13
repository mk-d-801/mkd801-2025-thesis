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

class TripletHardLoss(nn.Module):
    def __init__(self, margin=1.0, device='cuda'):
        super(TripletHardLoss, self).__init__()
        self.margin = margin
        self.device = device
        self.loss_fn = nn.TripletMarginLoss(margin=margin, p=2)

    def forward(self, embeddings, labels):
        anchor, positive, negative = self._get_triplets(embeddings, labels)
        if anchor is None:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        return self.loss_fn(anchor, positive, negative)
    
    def _get_triplets(self, embeddings, labels):
        anchor, positive, negative = [], [], []
        labels_np = labels.detach().cpu().numpy()

        emb_cpu = embeddings.detach().cpu()

        for i in range(len(labels_np)):
            pos = np.where(labels_np == labels_np[i])[0]
            neg = np.where(labels_np != labels_np[i])[0]

            if len(pos) < 2 or len(neg) < 1:
                continue

            anchor_emb = emb_cpu[i]

            pos_idx_candidates = pos[pos != i]
            neg_idx_candidates = neg

            pos_dists = torch.norm(anchor_emb - emb_cpu[pos_idx_candidates], dim=1)
            neg_dists = torch.norm(anchor_emb - emb_cpu[neg_idx_candidates], dim=1)

            pos_idx = pos_idx_candidates[torch.argmax(pos_dists).item()]
            neg_idx = neg_idx_candidates[torch.argmin(neg_dists).item()]

            anchor.append(embeddings[i])
            positive.append(embeddings[pos_idx])
            negative.append(embeddings[neg_idx])

        if len(anchor) == 0:
            return None, None, None

        return (
            torch.stack(anchor).to(self.device),
            torch.stack(positive).to(self.device),
            torch.stack(negative).to(self.device)
        )

class TripletSemiHardLoss(nn.Module):
    def __init__(self, margin=1.0, device='cuda'):
        super(TripletSemiHardLoss, self).__init__()
        self.margin = margin
        self.device = device
        self.loss_fn = nn.TripletMarginLoss(margin=margin, p=2)

    def forward(self, embeddings, labels):
        embeddings = F.normalize(embeddings, p=2, dim=1)
        anchor, positive, negative = self._get_triplets(embeddings, labels)
        if anchor is None:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        return self.loss_fn(anchor, positive, negative)

    def _get_triplets(self, embeddings, labels):
        anchor, positive, negative = [], [], []
        labels_np = labels.detach().cpu().numpy()

        emb_cpu = embeddings.detach().cpu()
        dist_matrix = torch.cdist(emb_cpu, emb_cpu)

        for i in range(len(labels_np)):
            pos = np.where(labels_np == labels_np[i])[0]
            neg = np.where(labels_np != labels_np[i])[0]
            pos = pos[pos != i]

            if len(pos) < 1 or len(neg) < 1:
                continue

            pos_dists = dist_matrix[i, pos]
            pos_idx = pos[torch.argmax(pos_dists).item()]
            d_ap = dist_matrix[i, pos_idx]

            neg_dists = dist_matrix[i, neg]

            semi_hard_mask = (neg_dists > d_ap) & (neg_dists < d_ap + self.margin)

            if not semi_hard_mask.any():
                continue 

            valid_neg_dists = neg_dists[semi_hard_mask]
            valid_neg_idx = neg[semi_hard_mask.cpu().numpy()]
            neg_idx = valid_neg_idx[torch.argmin(valid_neg_dists).item()]

            anchor.append(embeddings[i])
            positive.append(embeddings[pos_idx])
            negative.append(embeddings[neg_idx])

        if len(anchor) == 0:
            return None, None, None
        
        return (
            torch.stack(anchor).to(self.device),
            torch.stack(positive).to(self.device),
            torch.stack(negative).to(self.device)
        )
   
margin_matrix = [
    ["Min","Mid","Mid","Max","Max","Mid","Far"],
    ["Mid","Min","Mid","Max","Max","Mid","Far"],
    ["Mid","Mid","Min","Max","Max","Mid","Far"],
    ["Max","Max","Max","Min","Max","Max","Far"],
    ["Max","Max","Max","Max","Min","Max","Max"],
    ["Mid","Mid","Mid","Max","Max","Min","Far"],
    ["Far","Far","Far","Far","Max","Far","Min"]
]

margin_value = {
    "Min":0.0,
    "Near":1.0,
    "Mid":1.2,
    "Far":1.6,
    "Max":2.0,
    "Fear":2.0
}
def convert_margin_matrix(symbolic, value_dict):
    numeric = []
    for row in symbolic:
        numeric.append([value_dict[x] for x in row])
    return numeric

margin_matrix = convert_margin_matrix(margin_matrix, margin_value)
margin_matrix = torch.tensor(margin_matrix, dtype=torch.float32).cuda()

class TripletMarginLoss(nn.Module):
    def __init__(self, margin_matrix, device='cuda'):
        super(TripletMarginLoss, self).__init__()
        self.margin_matrix = margin_matrix.to(device) 
        self.device = device

    def forward(self, embeddings, labels):
        anchor, positive, negative, a_lab, n_lab = self._get_triplets(embeddings, labels)
        if anchor is None:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        margins = self.margin_matrix[a_lab, n_lab]  

        d_ap = torch.norm(anchor - positive, p=2, dim=1)
        d_an = torch.norm(anchor - negative, p=2, dim=1)

        loss = torch.clamp(d_ap - d_an + margins, min=0.0)
        return loss.mean()

    def _get_triplets(self, embeddings, labels):
        anchor, positive, negative = [], [], []
        a_lab_list, n_lab_list = [], []

        labels_np = labels.detach().cpu().numpy()

        for i in range(len(labels_np)):
            pos = np.where(labels_np == labels_np[i])[0]
            neg = np.where(labels_np != labels_np[i])[0]

            if len(pos) < 2 or len(neg) < 1:
                continue

            pos_idx = np.random.choice(pos[pos != i])
            neg_idx = np.random.choice(neg)

            anchor.append(embeddings[i])
            positive.append(embeddings[pos_idx])
            negative.append(embeddings[neg_idx])

            a_lab_list.append(labels_np[i])
            n_lab_list.append(labels_np[neg_idx])

        if len(anchor) == 0:
            return None, None, None, None, None

        return (
            torch.stack(anchor).to(self.device),
            torch.stack(positive).to(self.device),
            torch.stack(negative).to(self.device),
            torch.tensor(a_lab_list, dtype=torch.long).to(self.device),
            torch.tensor(n_lab_list, dtype=torch.long).to(self.device),
        )

class TripletHardMarginLoss(nn.Module):
    def __init__(self, margin_matrix, device='cuda'):
        super(TripletHardMarginLoss, self).__init__()
        self.margin_matrix = margin_matrix.to(device)
        self.device = device

    def forward(self, embeddings, labels):
        anchor, positive, negative, a_lab, n_lab = self._get_hard_triplets(embeddings, labels)
        
        if anchor is None:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        margins = self.margin_matrix[a_lab, n_lab]

        d_ap = torch.norm(anchor - positive, p=2, dim=1)
        d_an = torch.norm(anchor - negative, p=2, dim=1)

        loss = torch.clamp(d_ap - d_an + margins, min=0.0)
        
        return loss.mean()

    def _get_hard_triplets(self, embeddings, labels):
        anchor, positive, negative = [], [], []
        a_lab_list, n_lab_list = [], []

        labels_np = labels.detach().cpu().numpy()
        emb_cpu = embeddings.detach().cpu()

        for i in range(len(labels_np)):
            pos = np.where(labels_np == labels_np[i])[0]
            neg = np.where(labels_np != labels_np[i])[0]

            if len(pos) < 2 or len(neg) < 1:
                continue

            anchor_emb = emb_cpu[i]
            pos_idx_candidates = pos[pos != i]
            neg_idx_candidates = neg

            pos_dists = torch.norm(anchor_emb - emb_cpu[pos_idx_candidates], p=2, dim=1)
            hard_pos_idx = pos_idx_candidates[torch.argmax(pos_dists).item()]

            neg_dists = torch.norm(anchor_emb - emb_cpu[neg_idx_candidates], p=2, dim=1)
            hard_neg_idx = neg_idx_candidates[torch.argmin(neg_dists).item()]

            anchor.append(embeddings[i])
            positive.append(embeddings[hard_pos_idx])
            negative.append(embeddings[hard_neg_idx])

            a_lab_list.append(labels_np[i])
            n_lab_list.append(labels_np[hard_neg_idx])

        if len(anchor) == 0:
            return None, None, None, None, None

        return (
            torch.stack(anchor).to(self.device),
            torch.stack(positive).to(self.device),
            torch.stack(negative).to(self.device),
            torch.tensor(a_lab_list, dtype=torch.long).to(self.device),
            torch.tensor(n_lab_list, dtype=torch.long).to(self.device),
        )

class TripletSemiHardMarginLoss(nn.Module):
    def __init__(self, margin_matrix, device='cuda'):
        super(TripletSemiHardMarginLoss, self).__init__()
        self.margin_matrix = margin_matrix.to(device)
        self.device = device

    def forward(self, embeddings, labels):
        embeddings = F.normalize(embeddings, p=2, dim=1)
        anchor, positive, negative, a_lab, n_lab = self._get_triplets(embeddings, labels)

        if anchor is None:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        d_ap = torch.norm(anchor - positive, p=2, dim=1)
        d_an = torch.norm(anchor - negative, p=2, dim=1)

        margins = self.margin_matrix[a_lab, n_lab]

        loss = torch.clamp(d_ap - d_an + margins, min=0.0)
        return loss.mean()

    def _get_triplets(self, embeddings, labels):
        anchor, positive, negative = [], [], []
        a_lab_list, n_lab_list = [], []

        labels_np = labels.detach().cpu().numpy()
        emb_cpu = embeddings.detach().cpu()
        dist_matrix = torch.cdist(emb_cpu, emb_cpu)

        for i in range(len(labels_np)):
            pos = np.where(labels_np == labels_np[i])[0]
            neg = np.where(labels_np != labels_np[i])[0]
            pos = pos[pos != i]

            if len(pos) < 1 or len(neg) < 1:
                continue

            pos_dists = dist_matrix[i, pos]
            pos_idx = pos[torch.argmax(pos_dists).item()]
            d_ap = dist_matrix[i, pos_idx]

            neg_dists = dist_matrix[i, neg]
            margin_candidates = self.margin_matrix[
                labels_np[i], labels_np[neg]
            ].cpu()

            semi_hard_mask = (neg_dists > d_ap) & (neg_dists < d_ap + margin_candidates)

            if not semi_hard_mask.any():
                continue

            valid_neg_dists = neg_dists[semi_hard_mask]
            valid_neg_idx = neg[semi_hard_mask.numpy()]
            neg_idx = valid_neg_idx[torch.argmin(valid_neg_dists).item()]

            anchor.append(embeddings[i])
            positive.append(embeddings[pos_idx])
            negative.append(embeddings[neg_idx])

            a_lab_list.append(labels_np[i])
            n_lab_list.append(labels_np[neg_idx])

        if len(anchor) == 0:
            return None, None, None, None, None

        return (
            torch.stack(anchor).to(self.device),
            torch.stack(positive).to(self.device),
            torch.stack(negative).to(self.device),
            torch.tensor(a_lab_list, dtype=torch.long).to(self.device),
            torch.tensor(n_lab_list, dtype=torch.long).to(self.device),
        )

triplet = TripletLoss(margin=1.2,device='cuda')