import torch
import torch.nn as nn
import torch.nn.functional as F


class JSDivLoss(torch.nn.Module):
    def __init__(self, reduction='batchmean'):
        super(JSDivLoss, self).__init__()
        self.reduction = reduction

    def forward(self, p, q):
        p = F.softmax(p, dim=1)
        q = F.softmax(q, dim=1)

        m = 0.5 * (p + q)

        kl_p_m = F.kl_div(m.log(), p, reduction=self.reduction)
        kl_q_m = F.kl_div(m.log(), q, reduction=self.reduction)

        js_divergence = 0.5 * (kl_p_m + kl_q_m)

        return js_divergence


class KLDivLoss(torch.nn.Module):
    def __init__(self, reduction='batchmean'):
        super(KLDivLoss, self).__init__()
        self.reduction = reduction

    def forward(self, p, q):
        kl_p = F.kl_div(
            F.log_softmax(p,dim=1), 
            F.softmax(q,dim=1), 
            reduction=self.reduction
            )
        kl_q = F.kl_div(
            F.log_softmax(q,dim=1), 
            F.softmax(p,dim=1), 
            reduction=self.reduction
            )

        kl_divergence = (kl_p + kl_q) * 0.5

        return kl_divergence


def contrastive_loss(spk_emb, spk_id, temperature=0.1):
    """
    spk_emb: (B, hidden)
    spk_id: (B,)
    """
    spk_emb = F.normalize(spk_emb, p=2, dim=1)  # (B, d)
    
    sim_matrix = torch.mm(spk_emb, spk_emb.T)  # (B, B)
    sim_matrix /= temperature
    
    labels = spk_id.unsqueeze(0) == spk_id.unsqueeze(1)  # (B, B)
    diag_mask = torch.eye(labels.size(0), dtype=torch.bool, device=labels.device)
    labels = labels.masked_fill(diag_mask, False)  # 排除自身
    
    exp_sim = torch.exp(sim_matrix)
    pos_sum = (exp_sim * labels).sum(dim=1)  # 正样本相似度
    neg_sum = exp_sim.sum(dim=1) - pos_sum  # 负样本相似度
    loss = -torch.log(pos_sum / (pos_sum + neg_sum))
    
    return loss.mean()
