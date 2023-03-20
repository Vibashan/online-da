
from __future__ import print_function

import torch
import torch.nn as nn

import torch.nn.functional as F
import pdb

class MemConLoss_trans(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        super(MemConLoss_trans, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def get_score(self, mem_bank, query, items=None):
        bs, h, w, d = query.size()
        m, d = mem_bank.size()
        score = torch.matmul(query.float(), torch.t(mem_bank).float())# b X h X w X m
        score = score.view(bs*h*w, m) # 300x512
        score_memory = F.softmax(score,dim=1) # 300x512

        _, top_neg_idx = torch.topk(score_memory, items, dim=1, largest=False)

        neg_logits = torch.gather(score, 1, top_neg_idx)

        return neg_logits

    def forward(self, s_query, s_box_feat, mem_s_query, s_value, t_box_feat, t_value, mem_bank):
        
        batch_size, dim = s_query.shape
        mask = torch.eye(batch_size, dtype=torch.float32).cuda()

        anchor_feat = F.normalize(s_query, dim=1)
        contrast_feat = F.normalize(mem_s_query, dim=1)

        logits = torch.div(torch.matmul(anchor_feat, contrast_feat.T), self.temperature)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        sm_logits = logits - logits_max.detach()

        mem_query = s_box_feat.mean(dim=[2, 3]).contiguous().unsqueeze(-1).unsqueeze(-1).permute(0,2,3,1).detach()
        sm_neg_logits = self.get_score(mem_bank, mem_query, items=5)

        s_all_logits = torch.exp(torch.cat((sm_logits, sm_neg_logits), dim=1))
        log_prob = sm_logits - torch.log(s_all_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)  

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos 

        if torch.isnan(loss.mean()):
            loss = loss*0
            
        return loss.mean()
