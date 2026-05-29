import torch
import torch.nn as nn

from config.args_config import args
from models.st_tokenizer import StTokenizer
from models.backbone import Backbone
import time


class TrajKST(nn.Module):
    def __init__(self, device):
        super(TrajKST, self).__init__()
        
        self.device = device

        self.tokenizer = StTokenizer(device).to(device)

        self.special_token = nn.Embedding(num_embeddings=2, embedding_dim=args.d_model).to(device)
                
        self.backbone = Backbone(device).to(device)


        
 
    def forward(self, batch_road_id, batch_time_id, mask, num_mask, batch_sub, batch_road_index_id, traj_token_ids, kg_token_ids, traj_window_index):
        kg_text_embedding = self.backbone.gpt2.wte(kg_token_ids)  # [sum_win_num, token_len, 768]

        B = batch_road_id.shape[0]
        D = kg_text_embedding.size(-1)  # 768

        traj_kg_text_emb = torch.zeros(B, D, device=self.device)
        cnt = torch.zeros(B, device=self.device)

        for i, b in enumerate(traj_window_index):
            traj_kg_text_emb[b] += kg_text_embedding[i].mean(dim=0)
            cnt[b] += 1

        cnt = cnt.clamp_min(1.0)
        # [B,1,768]
        traj_kg_text_emb = (traj_kg_text_emb / cnt.unsqueeze(-1)).view(B, 1, -1)  # [B,1,768]

        traj_text_embedding, kg_embedding, traj_st_embedding = self.tokenizer(batch_road_id, batch_time_id, batch_road_index_id, batch_sub, traj_token_ids)


        kg_tokens = torch.cat(
            [kg_embedding, traj_kg_text_emb],dim=1)


        B, L, D = traj_st_embedding.shape


        mask_traj_text_token = traj_text_embedding.masked_fill(mask.unsqueeze(-1) == 0, 0)  # (B, 64, 768)
        mask_traj_st_token = traj_st_embedding.masked_fill(mask.unsqueeze(-1) == 0, 0)  # (B, 64, 768)


        clas_token = self.special_token(torch.tensor([0]).to(self.device)).expand(B, num_mask, D)  # (B, 32, D)  ()
        reg_token = self.special_token(torch.tensor([1]).to(self.device)).expand(B, num_mask, D)   # (B, 32, D)

        special_tokens = torch.stack([clas_token, reg_token], dim=2).view(B, -1, D)  # (B, 2*num_mask, D)  (B,64,768)

        mask_batch_tokens = torch.cat([kg_tokens, mask_traj_text_token, mask_traj_st_token], dim=1)

        batch_psm_tokens = torch.cat([mask_batch_tokens, special_tokens], dim=1) # (B, L + 2*num_mask, D) 拼接张量

        output = self.backbone(batch_psm_tokens, ["road_clas", "time_reg"])

        clas_output, time_output = output["road_clas"], output["time_reg"]
        
        # get output special tokens
        clas_indices = torch.arange(-2 * num_mask, 0, 2).to(self.device)    # 倒数偶数 [-2*num_mask,...,-4,-2]
        reg_indices = torch.arange(-2 * num_mask + 1, 1, 2).to(self.device)     # 倒数奇数 [-2*num_mask+1,...,-3,-1]

        return clas_output[:, clas_indices, :], time_output[:, reg_indices, :]
        
