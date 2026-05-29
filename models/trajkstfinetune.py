import torch
import torch.nn as nn

from transformers import GPT2Tokenizer

from config.args_config import args

from models.st_tokenizer import StTokenizer
from models.backbone import Backbone
from models.trajkst import TrajKST
import torch.nn.functional as F



TRAJ_RECOVER_PROMPT = "Recover the vacant road segments in the trajectory, generate the road segment id corresponding to each" # [CLS]
DES_PREDICTION_PROMPT = "Predict the destination of the trajectory, generate the road segment id corresponding to each"     # [CLS]
TTE_PROMPT = "predict the travel time of the trajectory, regress the time interval between each segment and its former one"  # [REG]


def length_to_mask(traj_len, max_len, device):
    # traj_len: [B]
    traj_len = torch.as_tensor(traj_len, device=device, dtype=torch.long).view(-1)
    traj_len = traj_len.clamp(min=0, max=max_len)
    idx = torch.arange(max_len, device=device).unsqueeze(0)   # [1, T]
    mask = idx < traj_len.unsqueeze(1)                        # [B, T]
    return mask

def masked_mean_pooling(hidden, mask):
    # hidden: [B, T, D]
    # mask:   [B, T]
    mask = mask.unsqueeze(-1).type_as(hidden)                 # [B, T, 1]
    summed = (hidden * mask).sum(dim=1)                       # [B, D]
    denom = mask.sum(dim=1).clamp(min=1e-6)                   # [B, 1]
    return summed / denom


class TrajKSTFineTune(TrajKST):
    def __init__(self, device, checkpoint=None):
        super(TrajKSTFineTune, self).__init__(device)

        # 读取第一阶段训练的最好的模型参数
        # if checkpoint is None:
        #     checkpoint = torch.load(f"./checkpoints/{args.city}_pretrain_best.pth", weights_only=True)
        # else:
        #     checkpoint = torch.load(checkpoint, weights_only=True)
        #self.load_state_dict(checkpoint["model_state_dict"], strict=False)
        self.load_pretrain_checkpoint(checkpoint)

        # ST-tokenizer
        for name, param in self.tokenizer.named_parameters():
            param.requires_grad = False

        self.special_token.requires_grad_(False)
        
        self.prompt_tokenizer = GPT2Tokenizer.from_pretrained("./models/gpt2")

        self.TRAJ_RECOVER_PROMPT = self.encode_prompt(TRAJ_RECOVER_PROMPT, 0)
        self.DES_PREDICTION_PROMPT = self.encode_prompt(DES_PREDICTION_PROMPT, 0)   # 增加目的地预测文本提示
        self.TTE_PROMPT = self.encode_prompt(TTE_PROMPT, 1)

        self.mse = nn.MSELoss()
        self.cross_entropy = nn.CrossEntropyLoss()

    def load_pretrain_checkpoint(self, checkpoint=None):
        ckpt_path = checkpoint or f"./checkpoints/{args.city}_pretrain_best.pth"
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)

        pretrained_state = checkpoint["model_state_dict"]
        current_state = self.state_dict()
        load_state = {}

        for k, v in pretrained_state.items():
            if k not in current_state:
                continue

            # 1) shape 完全一致，直接加载
            if current_state[k].shape == v.shape:
                load_state[k] = v
                continue

            # 2) 特判 space_road: 旧 checkpoint 少一行，新模型多一行
            if k == "tokenizer.space_road" \
                    and current_state[k].dim() == 2 \
                    and v.dim() == 2 \
                    and current_state[k].shape[1] == v.shape[1] \
                    and current_state[k].shape[0] == v.shape[0] + 1:
                tmp = current_state[k].clone()
                tmp[:v.shape[0]].copy_(v.to(tmp.device))

                # 如果新增这一行表示 padding token，用 0 更合适
                # tmp[v.shape[0]].zero_()

                # 如果新增这一行表示 mask token，保留当前初始化即可
                load_state[k] = tmp
                print(f"Partially loaded `{k}`: {tuple(v.shape)} -> {tuple(current_state[k].shape)}")
                continue

            print(f"Skip `{k}`: ckpt {tuple(v.shape)} != model {tuple(current_state[k].shape)}")

        msg = self.load_state_dict(load_state, strict=False)
        print(f"Loaded checkpoint from: {ckpt_path}")
        print("missing_keys:", msg.missing_keys)
        print("unexpected_keys:", msg.unexpected_keys)

    def encode_prompt(self, prompt, special_token_id):
        # 单条处理（只返回一条文本对应的）
        encoded_prompt = self.prompt_tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        prompt_embedding = self.backbone.gpt2.wte(encoded_prompt)
        
        special_token = self.special_token(torch.tensor([special_token_id]).to(self.device)).unsqueeze(1)
        
        prompt_with_special_token = torch.cat([prompt_embedding, special_token], dim=1)
        
        return prompt_with_special_token

        
    def forward(self, task_name, batch_road_id, batch_time_id, batch_sub, batch_road_index_id, traj_token_ids, kg_token_ids, traj_window_index, batch_label,batch_label_mask=None):

        kg_text_embedding = self.backbone.gpt2.wte(kg_token_ids)  # [sum_win_num, token_len, 768]

        B = batch_road_id.shape[0]  # 32
        D = kg_text_embedding.size(-1)  # 768

        traj_kg_text_emb = torch.zeros(B, D, device=self.device)
        cnt = torch.zeros(B, device=self.device)

        for i, b in enumerate(traj_window_index):
            traj_kg_text_emb[b] += kg_text_embedding[i].mean(dim=0)
            cnt[b] += 1

        cnt = cnt.clamp_min(1.0)
        # [B, 768]
        traj_kg_text_emb = (traj_kg_text_emb / cnt.unsqueeze(-1)).view(B, 1, -1)  # [B,1,768]

        traj_text_embedding, kg_embedding, traj_st_embedding = self.tokenizer(batch_road_id, batch_time_id,
                                                                              batch_road_index_id, batch_sub, traj_token_ids)
        kg_tokens = torch.cat([kg_embedding, traj_kg_text_emb], dim=1)

        B, L, D = traj_st_embedding.shape

        batch_tokens = torch.cat([kg_tokens, traj_text_embedding, traj_st_embedding], dim=1)


        if task_name == "traj_recover":
            num_mask_per_seq = int(args.mask_rate * L)

            clas_token = self.special_token(torch.tensor([0]).to(self.device)).expand(B, num_mask_per_seq, D)
            batch_psm_tokens = torch.cat([self.TRAJ_RECOVER_PROMPT.expand(B, -1, -1), batch_tokens, clas_token],
                                         dim=1)  # L = 19 + L + num_mask_per_seq

            traj_recover_road_clas_output = self.backbone(batch_psm_tokens, ["road_clas"])["road_clas"]

            predict_road_id = traj_recover_road_clas_output[:, -num_mask_per_seq:, :]

            traj_recover_loss = self.cross_entropy(predict_road_id.reshape(-1, predict_road_id.shape[-1]),
                                                   batch_label.view(-1))
            return traj_recover_loss, predict_road_id

        elif task_name == "tte":
            reg_token = self.special_token(torch.tensor([1]).to(self.device)).expand(B, L, D)
            batch_psm_tokens = torch.cat([self.TTE_PROMPT.expand(B, -1, -1), batch_tokens, reg_token],
                                         dim=1)  # L = 16 + L + L

            tte_output = self.backbone(batch_psm_tokens, ["tte"])["tte"]

            predict_time_features = tte_output[:, -L:, :]  # (B, L, 1)

            predict_delta_time = F.softplus(predict_time_features)  # [B, L, 1]

            if batch_label.dim() == 2:
                batch_label = batch_label.unsqueeze(-1)

            if batch_label_mask is None:
                raise ValueError("TTE task requires batch_label_mask.")

            valid_mask = batch_label_mask.unsqueeze(-1).bool()
            tte_loss = F.mse_loss(
                predict_delta_time[valid_mask],
                batch_label[valid_mask]
            )

            return tte_loss, predict_delta_time

        elif task_name == "destination_prediction":

            clas_token = self.special_token(torch.tensor([0]).to(self.device)).expand(B, 1, D)

            batch_psm_tokens = torch.cat([self.DES_PREDICTION_PROMPT.expand(B, -1, -1), batch_tokens, clas_token], dim=1)  # L = 17 + L + 1

            des_prediction_road_clas_output = self.backbone(batch_psm_tokens, ["road_clas"])["road_clas"]

            predict_road_id = des_prediction_road_clas_output[:, -1, :]  # (B, Nroad)

            des_prediction_loss = self.cross_entropy(predict_road_id, batch_label)
            #print(f"des_prediction_loss:{des_prediction_loss}")
            return des_prediction_loss, predict_road_id
