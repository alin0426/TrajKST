import torch
import logging
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm

from config import global_vars
from config.args_config import args
from .layers import MLP, GAT, CrossAttention, GAT_graph, Sentence_Transformer
from data_provider.file_loader import file_loader


from transformers import GPT2Tokenizer
import time
from models.backbone import Backbone
from torch_geometric.utils import from_networkx
from torch_geometric.data import Batch
from torch_scatter import scatter
from torch_geometric.data.data import Data

from data_provider.data_loader import Dataset
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
import datetime
import os
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class StTokenizer(nn.Module):
    def __init__(self, device):
        logging.info("Start initializing the ST tokenizer.")
        super(StTokenizer, self).__init__()
        
        self.device = device

        
        self.road_cnt = None


        self.osmid_to_index = None
        self.load_osmid_to_index()

        self.batch_road_id = None
        self.batch_time_id = None
        self.batch_time_features = None
        self.text_tokenizer = GPT2Tokenizer.from_pretrained("./models/gpt2")
        self.backbone = Backbone(device).to(device)

        self.batch_sub = None
        self.batch_road_index_id = None

        # 子图
        self.graph_encoder = None
        self.projector = None   # MLP

        self.node_id2idx = None
        self.node_emb = None
        self.edge_key2idx = None
        self.edge_emb = None
        self.road2eid = None

        self.load_kg()  # 加载知识图谱相关embedding表

        # 时空特征
        self.time_day = None
        self.time_week = None
        self.padding_idx = None
        self.space_road = None
        self.kge_proj = None
        self.feature_fusion = None

        self.build_tokenizer()


        
        logging.info("Finish initializing the ST tokenizer.")


    def encode_text(self, text_list):
        self.text_tokenizer.pad_token = self.text_tokenizer.eos_token  # 1 行搞定

        out = self.text_tokenizer(
            text_list,
            truncation=True,
            padding=True,
            return_tensors='pt'
        )
        return out['input_ids']  # (B, L_text)  值域 0~50256



    def nx_to_pyg_new(self, G):
        if G is None or G.number_of_nodes() == 0:
            print(f"节点为空")
            return None, "[EMPTY_SUBGRAPH]"

        sub_nodes_sorted = sorted(G.nodes(), key=int)  # 按 id 升序
        node_idx_map = {nid: i for i, nid in enumerate(sub_nodes_sorted)}   # 局部index映射
        x = self.node_emb[[self.node_id2idx[nid] for nid in sub_nodes_sorted]]  # [node_num, 1024]

        edge_index = []
        edge_attr_list = []
        for u, v, k, d in G.edges(keys=True, data=True):
            head_id = u
            tail_id = v
            r_id = d['relation_id']
            r_name = d['relation_name']
            conf = d['confidence']
            edge_index.append([
                node_idx_map[head_id],
                node_idx_map[tail_id]
            ])

            txt_vec = self.edge_emb[self.rel_id2idx[r_id]]  # [1024]
            conf_tensor = torch.tensor([conf], device=self.device)
            edge_attr_list.append(torch.cat([txt_vec, conf_tensor]))  # [1025]



        if len(edge_index) == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
            edge_attr_list = torch.empty((0, x.size(1) + 1), device=self.device)
        else:
            edge_index = torch.tensor(edge_index, dtype=torch.long, device=self.device).t().contiguous()
            edge_attr_list = torch.stack(edge_attr_list)


        pyg_graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr_list, num_nodes=len(sub_nodes_sorted))


        if pyg_graph.edge_index.numel() > 0:
            assert pyg_graph.edge_index.max().item() < pyg_graph.num_nodes, \
                "edge_index contains out-of-bound node indices!"

        return pyg_graph

    def osm_ids_to_indices(self, osm_ids):
        return [self.osm_id_to_index[int(osm_id)] for osm_id in osm_ids]


    def forward(self, batch_road_id, batch_time_id, batch_road_index_id, batch_sub, traj_token_ids):

        B, N, L = batch_road_id.shape[0], self.road_cnt, args.seq_len

        self.batch_road_id = batch_road_id
        self.batch_time_id = batch_time_id

        # SEMATIC
        traj_point_text_token_ids = traj_token_ids.view(-1, 45)

        traj_point_text_embedding = self.backbone.gpt2.wte(traj_point_text_token_ids)  # [B*64, 45, 768]

        # Mean pooling
        traj_point_text_embedding = traj_point_text_embedding.mean(dim=1)
        traj_text_embedding = traj_point_text_embedding.view(B, L, -1)


        # TSE-1部分
        self.batch_sub = batch_sub

        pyg_subs = []
        for g in batch_sub:
            data = self.nx_to_pyg_new(g)
            pyg_subs.append(data)

        batch_pyg_subs = Batch.from_data_list(pyg_subs).to(self.device)


        #TSE-2部分
        n_embeds, _ = self.graph_encoder(   # GAT
            batch_pyg_subs.x,
            batch_pyg_subs.edge_index.long(),
            batch_pyg_subs.edge_attr
        )

        # mean pooling
        g_embeds = scatter(n_embeds, batch_pyg_subs.batch, dim=0, reduce='mean')    # [batch_size, gnn_hidden_dim]，[32, 1024]
        kg_embedding = self.projector(g_embeds)     # [batch_size, 768]，MLP [32, 768]
        kg_embedding = kg_embedding.unsqueeze(1)  # [B, 1, 768]


        ts_min = batch_time_id // 60  # 总分钟数
        minute_of_day = (ts_min) % (24 * 60)
        batch_day_id = minute_of_day // 30

        minute_of_week = ts_min % (7 * 24 * 60)  # 0 … 10079
        batch_week_id = minute_of_week // (24 * 60)

        batch_day_embedding = self.time_day[batch_day_id.type(torch.LongTensor)]    # (B, seq_len, 384)

        batch_day_embedding = batch_day_embedding.transpose(1, 2).unsqueeze(-1) #(B, 384, seq_len, 1)

        batch_week_embedding = self.time_week[batch_week_id.type(torch.LongTensor)]  # (B, seq_len, 384)

        batch_week_embedding = batch_week_embedding.transpose(1, 2).unsqueeze(-1)  # (B, 384, seq_len, 1)


        traj_time_embedding = batch_day_embedding + batch_week_embedding    # (B, 384, seq_len, 1)  [32, 384, 64, 1]

        batch_road_index_id = batch_road_index_id.long()

        batch_road_embedding = self.space_road[batch_road_index_id]

        road_eid = self.road2eid_tensor[batch_road_id]
        node_idx = self.node_id2idx_tensor[road_eid]
        batch_road_kge = self.node_emb[node_idx]
        batch_road_kge = self.kge_proj(batch_road_kge)

        batch_space_feat = torch.cat([batch_road_embedding, batch_road_kge], dim=-1)

        traj_space_embedding = batch_space_feat.transpose(1, 2).unsqueeze(-1)  # (B, 384+128, seq_len, 1)


        traj_st_embedding = torch.cat([traj_time_embedding] + [traj_space_embedding], dim=1)    # (B, 768, seq_len, 1)

        traj_st_embedding = self.feature_fusion(traj_st_embedding)  # (B, 768, seq_len, 1)

        traj_st_embedding = traj_st_embedding.permute(0, 2, 1, 3).squeeze(-1)   # (B, seq_len, 768)


        return traj_text_embedding, kg_embedding, traj_st_embedding




    def load_kg(self):
        self.node_id2idx = file_loader.get_node_id2idx()
        self.node_emb = file_loader.get_node_emb().to(self.device)
        self.edge_key2idx = file_loader.get_edge_key2idx()
        self.rel_id2idx = file_loader.get_rel_id2idx()
        self.edge_emb = file_loader.get_edge_emb().to(self.device)
        self.road2eid = file_loader.get_road2eid()

        max_node_id = max(self.node_id2idx.keys())
        node_id2idx_tensor = torch.full(
            (max_node_id + 1,),
            -1,
            dtype=torch.long
        )
        for node_id, idx in self.node_id2idx.items():
            node_id2idx_tensor[node_id] = idx
        self.register_buffer(
            "node_id2idx_tensor",
            node_id2idx_tensor
        )

        max_road_id = max(self.road2eid.keys())
        road2eid_tensor = torch.full(
            (max_road_id + 1,),
            -1,
            dtype=torch.long
        )
        for road_id, eid in self.road2eid.items():
            road2eid_tensor[road_id] = eid
        self.register_buffer(
            "road2eid_tensor",
            road2eid_tensor
        )

    def load_osmid_to_index(self):
        self.osmid_to_index = file_loader.get_osmid_to_index()
        self.road_cnt = file_loader.get_road_cnt()

    def load_relation(self):
        self.edge_cnt = file_loader.get_edge_cnt()
        self.edges = file_loader.get_edges().to(self.device)
        self.edge_weight = file_loader.get_edge_weight().to(self.device)

        
    def build_tokenizer(self):
        logging.info("Start building static ST tokenizer.")

        # kg_graph_embedding_layer
        self.graph_encoder = GAT_graph(
            in_channels=1024,
            out_channels=1024,
            hidden_channels=1024,
            num_layers=4,
            dropout=0.0,
            num_heads=4,
        ).to(self.device)

        self.projector = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.Sigmoid(),
            nn.Linear(2048, 768),
        ).to(self.device)



        self.time_day = nn.Parameter(torch.empty(48, 384))
        nn.init.xavier_uniform_(self.time_day)


        self.time_week = nn.Parameter(torch.empty(7, 384))
        nn.init.xavier_uniform_(self.time_week)

        self.continuous_encoder = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(384)
        )


        self.padding_idx = self.road_cnt
        self.space_road = nn.Parameter(torch.empty(self.road_cnt+1, 384))  # （N, 256）
        nn.init.xavier_uniform_(self.space_road)

        self.kge_proj = nn.Linear(1024, 128)


        self.feature_fusion = nn.Conv2d(
            512 + 384, 768, kernel_size=(1, 1)
        )

