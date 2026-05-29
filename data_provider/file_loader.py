import pandas as pd
import logging
import os
import json
import torch
import torch.distributed as dist

from config.args_config import args
from config import global_vars

import networkx as nx
from torch_geometric.data import HeteroData
from models.layers import Sentence_Transformer
from transformers import AutoModel, AutoTokenizer

from torch.utils.data import DataLoader
from tqdm import tqdm

from torch_geometric.data.data import Data
from torch_geometric.utils import k_hop_subgraph

from pcst_fast import pcst_fast

import numpy as np


NODE_TYPE2ID = {
    "road": 0,
    "AOI": 1,
    "POI": 2,
    "Sub_Category": 3,
    "Mid_Category": 4,
    "Big_Category": 5,
    "Unknown": 6,
}

REL_TYPE2ID = {
    "connect": 0,
    "neraby": 1,
    "borderBy": 2,
    "ODFlow": 3,
    "competitive": 4,
    "subCateOf": 5,
    "locateAt": 6,
    "belongto": 7,
    "intersect": 8,
    "cate1Of": 9,
    "cate2Of": 10,
    "cate3Of": 11,
}

ID2REL_TYPE = {v: k for k, v in REL_TYPE2ID.items()}


class FileLoader:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FileLoader, cls).__new__(cls)
            cls._instance.edges = None
            cls._instance.edge_weight = None
            cls._instance.edge_cnt = None

            cls._instance.static_features = None
            cls._instance.dynamic_features = None
            cls._instance.road_cnt = None
            cls._instance.time_slots_cnt = None

            cls._instance.traj_data = None
            cls._instance.traj_cnt = None
            cls._instance.traj_category_cnt = None

            cls._instance.G = None
            cls._instance.node_id2idx = None
            cls._instance.node_emb = None
            cls._instance.edge_key2idx = None
            cls._instance.edge_emb = None
            cls._instance.node_text_table = None
            cls._instance.edge_text_table = None


            cls._instance.road_index = None
            cls._instance.osmid_to_index = None
            cls._instance.traj_kg_path = None

        return cls._instance

    def load_road_relation_file(self):
        relation_cache = global_vars.road_relation_tensor_file
        if os.path.exists(relation_cache):
            logging.info(f"Loading cached edges and weights from {relation_cache}")
            cache = torch.load(relation_cache, weights_only=True)
            self.edges, self.edge_weight, self.edge_cnt = cache['edges'], cache['edge_weight'], cache['edge_cnt']
        else:
            logging.info("Reading adjacency file from csv.")
            rel_data = pd.read_csv(global_vars.road_relation_file)

            self.edge_cnt = len(rel_data)
            self.edges = torch.tensor(rel_data[['origin_id', 'destination_id']].to_numpy(dtype='int64'), dtype=torch.int64).T
            self.edge_weight = torch.tensor(rel_data['geographical_weight'].to_numpy(dtype='float32'), dtype=torch.float32)

            torch.save({"edges": self.edges, "edge_weight": self.edge_weight, "edge_cnt": self.edge_cnt}, relation_cache)
            logging.info(f"Saved adjacency tensor cache to {relation_cache}")
            
        logging.info(f"Edges loaded. Shape: {self.edges.shape}, Edge count: {self.edge_cnt}")

    def load_traj_dataset_file(self):
        logging.info("Start reading trajectory data file.")
        ext = os.path.splitext(global_vars.cur_traj_file)[1].lower()
        #print(f"文件类型：{ext}")
        if ext == '.csv':
            traj_data = pd.read_csv(global_vars.cur_traj_file, delimiter=';')  # 设置develop，读取train数据
        elif ext == '.pkl':
            traj_data = pd.read_pickle(global_vars.cur_traj_file)

        self.traj_data = traj_data  # 轨迹数据
        self.traj_cnt = len(traj_data)

        if os.path.exists(global_vars.dataset_meta_file):
            logging.info(f"读取meta数据：{global_vars.dataset_meta_file}" )
            with open(global_vars.dataset_meta_file, 'r') as f:
                meta = json.load(f)
                self.traj_category_cnt = meta["traj_category_cnt"]
        else:
            logging.info(f"读取traj_file：{global_vars.traj_file}")
            traj_data_full = pd.read_csv(global_vars.traj_file, delimiter=';')
            self.traj_category_cnt = len(set(traj_data_full["usr_id"]))

        logging.info(f"Trajectory data loaded. Count: {self.traj_cnt}, Categories: {self.traj_category_cnt}")

    def load_road_data_file(self):
        logging.info(f'start reading road data file from csv.')
        road_index = pd.read_csv(global_vars.road_index_file)
        osm_id_to_index = dict(zip(road_index['osm_id'], road_index['index']))
        self.road_index = road_index
        self.osmid_to_index = osm_id_to_index
        self.road_cnt = len(road_index)
        self.padding_idx = self.road_cnt
        logging.info(f'road cnt:{self.road_cnt}')

    def load_traj_kg_path(self):
        logging.info(f"加载轨迹点kg_path数据:{global_vars.cached_traj_kg_path}")
        data = torch.load(global_vars.cached_traj_kg_path, map_location="cpu")
        self.traj_kg_path = data["traj_struct_cache"]
        logging.info(f"traj_kg_path size:{len(self.traj_kg_path)}")

    def load_all(self, rank=0):
        if self.traj_data is None:
            self.load_traj_dataset_file()
        if self.road_index is None:
            self.load_road_data_file()
        
        if rank == 0:
            self.save_meta_info(global_vars.dataset_meta_file)
            
    
    def get_traj_kg_path(self):
        if self.traj_kg_path is None:
            raise RuntimeError("traj_kg_path not loaded. Please call load_all() first.")
        return self.traj_kg_path

    def get_edges(self):
        if self.edges is None:
            raise RuntimeError("edges not loaded. Please call load_all() first.")
        return self.edges

    def get_edge_weight(self):
        if self.edge_weight is None:
            raise RuntimeError("edge_weight not loaded.")
        return self.edge_weight

    # st-tokenizer
    def get_node_id2idx(self):
        if self.node_id2idx is None:
            raise RuntimeError("node_id2idx not loaded.")
        return self.node_id2idx

    # bert得到的embedding,st-t
    def get_node_emb(self):
        if self.node_emb is None:
            raise RuntimeError("node_emb not loaded.")
        return self.node_emb

    # st-t
    def get_edge_key2idx(self):
        if self.edge_key2idx is None:
            raise RuntimeError("edge_key2idx not loaded.")
        return self.edge_key2idx

    # st-t
    def get_rel_id2idx(self):
        if self.rel_id2idx is None:
            raise RuntimeError("rel_id2idx not loaded.")
        return self.rel_id2idx

    # bert得到的embedding, st-t
    def get_edge_emb(self):
        if self.edge_emb is None:
            raise RuntimeError("edge_emb not loaded.")
        return self.edge_emb


    def get_entity_emb(self):
        if self.entity_emb is None:
            raise RuntimeError("entity_emb not loaded.")
        return self.entity_emb

    def get_relation_emb(self):
        if self.relation_emb is None:
            raise RuntimeError("relation_emb not loaded.")
        return self.relation_emb

    def get_entity2id(self):
        if self.entity2id is None:
            raise RuntimeError("entity2id not loaded.")
        return self.entity2id

    def get_relation2id(self):
        if self.relation2id is None:
            raise RuntimeError("relation2id not loaded.")
        return self.relation2id

    # osm_id -> entity_id, st-t
    def get_road2eid(self):
        if self.road2eid is None:
            raise RuntimeError("road2eid not loaded.")
        return self.road2eid


    # st-t
    def get_osmid_to_index(self):
        if self.osmid_to_index is None:
            raise RuntimeError("osmid to index not loaded.")
        return self.osmid_to_index

    def get_padding_idx(self):
        if self.padding_idx is None:
            raise RuntimeError("padding_idx not loaded.")
        return self.padding_idx


    def get_traj_data(self):
        if self.traj_data is None:
            raise RuntimeError("traj_data not loaded.")
        return self.traj_data
    
    def get_edge_cnt(self):
        return self.edge_cnt
    
    def get_road_cnt(self):
        return self.road_cnt
    
    def get_time_slots_cnt(self):
        return self.time_slots_cnt
    
    def get_traj_cnt(self):
        return self.traj_cnt
    
    def get_traj_category_cnt(self):
        return self.traj_category_cnt

    def get_meta_info(self):
        return {
            "road_cnt": self.road_cnt,
            "traj_cnt": self.traj_cnt,
            "traj_category_cnt": self.traj_category_cnt,
        }
        
    def save_meta_info(self, meta_file_path):
        current_meta_info = self.get_meta_info()

        if os.path.exists(meta_file_path):
            with open(meta_file_path, 'r') as f:
                existing_meta_info = json.load(f)
            
            if existing_meta_info == current_meta_info:
                logging.info(f"Meta info is up-to-date. No changes made to {meta_file_path}")
                return 
            else:
                logging.info(f"Meta info has changed. Updating {meta_file_path}")
                
        with open(meta_file_path, 'w') as f:
            json.dump(current_meta_info, f, indent=4)
        logging.info(f"Meta info saved to {meta_file_path}")



    def load_kg_nx(self):
        logging.info("Start reading KG data file.")
        kg_entity = pd.read_csv(global_vars.kg_entity_file)
        logging.info(f"kg_entity len:{len(kg_entity)}")

        kg_relation = pd.read_csv(global_vars.kg_relation_file)
        logging.info(f"kg_relation len:{len(kg_relation)}")

        kg_triple = pd.read_csv(global_vars.kg_triple_file)
        logging.info(f"kg_triple len:{len(kg_triple)}")

        rel_map = kg_relation.set_index('id').to_dict('index')

        # road_id -> entity.id
        self.road2eid = {}
        node_text_table = {}

        self.G = nx.MultiDiGraph()

        for _, row in kg_entity.iterrows():
            eid = row['id']
            name = row['name']
            # 只处理 road 类实体（构建road id -> entity_id 的映射）
            if row['type'] == 'road' and name.startswith('road_'):
                road_id = int(name.split('_')[1])
                self.road2eid[road_id] = eid
            props = json.loads(row['properties']) if pd.notna(row['properties']) else {}

            props.pop('geometry', None)
            props.pop('osm_id', None)
            props.pop('bridge', None)
            props.pop('oneway', None)
            self.G.add_node(row['id'],
                       name=row['name'],    # （road+osm_id）
                       type=row['type'],    # road
                       **props)

        edge_text_table = []  # 全局边desc
        for _, row in kg_triple.iterrows():
            r_id = row['relation_id']
            r_name = rel_map[r_id]['name']
            r_desc = rel_map[r_id]['description']
            conf = row['confidence'] if pd.notna(row['confidence']) else 1.0

            self.G.add_edge(int(row['head_id']), int(row['tail_id']),
                       relation_id=int(r_id),
                       relation_name=r_name,
                       relation_desc=r_desc,
                       confidence=conf)
            #edge_text_table.append(r_name)  =

        logging.info(f"KG data loaded. nodes Count: {len(self.G.nodes())}, edges Count: {len(self.G.edges())}")

        logging.info("构建节点和边的emb")
        if not os.path.exists(global_vars.cached_kg_nodes):
            logging.info(f"不存在相关文件：{global_vars.cached_kg_nodes}")
            self.build_cache_from_nx(self.G)
        else:
            logging.info(f"存在相关文件：{global_vars.cached_kg_nodes}")

        # 加载节点表
        logging.info(f"加载节点embed文件:{global_vars.cached_kg_nodes}")
        node_cache = torch.load(global_vars.cached_kg_nodes)
        self.node_id2idx = node_cache['id2idx']  # dict{node_id: idx}
        self.node_emb = node_cache['emb']
        logging.info(f'node_embed shape:{self.node_emb.shape}')

        # 加载边表
        logging.info(f"加载边embed文件:{global_vars.cached_kg_edges}")
        edge_cache = torch.load(global_vars.cached_kg_edges)
        self.edge_key2idx = edge_cache['key2idx']  # dict{edge_attr_str: idx}
        self.edge_emb = edge_cache['emb']
        logging.info(f'edge_embed shape:{self.edge_emb.shape}')
        self.rel_id2idx = {}

        for key, idx in self.edge_key2idx.items():
            rid = int(key.split('relation_id:')[1].split(';')[0])
            self.rel_id2idx[rid] = idx



    def text2embedding(self, text):
        from .data_loader import DatasetKG
        # print("text的类型：", type(text))
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        pretrained_repo = './models/all-roberta-large-v1'
        if len(text) == 0:
            return torch.zeros((0, 1024))   # bert输出维度1024

        print("加载tokenizer:")
        text_tokenizer = AutoTokenizer.from_pretrained(pretrained_repo)
        encoding = text_tokenizer(text, padding=True, truncation=True, return_tensors='pt')
        print("加载dataset:")
        dataset = DatasetKG(input_ids=encoding.input_ids, attention_mask=encoding.attention_mask)

        # DataLoader
        dataloader = DataLoader(dataset, batch_size=1024, shuffle=False)

        # bert
        print("加载模型：")
        model = Sentence_Transformer(pretrained_repo)
        model.to(device)
        model.eval()

        # Placeholder for storing the embeddings
        all_embeddings = []

        # Iterate through batches
        with torch.no_grad():

            for batch in tqdm(dataloader, desc="BERT encoding"):
                # Move batch to the appropriate device
                batch = {key: value.to(device) for key, value in batch.items()}

                # Forward pass
                embeddings = model(input_ids=batch["input_ids"], att_mask=batch["att_mask"])

                # Append the embeddings to the list
                all_embeddings.append(embeddings)


        # Concatenate the embeddings from all batches
        all_embeddings = torch.cat(all_embeddings, dim=0).cpu()

        return all_embeddings


    def build_cache_from_nx(self, G: nx.MultiDiGraph, save_prefix='cache'):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        node_order = sorted(G.nodes(), key=int)  # 假设 id 可转 int
        node_id2idx = {nid: i for i, nid in enumerate(node_order)}


        nodes = []
        node_attr_list = []
        for nid in node_order:
            attr = G.nodes[nid]
            pieces = [f"{k}:{v}" for k, v in attr.items()]
            node_attr_str = "; ".join(pieces)
            node_attr_list.append(node_attr_str)

        edges = []
        edge_attr_list = []
        head_list = []
        tail_list = []
        for u, v, k, d in G.edges(keys=True, data=True):
            head_id = u
            tail_id = v
            r_id = d['relation_id']
            r_name = d['relation_name']
            r_desc = d['relation_desc']
            conf = d['confidence']
            edge_attr = 'relation_id:' + str(
                r_id) + '; relation_name:' + r_name + '; relation_des:' + r_desc + ';'

            edge_attr_list.append(edge_attr)

        edge_attr_set = sorted(set(edge_attr_list))

        key2idx = {key: i for i, key in enumerate(edge_attr_set)}

        node_emb = self.text2embedding(node_attr_list)

        edge_emb = self.text2embedding(edge_attr_set)

        torch.save({'key2idx': key2idx, 'emb': edge_emb}, global_vars.cached_kg_edges)
        print(f"保存图的边embedding：{global_vars.cached_kg_edges}")
        torch.save({'id2idx': node_id2idx, 'emb': node_emb}, global_vars.cached_kg_nodes)
        print(f"保存图的节点embedding：{global_vars.cached_kg_nodes}")

        print('KG 全局文本缓存完成，节点:', len(node_attr_list), '边:', len(edge_attr_set))


    def node_name(self, node_id, mask_roads):
        if node_id in mask_roads:
            return "[MASK]"

        ntype = self.G.nodes[node_id].get("type")
        if ntype == "road":
            index = int(node_id) - 1
            return self.G.nodes[node_id].get("road_name") or f"road_{index}"
        if ntype == "POI":
            return self.G.nodes[node_id].get("POI_name") or f"poi_{node_id}"
        if ntype == "AOI":
            return self.G.nodes[node_id].get("AOI_name") or f"aoi_{node_id}"
        if ntype == 'Sub_Category':
            return self.G.nodes[node_id].get("name")
        if ntype == 'Mid_Category':
            return self.G.nodes[node_id].get("name")
        if ntype == 'Big_Category':
            return self.G.nodes[node_id].get("name")
        return f"{ntype}_{node_id}"

    def pair_to_text(self, pair, mask_roads):
        if pair["path"] is not None:
            texts = []
            path = pair["path"]
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                edge_raw = self.G.get_edge_data(u, v)
                if edge_raw is None:  # 没有这条边
                    edge_raw = self.G.get_edge_data(v, u)
                edge_data = list(edge_raw.values())[0]
                rel = edge_data["relation_name"]
                texts.append(
                    f"{self.node_name(u, mask_roads)}|{rel}|{self.node_name(v, mask_roads)}"
                )
            return "[KG] " + ";".join(texts)
        else:
            u = "[MASK]" if pair["r1"] in mask_roads else self.node_name(pair["r1"], set())
            v = "[MASK]" if pair["r2"] in mask_roads else self.node_name(pair["r2"], set())
            return f"[ROAD] {u}|connect|{v};"

    def sentence_sliding_window(self, sentences, window_size=5, stride=3):
        """
        输出：
            windows: List[str]
                每个元素是一个 window 的文本（sentence 拼接）
        """
        windows = []

        n = len(sentences)
        if n == 0:
            return windows

        i = 0
        while i < n:
            chunk = sentences[i:i + window_size]

            if len(chunk) == 0:
                break

            windows.append(" ".join(chunk))

            i += stride

        return windows


class TokenShardReader:
    """
    根据chunk_id找到文件，load到CPU，做shard-level cache

    """

    def __init__(self, token_dir, chunk_size, total_len):
        self.token_dir = token_dir
        self.chunk_size = chunk_size
        self.total_len = total_len

        self._cur_chunk_id = None
        self._cur_tokens = None
        self._cur_chunk_start = None

    def get(self, idx):
        if idx < 0 or idx >= self.total_len:
            raise IndexError(idx)

        chunk_id = idx // self.chunk_size
        chunk_start = chunk_id * self.chunk_size
        offset = idx - chunk_start

        # 同一个 batch 中，只要 index 在同一个 chunk，token 文件只 load 一次
        if chunk_id != self._cur_chunk_id:  # chunk 内缓存
            chunk_end = min(chunk_start + self.chunk_size, self.total_len)
            path = os.path.join(self.token_dir, f"traj_tokens_{chunk_start}_{chunk_end}.pt")
            #print(f"读取cached_token_ids文件：{path}")

            self._cur_tokens = torch.load(path, map_location="cpu")
            #print(f"读取的文件大小：{self._cur_tokens.shape}")
            self._cur_chunk_id = chunk_id
            self._cur_chunk_start = chunk_start

            # 强校验
            assert len(self._cur_tokens) == (chunk_end - chunk_start)

        return self._cur_tokens[offset]  # [64, L]


file_loader = FileLoader()
