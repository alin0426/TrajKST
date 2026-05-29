import pandas as pd
from config import global_vars
import networkx as nx
import json
import torch
import pickle
from tqdm import tqdm
from data_provider.retriver_sam import sampling_sub
import numpy as np
from ast import literal_eval

# 提前离线缓存轨迹数据中，每条轨迹对应的subgraph的节点id
print("Start reading KG data file.")
kg_entity = pd.read_csv('./dataset/porto/kg_entity_porto.csv')  # 实体表(63276), 1~63277

kg_relation = pd.read_csv('./dataset/porto/kg_relation_porto.csv')  # 关系表(12), 1~12

kg_triple = pd.read_csv('./dataset/porto/kg_triple_porto.csv')  # 三元组表(258820), 1~258820



# 把 relation 做成 id->row 映射，方便后面查
rel_map = kg_relation.set_index('id').to_dict('index')

# 新增反向映射： road_id -> entity.id
road2eid = {}  # 423122 -> 10086

G = nx.MultiDiGraph()
for _, row in kg_entity.iterrows():
    eid = row['id']
    name = row['name']
    # 只处理 road 类实体（构建road id -> entity_id 的映射）
    if row['type'] == 'road' and name.startswith('road_'):
        road_id = int(name.split('_')[1])  # "road_423122" -> 423122
        road2eid[road_id] = eid

    props = json.loads(row['properties']) if pd.notna(row['properties']) else {}
    props.pop('geometry', None)
    props.pop('osm_id', None)
    props.pop('bridge', None)
    props.pop('oneway', None)
    G.add_node(row['id'],
               name=row['name'],    # （road+osm_id）
               type=row['type'],    # road
               **props)


for _, row in kg_triple.iterrows():
    r_id = row['relation_id']
    r_name = rel_map[r_id]['name']
    r_desc = rel_map[r_id]['description']
    conf = row['confidence'] if pd.notna(row['confidence']) else 1.0

    G.add_edge(int(row['head_id']), int(row['tail_id']),
               relation_id=int(r_id),    # 关系id
               relation_name=r_name,    # 关系名称
               relation_desc=r_desc,    # 关系描述
               confidence=conf)     # 边 置信度




cached_data = torch.load('./dataset/porto/cached_traj_dataset.pth', weights_only=True)

traj_road_id_lists = cached_data['traj_road_id_lists']


def k_hop_from_road(road_ids, k=2):
    """road_ids: List[int]  例如 [423122, 423123]"""
    # 根据road_id, 得到对应entity_id
    seed_eids = [road2eid[r] for r in road_ids if r in road2eid]

    if not seed_eids:
        print(f"road_ids {road_ids} is empty")
        return nx.MultiDiGraph(), seed_eids  # 空图
    # k-hop
    nodes = set(seed_eids)  # 实体id

    for _ in range(k):
        nxt = set()

        for n in nodes:
            nxt.update(G.successors(n))  # 下游
            nxt.update(G.predecessors(n))  # 上游
        nodes.update(int(n) for n in nxt)


    return G.subgraph(nodes).copy(), set(seed_eids)


all_sub_nodes = []

for sid, traj_road_id_tensor in tqdm(enumerate(traj_road_id_lists), total=len(traj_road_id_lists)):
    traj_road_id_list = traj_road_id_tensor.tolist()
    # 1. k-hop
    sub, road_eids = k_hop_from_road(traj_road_id_list, k=2)

    if sub.number_of_nodes() == 0:
        all_sub_nodes.append([])
        continue

    # 2. PCST（只拿节点）
    keep_nodes = sampling_sub(sub, road_eids)
    keep_nodes = sorted(set(keep_nodes))
    all_sub_nodes.append(keep_nodes)


np.savez_compressed(
    "./dataset/porto/subgraph_nodes.npz",
    sub_nodes=np.array(
        [np.array(x, dtype=np.int32) for x in all_sub_nodes],
        dtype=object
    )
)


