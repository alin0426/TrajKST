import torch
import pandas as pd
import networkx as nx
import json
from transformers import GPT2Tokenizer
from tqdm import tqdm
from collections import Counter
import numpy as np
from ast import literal_eval

"""
1. 轨迹数据，KG数据
2. 每条轨迹，构建KG子图
3. 子图中找path
需要mask
4. path文本化（sentence）
5. sentence为单位做滑动窗口
6. tokenizer -> token_ids
7. 缓存
"""


cached_data = pd.read_csv('./dataset/porto/traj_porto_test_mapper.csv',delimiter=';')
def pad_or_truncate(paths, max_len=64, pad_value=0):
    """截断或填充到固定长度"""
    result = []
    for p in paths:
        if len(p) >= max_len:
            result.append(p[:max_len])           # 截断
        else:
            result.append(p + [pad_value] * (max_len - len(p)))  # 填充
    return torch.tensor(result, dtype=torch.long)

# 使用
paths = cached_data['path'][178*128+114:178*128+115].apply(literal_eval).tolist()
traj_road_id_lists = pad_or_truncate(paths, max_len=64)


num_traj = len(traj_road_id_lists)  # 总轨迹条数


print("Start reading KG data file.")
kg_entity = pd.read_csv('./dataset/porto/kg_entity_porto.csv')  # 实体表

kg_relation = pd.read_csv('./dataset/porto/kg_relation_porto.csv')  # 关系表

kg_triple = pd.read_csv('./dataset/porto/kg_triple_porto.csv')  # 三元组表



def load_kg_nx(kg_entity, kg_relation, kg_triple):
    rel_map = kg_relation.set_index('id').to_dict('index')

    road2eid = {}

    G = nx.MultiDiGraph()
    for _, row in kg_entity.iterrows():
        eid = row['id']
        name = row['name']

        if row['type'] == 'road' and name.startswith('road_'):
            road_id = int(name.split('_')[1])  # "road_423122" -> 423122
            road2eid[road_id] = eid

        props = json.loads(row['properties']) if pd.notna(row['properties']) else {}
        # 去掉指定字段
        props.pop('geometry', None)
        props.pop('osm_id', None)
        props.pop('bridge', None)
        props.pop('oneway', None)
        G.add_node(row['id'],
                    name=row['name'],  # （road+osm_id）
                    type=row['type'],  # road
                    **props)


    for _, row in kg_triple.iterrows():
        h = int(row['head_id'])
        t = int(row['tail_id'])

        r_id = row['relation_id']
        r_name = rel_map[r_id]['name']
        r_desc = rel_map[r_id]['description']
        conf = row['confidence'] if pd.notna(row['confidence']) else 1.0

        # 正向
        G.add_edge(h, t,
                    relation_id=int(r_id),  # 关系id
                    relation_name=r_name,  # 关系名称
                    relation_desc=r_desc,  # 关系描述
                    confidence=conf,     # 边 置信度
                    direction="forward")

        G.add_edge(t, h,
                   relation_id=int(r_id),  # 关系id
                   relation_name=r_name+"_rev",  # 关系名称
                   relation_desc="REVERSE_OF:" + r_desc,  # 关系描述
                   confidence=conf,  # 边 置信度
                   direction="reverse")


    print(f"KG data loaded. nodes Count: {len(G.nodes())}, edges Count: {len(G.edges())}")

    return G, road2eid

# k-hop
def build_traj_subgraph(G, road_ids, k=2):
    nodes = set(road_ids)   # 已收集的节点
    frontier = set(road_ids)  # 当前要向外扩的“前沿”

    for _ in range(k):
        nxt = set()
        for n in frontier:
            nxt |= set(G.neighbors(n))  # 所有邻居
        frontier = nxt - nodes  # 去掉已经收集的
        nodes |= frontier    # 合并到集合

    return G.subgraph(nodes).copy()

def find_best_road_semantic_path(
    G,
    r1,
    r2,
    max_hop=5,
    max_paths=10
):
    """
    输入两个 Road 节点 r1, r2
    返回一条最优的 Road–Road 语义路径（list of node ids）
    """
    ALLOWED_TRANSITIONS = {
        "road": {"POI", "AOI"},
        "POI": {"road", "AOI", "POI", "Sub_Category"},
        "AOI": {"road", "POI", "AOI", "Sub_Category"},
        "Sub_Category": {"POI", "AOI", "Mid_Category"},
        "Mid_Category": {"Sub_Category", "Big_Category"},
        "Big_Category": {"Mid_Category"},
    }


    NODE_WEIGHT = {
        "POI": 1.0,
        "AOI": 0.8,
        "Sub_Category": 0.4,
        "Mid_Category": 0.3,
        "Big_Category": 0.2,
    }


    LENGTH_PENALTY = 0.5  # 长度惩罚系数,鼓励短路径

    #  路径枚举
    paths = []  # 所有合法路径

    # dfs(r1, [r1], {r1})，递归枚举
    def dfs(cur, path, visited):
        # path 存的是节点序列，长度=边数+1，超过就剪枝
        if len(path) > max_hop + 1:  # 长度已超过规定
            return

        if cur == r2 and len(path) > 2:
            paths.append(path.copy())
            return

        cur_type = G.nodes[cur].get("type")

        for nxt in G.neighbors(cur):
            if nxt in visited:
                continue

            nxt_type = G.nodes[nxt].get("type")


            if cur_type == "road" and nxt_type == "road":
                continue

            if cur == r1 and nxt == r2:
                continue

            if nxt_type not in ALLOWED_TRANSITIONS.get(cur_type, set()):
                continue

            # dfs回溯
            visited.add(nxt)
            path.append(nxt)
            dfs(nxt, path, visited)
            path.pop()
            visited.remove(nxt)

            if len(paths) >= max_paths:
                return

    dfs(r1, [r1], {r1})

    if not paths:
        return None


    # 路径打分
    def score_path(path):
        score = 0.0
        for n in path[1:-1]:
            ntype = G.nodes[n].get("type")
            score += NODE_WEIGHT.get(ntype, 0.0)

        score -= LENGTH_PENALTY * len(path)
        return score

    best_path = max(paths, key=score_path)
    return best_path

from collections import deque

ALLOWED_NEXT = {
        "road": {"POI", "AOI"},
        "POI": {"road", "AOI", "POI", "Sub_Category"},
        "AOI": {"road", "POI", "AOI", "Sub_Category"},
        "Sub_Category": {"POI", "AOI", "Mid_Category"},
        "Mid_Category": {"Sub_Category", "Big_Category"},
        "Big_Category": {"Mid_Category"},
    }
MAX_HOP = 5

def find_typed_bfs_path(G, r1, r2):
    """
    找一条 r1 -> r2 的语义路径
    类型受限 + hop 受限
    """
    queue = deque()
    queue.append((r1, [r1]))
    visited = set([r1])

    while queue:
        cur, path = queue.popleft()

        if len(path) > MAX_HOP + 1:
            continue

        if cur == r2 and len(path) > 2:
            return path

        cur_type = G.nodes[cur].get("type")

        for nxt in G.successors(cur):
            if nxt in visited:
                continue

            nxt_type = G.nodes[nxt].get("type")
            if nxt_type not in ALLOWED_NEXT.get(cur_type, set()):
                continue

            visited.add(nxt)
            queue.append((nxt, path + [nxt]))

    return None

def node_name(G, node_id):
    """
    根据节点类型，返回适合人类阅读的名字；
    若都没有，回退到 road_<id>
    """
    ntype = G.nodes[node_id].get("type")
    if ntype == "road":
        return G.nodes[node_id].get("road_name") or f"road_{node_id}"
    if ntype == "POI":
        return G.nodes[node_id].get("POI_name") or f"poi_{node_id}"
    if ntype == "AOI":
        return G.nodes[node_id].get("AOI_name") or f"aoi_{node_id}"
    if ntype == 'Sub_Category':
        return G.nodes[node_id].get("name")
    if ntype == 'Mid_Category':
        return G.nodes[node_id].get("name")
    if ntype == 'Big_Category':
        return G.nodes[node_id].get("name")

    return f"{ntype}_{node_id}"


def road_pair_to_token(G, r1, r2):

    u_name = node_name(G, r1)
    v_name = node_name(G, r2)
    return f"[ROAD] {u_name} -> {v_name}"


def kg_path_to_token(G, path):
    texts = []
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        edge_data = list(G.get_edge_data(u, v).values())[0]
        rel = edge_data["relation_name"]
        u_name = node_name(G, u)
        v_name = node_name(G, v)

        texts.append(f"{u_name} --{rel}--> {v_name}")
    return "[KG] " + " ; ".join(texts)




# 单条轨迹 -> text
def traj_to_sentences(G_full, road_seq, k):

    subG = build_traj_subgraph(G_full, road_seq, k=k)


    pairs = []
    k = 0
    sentences = []
    for i in range(len(road_seq) - 1):
        r1, r2 = road_seq[i], road_seq[i + 1]
        if r1 == r2:
            continue


        path = find_typed_bfs_path(subG, r1, r2)
        pairs.append({
            "r1": r1,
            "r2": r2,
            "path": path  # node_id list or None
        })


    traj_struct = {
        "road_seq": road_seq,
        "pairs": pairs
    }
    return traj_struct

G_full, road2eid = load_kg_nx(kg_entity, kg_relation, kg_triple)


text_cache = {}      # sentence/window -> token_ids
traj_token_cache = []  # 每条轨迹的 token windows
traj_struct_cache = []


for road_seq_tensor in tqdm(traj_road_id_lists, desc="Processing trajectories"):


    road_seq = road_seq_tensor.tolist()
    road_eids = [road2eid[r] for r in road_seq if r in road2eid]

    if len(road_eids) < 2:
        traj_struct_cache.append(None)
        continue

    traj_struct = traj_to_sentences(G_full, road_eids, k=2)
    print(f"traj_struct:{traj_struct}")

    traj_struct_cache.append(traj_struct)



torch.save(
    {
        "traj_struct_cache": traj_struct_cache
    },
    "./dataset/porto/test_cache/cached_traj_kg_path.pth"
)

#print('traj_struct_cache:',traj_struct_cache)

