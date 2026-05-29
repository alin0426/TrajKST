from pcst_fast import pcst_fast
from collections import deque

import numpy as np


NODE_WEIGHT = {
    "road": 5.0,  # road靠连接作用被留下
    "AOI": 8.0,
    "POI": 10.0,
    "Sub_Category": 6.0,
    "Mid_Category": 5.0,
    "Big_Category": 4.0,
}

REL_WEIGHT = {
    "connect": 0.9,
    "nearby": 1.2,
    "borderBy": 0.8,
    "ODFlow": 0.9,
    "competitive": 0.8,
    "subCateOf": 0.5,
    "locateAt": 1.5,
    "belongto": 1.3,
    "intersect": 0.8,
    "cate1Of": 0.5,
    "cate2Of": 0.6,
    "cate3Of": 0.7,
}


def compute_hop_to_roads(G_sub, road_eids):
    hop = {n: float("inf") for n in G_sub.nodes}
    q = deque()

    for r in road_eids:
        if r in G_sub:
            hop[r] = 0
            q.append(r)

    while q:
        cur = q.popleft()

        next_nodes = set()
        next_nodes.update(G_sub.successors(cur))
        next_nodes.update(G_sub.predecessors(cur))

        for nxt in next_nodes:
            if hop[nxt] > hop[cur] + 1:
                hop[nxt] = hop[cur] + 1
                q.append(nxt)

    return hop

# 计算 节点 prize
def assign_node_prize(G_sub, road_eids):
    hop_dict = compute_hop_to_roads(G_sub, road_eids)

    TYPE_DECAY = {
        "road": 0.75,
        "AOI": 0.9,
        "POI": 0.95,
        "Sub_Category": 0.9,
        "Mid_Category": 0.85,
        "Big_Category": 0.8,
    }

    prize = {}
    for n in G_sub.nodes:
        ntype = G_sub.nodes[n].get("type", "Unknown")     # 节点类型
        base = NODE_WEIGHT.get(ntype, 5.0)

        decay = TYPE_DECAY.get(ntype, 0.8)
        prize[n] = base * (decay ** hop_dict[n])



    # 强制保留 Road anchor
    for r in road_eids:
        prize[r] = 50.0

    return prize

# 计算边 代价
def assign_edge_cost(G_sub):
    edge_cost = {}

    for u, v, k, data in G_sub.edges(keys=True, data=True):
        rel = data.get("relation_name", "Unknown")  # 边类型
        conf = data.get("confidence", 1.0)
        w = REL_WEIGHT.get(rel, 0.5)

        alpha = 2
        cost = (1.0 / w) * (1.0 / (1.0 + alpha * (conf -1)))

        edge_cost[(u, v, k)] = cost

    return edge_cost



def rooted_pcst(G, node_prize, edge_cost, root_nodes):
    """
    G: nx.MultiDiGraph
    node_prize: dict[eid -> float]
    edge_cost: dict[(u, v, k) -> float]
    root_nodes: List[eid]
    """

    # 引入虚拟节点
    SUPER_ROOT_EID = -1_000_000  # 保证不与真实 eid 冲突

    nodes = sorted(G.nodes)
    if SUPER_ROOT_EID not in nodes:
        nodes.append(SUPER_ROOT_EID)

    node2idx = {n: i for i, n in enumerate(nodes)}  # node_id -> index
    idx2node = {i: n for n, i in node2idx.items()}

    G_SCALE = 7
    prizes = np.zeros(len(nodes), dtype=np.float64)
    for n, idx in node2idx.items():
        if n == SUPER_ROOT_EID:
            prizes[idx] = 0.0
        else:
            prizes[idx] = node_prize.get(n, 0.0)/ G_SCALE


    edges = []
    costs = []


    for u, v, k in G.edges(keys=True):
        edges.append((node2idx[u], node2idx[v]))  # 对边中的端点换成index
        edges.append((node2idx[v], node2idx[u]))
        costs.append(edge_cost[(u, v, k)])
        costs.append(edge_cost[(u, v, k)])


    for r in root_nodes:
        if r in node2idx:
            u = node2idx[SUPER_ROOT_EID]
            v = node2idx[r]
            edges.append((u, v))
            costs.append(0.0)   # 核心：0 cost


    root_idx = node2idx[SUPER_ROOT_EID]


    edges_np = np.array(edges,   dtype=np.int32)      # shape (E, 2)

    prizes_np = np.array(prizes,  dtype=np.float64)    # shape (V,)

    costs_np = np.array(costs,   dtype=np.float64)    # shape (E,)

    num_v = len(prizes)                            # 总节点数



    # 3. run PCST，返回选中的节点和边
    selected_nodes, selected_edges = pcst_fast(
        edges_np,  # 重编号后的边列表
        prizes_np,
        costs_np,
        root_idx,
        1,
        "strong",
        0,
    )


    # 4. build subgraph
    keep_nodes = {idx2node[i] for i in selected_nodes}  # 还原节点id
    keep_nodes.discard(SUPER_ROOT_EID)

    return keep_nodes



# 子图采样
def sampling_sub(G_sub, road_eids):

    node_prize = assign_node_prize(G_sub, road_eids)

    edge_cost = assign_edge_cost(G_sub)

    final_nodes = rooted_pcst(G_sub, node_prize, edge_cost, road_eids)
    return final_nodes