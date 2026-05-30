import os
import torch
import torch.distributed as dist
import pandas as pd
from config.args_config import args

# 数据文件路径

device = torch.device("cuda:0" if args.use_gpu and torch.cuda.is_available() else "cpu")
device_id = dist.get_rank() if dist.is_initialized() else -1

city_root_path = os.path.join(args.dataset_path, args.city)

save_root_path = os.path.join(city_root_path, 'test_cache' if args.test_cache else '')
os.makedirs(save_root_path, exist_ok=True)

# road_relation_file = os.path.join(city_root_path, f"roadmap_{args.city}", f"roadmap_{args.city}.rel")
# road_relation_tensor_file = os.path.join(city_root_path, f'cached_{args.city}_relation.pth')
#
# road_static_file = os.path.join(city_root_path, f"roadmap_{args.city}", f"road_features_{args.city}.csv")
# road_static_tensor_file = os.path.join(city_root_path, f'cached_{args.city}_static.pth')
#
# road_dynamic_file = os.path.join(city_root_path, f'{args.city}.dyna')
# road_dynamic_tensor_file = os.path.join(city_root_path, f'cached_{args.city}_dynamic.pth')

# 增加知识图谱数据文件
kg_entity_file = os.path.join(city_root_path, f'kg_entity_{args.city}.csv')     # 实体表
kg_relation_file = os.path.join(city_root_path, f'kg_relation_{args.city}.csv')     # 关系表
kg_triple_file = os.path.join(city_root_path, f'kg_triple_{args.city}.csv')     # 三元组表
cached_kg_nodes = os.path.join(city_root_path, f'cached_kg_node_{args.city}.pt')
cached_kg_edges = os.path.join(city_root_path, f'cached_kg_edge_{args.city}.pt')
# 增加road_index文件
road_index_file = os.path.join(city_root_path, f'osmid_to_index_{args.city}.csv')
# 增加traj_text_token_ids文件
token_cache_dir = os.path.join(save_root_path, f'cached_traj_text_tokens')
# token_cache_dir_nouser = os.path.join(save_root_path, f'cached_traj_text_tokens_nouser')
token_cache_dir_des = os.path.join(save_root_path, f'cached_traj_text_tokens_des')
token_cache_dir_time = os.path.join(save_root_path, f'cached_traj_text_tokens_time')
token_chunk_size = 16000
# 相似轨迹的query,target的traj_text_token_ids文件
# token_cache_dir_query = os.path.join(save_root_path, f'cached_traj_text_tokens_query')
# token_cache_dir_target = os.path.join(save_root_path, f'cached_traj_text_tokens_target')

# 增加子图节点文件
cached_subgraph_nodes = os.path.join(save_root_path, f'subgraph_nodes.npz')
# cached_subgraph_nodes_query = os.path.join(save_root_path, f'subgraph_nodes_query.npz')
# cached_subgraph_nodes_target = os.path.join(save_root_path, f'subgraph_nodes_target.npz')
# 增加traj_kg_path文件
cached_traj_kg_path = os.path.join(save_root_path, f'cached_traj_kg_path.pth')
# cached_traj_kg_path_query = os.path.join(save_root_path, f'cached_traj_kg_path_query.pth')
# cached_traj_kg_path_target = os.path.join(save_root_path, f'cached_traj_kg_path_target.pth')


dataset_meta_file = os.path.join(city_root_path, f'cached_meta_{args.city}.json')

traj_file = os.path.join(city_root_path, f'traj_{args.city}_train_mapper.csv')
if args.test_cache:
    traj_file = os.path.join(city_root_path, f'traj_{args.city}_test_mapper.pkl')
traj_file_short = os.path.join(city_root_path, f'traj_{args.city}_train.csv')
cur_traj_file = traj_file_short if args.develop else traj_file

def generate_cached_file_name(dataset_name, is_short=False):
    suffix = "_short" if is_short else ""
    return os.path.join(save_root_path, f"cached_{dataset_name}_dataset{suffix}.pth")

datasets = ["traj","time_reg",  "traj_recover", "destination_prediction"]
cached_files = {dataset: generate_cached_file_name(dataset, args.develop) for dataset in datasets}

cached_traj_dataset = cached_files["traj"]      # cached_traj_dataset.pth
cached_time_reg_dataset = cached_files["time_reg"]
cached_traj_recover_dataset = cached_files["traj_recover"]
cached_des_prediction_dataset = cached_files["destination_prediction"]  # 增加目的地预测

# 增加相似轨迹搜索
# cached_traj_similar_query_dataset = os.path.join(city_root_path, f'cached_traj_similar_query_dataset.pth')
# cached_traj_similar_target_dataset = os.path.join(city_root_path, f'cached_traj_similar_target_dataset.pth')
# cached_traj_similar_test_dataset = os.path.join(city_root_path, f'cached_traj_similar_test_dataset.pth')
# cached_traj_similar_negindex_dataset = os.path.join(city_root_path, f'cached_traj_similar_negindex_dataset.npy')
#
# cached_traj_similar_evaluate = os.path.join(city_root_path, f'cached_traj_similar_evaluate.h5')
#
# road_dynamic_embedding_file = os.path.join(city_root_path, "road_dyna_embedding.npy")

# start_time = pd.to_datetime("2018-10-01T00:00:00Z")
# end_time = pd.to_datetime("2018-11-30T23:30:00Z")
# interval = 1800 # 30min的秒数
