from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch
import os
import logging
from ast import literal_eval

from config import global_vars
from config import random_seed
from config.args_config import args
from . import file_loader
from utils.timefeatures import time_features
from data_provider.file_loader import file_loader, TokenShardReader

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import euclidean_distances

from tqdm import tqdm

import ast


class DatasetTraj(Dataset):
    def __init__(self):
        logging.info("Start loading trajectory dataset.")
        
        super().__init__()
        
        self.dataset_len = file_loader.get_traj_cnt()
        self.traj_road_id_lists = None
        self.traj_time_stamp_lists = None
        self.traj_time_features_lists = None
        self.traj_road_index_lists = None
        self.traj_distance_lists = None
        self.traj_taxi_id = None
        self.sub_nodes_list = None
        self.traj_kg_path = None
        self.continuous_features = None

        self.token_reader = TokenShardReader(
            token_dir=global_vars.token_cache_dir,
            chunk_size=global_vars.token_chunk_size,
            total_len=self.dataset_len  # 总轨迹数据条数
        )

        
        if os.path.exists(global_vars.cached_traj_dataset):
            self.load_cached_traj_data()
        else:
            self.read_traj_data()
            self.cache_traj_data()

        self.load_subgraph_nodes()
        self.load_traj_kg_path()

        logging.info(
            f"(DatasetTraj) Number of trajectories: {self.dataset_len}\n"
            f"Shape of traj_road_id_lists: {self.traj_road_id_lists.shape}, \n"
            f"Shape of traj_time_stamp_lists: {self.traj_time_stamp_lists.shape}, \n"
            f"Shape of traj_time_features_lists: {self.traj_time_features_lists.shape}, \n"
            f"shape of traj_road_index_lists: {self.traj_road_index_lists.shape}, \n")
        
        logging.info("Finish loading trajectory dataset.")
    
    def read_traj_data(self):
        
        traj_data = file_loader.get_traj_data()
        osm2idx = file_loader.get_osmid_to_index()
        padding_idx = file_loader.get_padding_idx()

        self.traj_road_id_lists = torch.tensor([ x[:args.seq_len] if len(x) > args.seq_len else x + [x[-1]] * (args.seq_len - len(x)) 
                            for x in traj_data["path"].apply(lambda x: literal_eval(x))], dtype=torch.int64)
        logging.info("(DatasetTraj) Finish reading road id.")

        # osm_id -> index_id
        traj_osm = self.traj_road_id_lists.numpy()  # (N, seq_len)

        traj_index = [
            [osm2idx.get(int(o), padding_idx) for o in row]
            for row in traj_osm
        ]

        self.traj_road_index_lists = torch.tensor(
            traj_index, dtype=torch.int64
        )
        logging.info("(DatasetTraj) Finish reading road index.")

        self.traj_time_stamp_lists = torch.tensor(
            [x[:args.seq_len] if len(x) > args.seq_len else x + [x[-1]] * (args.seq_len - len(x))
             for x in traj_data["tlist"].apply(lambda x: literal_eval(x))], dtype=torch.int64)
        logging.info("(DatasetTraj) Finish reading time id.")

        data_stamp = time_features(pd.to_datetime(self.traj_time_stamp_lists.flatten(), unit='s'), freq='s').transpose(1, 0)
        data_stamp = torch.from_numpy(data_stamp)
        self.traj_time_features_lists = torch.reshape(data_stamp, (self.dataset_len, args.seq_len, data_stamp.shape[-1]))
        logging.info("(DatasetTraj) Finish reading time features.")

        self.traj_taxi_id = torch.tensor(traj_data["usr_id"], dtype=torch.int64)
        logging.info("(DatasetTraj) Finish reading taxi id.")

        self.traj_distance_lists = torch.tensor(
            [x[:args.seq_len] if len(x) > args.seq_len else x + [0] * (args.seq_len - len(x))
             for x in traj_data["dist_list"].apply(lambda x: literal_eval(x))], dtype=torch.int64)
        logging.info("(DatasetTraj) Finish reading traj distance.")


        # speed_values = [
        #     x[:args.seq_len] if len(x) > args.seq_len else x + [0] * (args.seq_len - len(x))
        #     for x in traj_data["speed_list"].apply(lambda x: literal_eval(str(x)))
        # ]
        #
        # acc_values = [
        #     x[:args.seq_len] if len(x) > args.seq_len else x + [0] * (args.seq_len - len(x))
        #     for x in traj_data["acc_list"].apply(lambda x: literal_eval(str(x)))
        # ]
        #
        # # 转为 Float Tensor
        # self.raw_speed = torch.tensor(speed_values, dtype=torch.float32)
        # self.raw_acc = torch.tensor(acc_values, dtype=torch.float32)
        #
        # # 2.2 归一化 (Normalization) - 非常关键！
        # # 神经网络对数值范围敏感，必须归一化到 N(0, 1) 或 [-1, 1]
        #
        # # 计算全局均值和标准差 (避免除以0，std加个极小值)
        # speed_mean, speed_std = self.raw_speed.mean(), self.raw_speed.std()
        # acc_mean, acc_std = self.raw_acc.mean(), self.raw_acc.std()
        #
        # # 执行 Z-Score 归一化
        # self.norm_speed = (self.raw_speed - speed_mean) / (speed_std + 1e-5)
        # self.norm_acc = (self.raw_acc - acc_mean) / (acc_std + 1e-5)
        #
        # # 2.3 堆叠成矩阵 [N, seq_len, 2] -> (Speed, Acc)
        # # 如果你有经纬度(lat/lon)或方向(bearing)，也可以在这里加进去，变成 [N, seq_len, 5]
        # self.continuous_features = torch.stack([self.norm_speed, self.norm_acc], dim=-1)
        #
        # logging.info("(DatasetTrajClassify) Finish reading continuous features.")

    def cache_traj_data(self):
        logging.info(f"Caching trajectory dataset to {global_vars.cached_traj_dataset}")
        torch.save({
            'traj_road_id_lists': self.traj_road_id_lists,
            'traj_time_stamp_lists': self.traj_time_stamp_lists,
            'traj_time_features_lists': self.traj_time_features_lists,
            'traj_road_index_lists': self.traj_road_index_lists,
            'traj_taxi_id': self.traj_taxi_id,
            'traj_distance_lists': self.traj_distance_lists,
            #'continuous_features': self.continuous_features,
        }, global_vars.cached_traj_dataset)
        
    def load_cached_traj_data(self):
        logging.info(f"Loading cached trajectory dataset from {global_vars.cached_traj_dataset}")
        cached_data = torch.load(global_vars.cached_traj_dataset, weights_only=True)
        self.traj_road_id_lists = cached_data['traj_road_id_lists']
        self.traj_time_stamp_lists = cached_data['traj_time_stamp_lists']
        self.traj_time_features_lists = cached_data['traj_time_features_lists']
        self.traj_road_index_lists = cached_data['traj_road_index_lists']
        #self.continuous_features = cached_data['continuous_features']
        #self.traj_taxi_id = cached_data['traj_taxi_id']
        #self.traj_distance_lists = cached_data['traj_distance_lists']
        traj_data = file_loader.get_traj_data()
        self.traj_len = torch.tensor(traj_data["hop"], dtype=torch.int64)

    def load_subgraph_nodes(self):
        logging.info(f"加载子图数据文件:{global_vars.cached_subgraph_nodes}")
        data = np.load(global_vars.cached_subgraph_nodes, allow_pickle=True)
        self.sub_nodes_list = data["sub_nodes"]
        logging.info(f"sub_nodes_list size:{len(self.sub_nodes_list)}")

    def load_traj_kg_path(self):
        logging.info(f"加载轨迹点kg_path数据:{global_vars.cached_traj_kg_path}")
        data = torch.load(global_vars.cached_traj_kg_path, map_location="cpu")
        self.traj_kg_path = data["traj_struct_cache"]
        logging.info(f"traj_kg_path size:{len(self.traj_kg_path)}")

        
    def __getitem__(self, index):
        traj_token_ids = self.token_reader.get(index)

        return (self.traj_road_id_lists[index],
                self.traj_time_stamp_lists[index],
                self.traj_road_index_lists[index],
                self.traj_time_features_lists[index],
                traj_token_ids,
                self.sub_nodes_list[index],
                self.traj_kg_path[index],
                # self.continuous_features[index],
                #self.traj_len[index],
                )
            
    def __len__(self):
        return self.dataset_len


class DatasetDestinationPrediction(Dataset):
    def __init__(self):
        logging.info("Start loading destination prediction dataset.")

        super().__init__()

        self.dataset_len = file_loader.get_traj_cnt()
        self.traj_road_id_lists = None
        self.traj_time_stamp_lists = None
        self.traj_road_index_lists = None
        self.destination_labels = None
        self.sub_nodes_list = None
        self.traj_kg_path = None
        self.continuous_features = None


        if os.path.exists(global_vars.cached_des_prediction_dataset):
            self.load_cached_traj_data()
        else:
            self.read_traj_data()
            self.cache_traj_data()

        self.token_reader = TokenShardReader(
            token_dir=global_vars.token_cache_dir_des,
            chunk_size=global_vars.token_chunk_size,
            total_len=self.dataset_len  # 总轨迹数据条数
        )

        self.load_subgraph_nodes()
        self.load_traj_kg_path()

        logging.info(
            f"(DatasetDestinationPrediction) Number of trajectories: {self.dataset_len}\n"
            f"Shape of traj_road_id_lists: {self.traj_road_id_lists.shape}, \n"
            f"Shape of traj_time_stamp_lists: {self.traj_time_stamp_lists.shape}, \n"
            f"shape of traj_road_index_lists: {self.traj_road_index_lists.shape}, \n"
            f"Shape of destination_labels: {self.destination_labels.shape}, \n"
            #f"Shape of continuous_features: {self.continuous_features.shape}, \n"
        )

        logging.info("Finish loading destination prediction dataset.")


    def read_traj_data(self):
        traj_data = file_loader.get_traj_data()
        osm2idx = file_loader.get_osmid_to_index()
        padding_idx = file_loader.get_padding_idx()

        def trim_tail_points(road, time, dis, tail_len=5):
            keep_len = max(len(road) - tail_len, 1)
            return (
                road[:keep_len],
                time[:keep_len],
                dis[:keep_len],
                # speed[:keep_len],
                # acc[:keep_len],
            )
        processed_road = []
        processed_time = []
        processed_dis = []


        for r, t, d in zip(
                traj_data["path"],
                traj_data["tlist"],
                traj_data["dist_list"],
                # traj_data["speed_list"],
                # traj_data["acc_list"]
        ):
            road = literal_eval(r)
            time = literal_eval(t)
            dis = literal_eval(d)
            # speed = literal_eval(s)
            # acc = literal_eval(a)

            # r_new, t_new, d_new, s_new, a_new = trim_tail_points(
            #          road, time, dis, speed, acc, tail_len=5
            # )
            r_new, t_new, d_new = trim_tail_points(
                road, time, dis, tail_len=5
            )

            processed_road.append(r_new)
            processed_time.append(t_new)
            processed_dis.append(d_new)
            # processed_speed.append(s_new)
            # processed_acc.append(a_new)


        total_cnt = len(processed_road)
        single_road_cnt = sum(1 for x in processed_road if len(x) < 2)

        self.traj_road_id_lists = torch.tensor(
            [
                (
                    x[:args.seq_len]
                    if len(x) > args.seq_len
                    else x + [x[-1]] * (args.seq_len - len(x))
                )
                if len(x) >= 2
                else
                (
                    [x[0]] * args.seq_len
                )
                for x in processed_road
            ],
            dtype=torch.int64
        )

        logging.info("(DatasetDestinationPrediction) Finish reading road id.")

        #osm_id -> index_id
        traj_osm = self.traj_road_id_lists.numpy()  # (N, seq_len)

        traj_index = [
            [osm2idx.get(int(o), padding_idx) for o in row]
            for row in traj_osm
        ]

        self.traj_road_index_lists = torch.tensor(
            traj_index, dtype=torch.int64
        )
        logging.info("(DatasetDestinationPrediction) Finish reading road index.")

        self.traj_time_stamp_lists = torch.tensor(
            [
                (
                    x[:args.seq_len]
                    if len(x) > args.seq_len
                    else x + [x[-1]] * (args.seq_len - len(x))
                )
                if len(x) >= 2
                else
                (
                        [x[0]] * args.seq_len
                )
                for x in processed_time
            ],
            dtype=torch.int64
        )

        logging.info("(DatasetDestinationPrediction) Finish reading time id.")

        self.destination_labels = torch.tensor(
            [osm2idx.get(int(x[-1]), padding_idx)
             for x in traj_data["path"].apply(lambda x: literal_eval(x))],
            dtype=torch.int64
        )

        logging.info("(DatasetDestinationPrediction) Finish reading Destination Prediction label.")

        self.traj_taxi_id = torch.tensor(traj_data["usr_id"], dtype=torch.int64)
        logging.info("(DatasetTraj) Finish reading taxi id.")

        self.traj_distance_lists = torch.tensor(
            [
                (
                    x[:args.seq_len]
                    if len(x) > args.seq_len
                    else x + [0] * (args.seq_len - len(x))
                )
                if len(x) >= 2
                else
                (
                        [0] * args.seq_len
                )
                for x in processed_dis
            ],
            dtype=torch.int64
        )
        logging.info("(DatasetTraj) Finish reading traj distance.")

        # speed_padded = [
        #     (x[:args.seq_len] if len(x) > args.seq_len else x + [0.0] * (args.seq_len - len(x)))
        #     if len(x) >= 1 else [0.0] * args.seq_len
        #     for x in processed_speed
        # ]
        #
        # acc_padded = [
        #     (x[:args.seq_len] if len(x) > args.seq_len else x + [0.0] * (args.seq_len - len(x)))
        #     if len(x) >= 1 else [0.0] * args.seq_len
        #     for x in processed_acc
        # ]
        #
        # # 2. 转 Tensor
        # raw_speed = torch.tensor(speed_padded, dtype=torch.float32)
        # raw_acc = torch.tensor(acc_padded, dtype=torch.float32)
        #
        # # 3. 归一化
        # # 为了保证和分类任务的数据分布一致，建议使用相同的均值/方差
        # # 如果不知道，就用当前的重新算
        # self.norm_speed = (raw_speed - raw_speed.mean()) / (raw_speed.std() + 1e-5)
        # self.norm_acc = (raw_acc - raw_acc.mean()) / (raw_acc.std() + 1e-5)
        #
        # # 4. Stack
        # self.continuous_features = torch.stack([self.norm_speed, self.norm_acc], dim=-1)
        #
        # logging.info("(DatasetDestinationPrediction) Finish reading continuous features.")

    def cache_traj_data(self):
        logging.info(f"Caching destination prediction dataset to {global_vars.cached_des_prediction_dataset}")
        torch.save({
            'traj_road_id_lists': self.traj_road_id_lists,
            'traj_time_stamp_lists': self.traj_time_stamp_lists,
            'traj_road_index_lists': self.traj_road_index_lists,
            'destination_labels': self.destination_labels,
            'traj_distance_lists': self.traj_distance_lists,
            'traj_taxi_id': self.traj_taxi_id,
            #'continuous_features': self.continuous_features,
        }, global_vars.cached_des_prediction_dataset)

    def load_cached_traj_data(self):
        logging.info(f"Loading cached destination prediction dataset from {global_vars.cached_des_prediction_dataset}")
        cached_data = torch.load(global_vars.cached_des_prediction_dataset, weights_only=True)
        self.traj_road_id_lists = cached_data['traj_road_id_lists']
        self.traj_time_stamp_lists = cached_data['traj_time_stamp_lists']
        self.traj_road_index_lists = cached_data['traj_road_index_lists']
        self.destination_labels = cached_data['destination_labels']
        #self.continuous_features = cached_data['continuous_features']

    def load_subgraph_nodes(self):
        logging.info(f"加载子图数据文件:{global_vars.cached_subgraph_nodes}")
        data = np.load(global_vars.cached_subgraph_nodes, allow_pickle=True)
        self.sub_nodes_list = data["sub_nodes"]
        logging.info(f"sub_nodes_list size:{len(self.sub_nodes_list)}")

    def load_traj_kg_path(self):
        logging.info(f"加载轨迹点kg_path数据:{global_vars.cached_traj_kg_path}")
        data = torch.load(global_vars.cached_traj_kg_path, map_location="cpu")
        self.traj_kg_path = data["traj_struct_cache"]
        logging.info(f"traj_kg_path size:{len(self.traj_kg_path)}")


    def __getitem__(self, index):
        traj_token_ids = self.token_reader.get(index)

        return (self.traj_road_id_lists[index],
                self.traj_time_stamp_lists[index],
                self.traj_road_index_lists[index],
                traj_token_ids,
                self.sub_nodes_list[index],
                self.traj_kg_path[index],
                #self.continuous_features[index],
                self.destination_labels[index])

    def __len__(self):
        return self.dataset_len


class DatasetTTE(Dataset):
    def __init__(self):
        logging.info("Start loading TTE dataset.")

        super().__init__()

        self.dataset_len = file_loader.get_traj_cnt()
        self.traj_road_id_lists = None
        self.traj_time_stamp_lists = None
        self.traj_time_features_lists = None
        self.traj_road_index_lists = None
        self.traj_distance_lists = None
        self.traj_taxi_id = None
        self.sub_nodes_list = None
        self.traj_kg_path = None
        self.continuous_features = None

        self.tte_labels = None
        self.tte_label_masks = None

        self.token_reader = TokenShardReader(
            token_dir=global_vars.token_cache_dir_time,
            chunk_size=global_vars.token_chunk_size,
            total_len=self.dataset_len
        )

        if os.path.exists(global_vars.cached_time_reg_dataset):
            self.load_cached_traj_data()
        else:
            self.read_traj_data()
            self.cache_traj_data()

        self.load_subgraph_nodes()
        self.load_traj_kg_path()

        logging.info(
            f"(DatasetTTE) Number of trajectories: {self.dataset_len}\n"
            f"Shape of traj_road_id_lists: {self.traj_road_id_lists.shape}, \n"
            f"Shape of traj_time_stamp_lists: {self.traj_time_stamp_lists.shape}, \n"
            #f"Shape of traj_time_features_lists: {self.traj_time_features_lists.shape}, \n"
            f"Shape of traj_road_index_lists: {self.traj_road_index_lists.shape}, \n"
            f"Shape of tte_labels: {self.tte_labels.shape}, \n"
            f"Shape of tte_label_masks: {self.tte_label_masks.shape}, \n"
        )

        logging.info("Finish loading TTE dataset.")

    def read_traj_data(self):
        traj_data = file_loader.get_traj_data()
        osm2idx = file_loader.get_osmid_to_index()
        padding_idx = file_loader.get_padding_idx()

        raw_path_lists = traj_data["path"].apply(lambda x: literal_eval(x)).tolist()
        raw_time_lists = traj_data["tlist"].apply(lambda x: literal_eval(x)).tolist()

        self.dataset_len = len(raw_path_lists)

        self.traj_len_lists = torch.tensor(
            [min(len(x), args.seq_len) for x in raw_path_lists],
            dtype=torch.int64
        )

        self.traj_road_id_lists = torch.tensor([
            x[:args.seq_len] if len(x) > args.seq_len else x + [x[-1]] * (args.seq_len - len(x))
            for x in raw_path_lists
        ], dtype=torch.int64)
        logging.info("(DatasetTTE) Finish reading road id.")

        # osm_id -> index_id
        traj_osm = self.traj_road_id_lists.numpy()
        traj_index = [
            [osm2idx.get(int(o), padding_idx) for o in row]
            for row in traj_osm
        ]
        self.traj_road_index_lists = torch.tensor(traj_index, dtype=torch.int64)
        logging.info("(DatasetTTE) Finish reading road index.")

        self.traj_time_stamp_lists = torch.tensor([
            x[:args.seq_len] if len(x) > args.seq_len else x + [x[-1]] * (args.seq_len - len(x))
            for x in raw_time_lists
        ], dtype=torch.int64)
        logging.info("(DatasetTTE) Finish reading raw timestamps.")

        self.traj_time_features_lists = torch.zeros(
            (self.dataset_len, args.seq_len, 6),
            dtype=torch.float32
        )
        logging.info("(DatasetTTE) Finish masking time features.")

        self.traj_taxi_id = torch.tensor(traj_data["usr_id"], dtype=torch.int64)
        logging.info("(DatasetTTE) Finish reading taxi id.")

        if "dist_list" in traj_data.columns:
            self.traj_distance_lists = torch.tensor([
                x[:args.seq_len] if len(x) > args.seq_len else x + [0] * (args.seq_len - len(x))
                for x in traj_data["dist_list"].apply(lambda x: literal_eval(x))
            ], dtype=torch.int64)
        else:
            self.traj_distance_lists = torch.zeros(
                (self.dataset_len, args.seq_len),
                dtype=torch.int64
            )
        logging.info("(DatasetTTE) Finish reading traj distance.")

        self.tte_labels = torch.zeros(
            (self.dataset_len, args.seq_len, 1),
            dtype=torch.float32
        )
        self.tte_label_masks = torch.zeros(
            (self.dataset_len, args.seq_len),
            dtype=torch.bool
        )

        for i, ts in enumerate(raw_time_lists):
            cur_len = min(len(ts), args.seq_len)
            if cur_len <= 1:
                continue

            ts_tensor = torch.tensor(ts[:cur_len], dtype=torch.int64)
            delta_t = (ts_tensor[1:] - ts_tensor[:-1]).to(torch.float32) / 60.0

            if (delta_t < 0).any():
                raise ValueError(f"Found negative delta_t in sample {i}.")

            self.tte_labels[i, 1:cur_len, 0] = delta_t
            self.tte_label_masks[i, 1:cur_len] = True

        logging.info("(DatasetTTE) Finish reading TTE labels.")


    def cache_traj_data(self):
        logging.info(f"Caching TTE dataset to {global_vars.cached_time_reg_dataset}")
        torch.save({
            'traj_road_id_lists': self.traj_road_id_lists,
            'traj_time_stamp_lists': self.traj_time_stamp_lists,
            'traj_road_index_lists': self.traj_road_index_lists,
            'traj_taxi_id': self.traj_taxi_id,
            'traj_distance_lists': self.traj_distance_lists,
            'tte_labels': self.tte_labels,
            'tte_label_masks': self.tte_label_masks,
        }, global_vars.cached_time_reg_dataset)

    def load_cached_traj_data(self):
        logging.info(f"Loading TTE dataset from {global_vars.cached_time_reg_dataset}")
        cached_data = torch.load(global_vars.cached_time_reg_dataset, weights_only=True)

        self.traj_road_id_lists = cached_data['traj_road_id_lists']
        self.traj_time_stamp_lists = cached_data['traj_time_stamp_lists']
        self.traj_road_index_lists = cached_data['traj_road_index_lists']
        self.tte_labels = cached_data['tte_labels']
        self.tte_label_masks = cached_data['tte_label_masks']

    def load_subgraph_nodes(self):
        logging.info(f"加载子图数据文件:{global_vars.cached_subgraph_nodes}")
        data = np.load(global_vars.cached_subgraph_nodes, allow_pickle=True)
        self.sub_nodes_list = data["sub_nodes"]
        logging.info(f"sub_nodes_list size:{len(self.sub_nodes_list)}")

    def load_traj_kg_path(self):
        logging.info(f"加载轨迹点kg_path数据:{global_vars.cached_traj_kg_path}")
        data = torch.load(global_vars.cached_traj_kg_path, map_location="cpu")
        self.traj_kg_path = data["traj_struct_cache"]
        logging.info(f"traj_kg_path size:{len(self.traj_kg_path)}")

    def __getitem__(self, index):
        traj_token_ids = self.token_reader.get(index)  # [64, 45] CPU

        return (
            self.traj_road_id_lists[index],
            self.traj_time_stamp_lists[index],
            self.traj_road_index_lists[index],
            traj_token_ids,
            self.sub_nodes_list[index],
            self.traj_kg_path[index],
            self.tte_labels[index],
            self.tte_label_masks[index],
        )

    def __len__(self):
        return self.dataset_len


class DatasetTrajRecover(Dataset):
    def __init__(self):
        logging.info("Start loading trajectory recover dataset.")

        super().__init__()

        self.dataset_len = file_loader.get_traj_cnt()
        self.traj_road_id_lists = None          # original OSM/road ids, masked positions set to 0
        self.traj_time_stamp_lists = None
        self.traj_time_features_lists = None
        self.traj_road_index_lists = None       # road index ids, masked positions set to road_cnt
        self.sub_nodes_list = None
        self.traj_kg_path = None
        self.traj_recover_labels = None         # [N, num_mask], road_index labels at masked positions

        # New fields for case export
        self.traj_original_road_id_lists = None
        self.traj_original_road_index_lists = None
        self.traj_recover_mask = None           # bool, True means masked
        self.traj_recover_mask_pos = None       # [N, num_mask]
        self.traj_meta = []

        self.token_reader = TokenShardReader(
            token_dir=global_vars.token_cache_dir,
            chunk_size=global_vars.token_chunk_size,
            total_len=self.dataset_len
        )

        if os.path.exists(global_vars.cached_traj_recover_dataset):
            self.load_cached_traj_data()
            if self.traj_recover_mask_pos is None or self.traj_original_road_index_lists is None:
                logging.warning(
                    "Cached recover dataset does not contain case-study fields. "
                    "Rebuilding dataset cache."
                )
                self.read_traj_data()
                self.cache_traj_data()
        else:
            self.read_traj_data()
            self.cache_traj_data()

        self.load_subgraph_nodes()
        self.load_traj_kg_path()

        logging.info(
            f"(DatasetTrajRecover) Number of trajectories: {self.dataset_len}\n"
            f"Shape of traj_road_id_lists: {self.traj_road_id_lists.shape}, \n"
            f"Shape of traj_time_stamp_lists: {self.traj_time_stamp_lists.shape}, \n"
            f"Shape of traj_road_index_lists: {self.traj_road_index_lists.shape}, \n"
            f"Shape of traj_recover_labels: {self.traj_recover_labels.shape}, \n"
            f"Shape of traj_recover_mask_pos: {self.traj_recover_mask_pos.shape}, \n"
        )

        logging.info("Finish loading trajectory recover dataset.")

    def _safe_literal_eval(self, x):
        if isinstance(x, str):
            return literal_eval(x)
        return x

    def _pad_or_truncate(self, x, length):
        x = list(x)
        if len(x) > length:
            return x[:length]
        return x + [x[-1]] * (length - len(x))

    def _extract_optional_gps(self, row, seq_len):
        lon = None
        lat = None

        lon_candidates = ["lngs", "lons", "lon", "lng"]
        lat_candidates = ["lats", "lat"]

        for lon_col in lon_candidates:
            if lon_col in row.index:
                lon = self._safe_literal_eval(row[lon_col])
                break

        for lat_col in lat_candidates:
            if lat_col in row.index:
                lat = self._safe_literal_eval(row[lat_col])
                break

        if lon is None or lat is None:
            for coord_col in ["polyline_list", "coordinates", "coords", "gps_points", "points"]:
                if coord_col in row.index:
                    coords = self._safe_literal_eval(row[coord_col])
                    lon = [p[0] for p in coords]
                    lat = [p[1] for p in coords]
                    break

        if lon is None or lat is None:
            return None, None

        lon = self._pad_or_truncate(lon, seq_len)
        lat = self._pad_or_truncate(lat, seq_len)
        return lon, lat

    def read_traj_data(self):
        traj_data = file_loader.get_traj_data()
        osm2idx = file_loader.get_osmid_to_index()
        padding_idx = file_loader.get_padding_idx()

        road_cnt = file_loader.get_road_cnt()

        mask_seed = int(getattr(args, "mask_seed", 42))
        self.mask, num_mask = self.padding_mask(self.dataset_len, args.seq_len, seed=mask_seed)

        self.traj_recover_mask = (self.mask == 0).bool()
        self.traj_recover_mask_pos = torch.stack([
            torch.where(self.traj_recover_mask[i])[0]
            for i in range(self.dataset_len)
        ], dim=0).to(torch.int64)


        raw_paths = [
            self._pad_or_truncate(self._safe_literal_eval(x), args.seq_len)
            for x in traj_data["path"]
        ]
        self.traj_original_road_id_lists = torch.tensor(raw_paths, dtype=torch.int64)
        logging.info("(DatasetTrajRecover) Finish reading original road id.")

        # osm_id -> index_id
        traj_osm = self.traj_original_road_id_lists.numpy()
        traj_index = [
            [osm2idx.get(int(o), padding_idx) for o in row]
            for row in traj_osm
        ]
        self.traj_original_road_index_lists = torch.tensor(traj_index, dtype=torch.int64)
        logging.info("(DatasetTrajRecover) Finish reading original road index.")

        self.traj_recover_labels = self.traj_original_road_index_lists[self.traj_recover_mask].reshape(self.dataset_len, -1)

        # Masked inputs
        self.traj_road_id_lists = self.traj_original_road_id_lists.clone()
        self.traj_road_index_lists = self.traj_original_road_index_lists.clone()

        self.traj_road_id_lists[self.traj_recover_mask] = 0
        self.traj_road_index_lists[self.traj_recover_mask] = road_cnt
        logging.info("(DatasetTrajRecover) Finish reading road id/index (masked).")

        # Timestamps
        raw_times = [
            self._pad_or_truncate(self._safe_literal_eval(x), args.seq_len)
            for x in traj_data["tlist"]
        ]
        self.traj_time_stamp_lists = torch.tensor(raw_times, dtype=torch.int64)
        self.traj_time_stamp_lists[self.traj_recover_mask] = 0
        logging.info("(DatasetTrajRecover) Finish reading time stamp.")

        self.traj_meta = []
        traj_data_reset = traj_data.reset_index(drop=True)
        for idx, row in traj_data_reset.iterrows():
            lon, lat = self._extract_optional_gps(row, args.seq_len)

            meta = {
                "sample_idx": int(idx),
                "traj_id": int(row["traj_id"]) if "traj_id" in row.index else int(idx),
                "mask_rate": float(args.mask_rate),
                "mask_seed": mask_seed,
                "path_road_id": raw_paths[idx],
                "path_road_index": self.traj_original_road_index_lists[idx].tolist(),
                "mask_pos": self.traj_recover_mask_pos[idx].tolist(),
            }

            for col in ["usr_id", "taxi_id"]:
                if col in row.index:
                    try:
                        meta[col] = int(row[col])
                    except Exception:
                        meta[col] = str(row[col])

            if lon is not None and lat is not None:
                meta["lon"] = lon
                meta["lat"] = lat

            self.traj_meta.append(meta)

    def cache_traj_data(self):
        logging.info(f"Caching trajectory recover dataset to {global_vars.cached_traj_recover_dataset}")
        torch.save({
            "traj_road_id_lists": self.traj_road_id_lists,
            "traj_time_stamp_lists": self.traj_time_stamp_lists,
            "traj_road_index_lists": self.traj_road_index_lists,
            "traj_recover_labels": self.traj_recover_labels,
            "mask": self.mask,

            # New case-study fields
            "traj_original_road_id_lists": self.traj_original_road_id_lists,
            "traj_original_road_index_lists": self.traj_original_road_index_lists,
            "traj_recover_mask": self.traj_recover_mask,
            "traj_recover_mask_pos": self.traj_recover_mask_pos,
            "traj_meta": self.traj_meta,
        }, global_vars.cached_traj_recover_dataset)

    def load_cached_traj_data(self):
        logging.info(f"Loading cached trajectory recover dataset from {global_vars.cached_traj_recover_dataset}")
        cached_data = torch.load(global_vars.cached_traj_recover_dataset, weights_only=False)
        self.traj_road_id_lists = cached_data["traj_road_id_lists"]
        self.traj_time_stamp_lists = cached_data["traj_time_stamp_lists"]
        self.traj_road_index_lists = cached_data["traj_road_index_lists"]
        self.traj_recover_labels = cached_data["traj_recover_labels"]
        self.mask = cached_data["mask"]

        self.traj_original_road_id_lists = cached_data.get("traj_original_road_id_lists")
        self.traj_original_road_index_lists = cached_data.get("traj_original_road_index_lists")
        self.traj_recover_mask = cached_data.get("traj_recover_mask")
        self.traj_recover_mask_pos = cached_data.get("traj_recover_mask_pos")
        self.traj_meta = cached_data.get("traj_meta", [])

    def load_subgraph_nodes(self):
        logging.info(f"加载子图数据文件:{global_vars.cached_subgraph_nodes}")
        data = np.load(global_vars.cached_subgraph_nodes, allow_pickle=True)
        self.sub_nodes_list = data["sub_nodes"]
        logging.info(f"sub_nodes_list size:{len(self.sub_nodes_list)}")

    def load_traj_kg_path(self):
        logging.info(f"加载轨迹点kg_path数据:{global_vars.cached_traj_kg_path}")
        data = torch.load(global_vars.cached_traj_kg_path, map_location="cpu")
        self.traj_kg_path = data["traj_struct_cache"]
        logging.info(f"traj_kg_path size:{len(self.traj_kg_path)}")

    def get_mask(self):
        if self.mask is None:
            raise RuntimeError("mask not loaded. Please call load_all() first.")
        return self.mask

    def __getitem__(self, index):
        traj_token_ids = self.token_reader.get(index).clone()
        point_mask = self.traj_recover_mask[index]

        traj_token_ids[point_mask] = 0

        return (
            self.traj_road_id_lists[index],
            self.traj_time_stamp_lists[index],
            self.traj_road_index_lists[index],
            traj_token_ids,
            self.sub_nodes_list[index],
            self.traj_kg_path[index],
            self.traj_recover_labels[index],
            self.traj_recover_mask_pos[index],
        )

    def __len__(self):
        return self.dataset_len

    def padding_mask(self, B, L, seed=42):
        mask = torch.ones(B, L)
        num_mask = int(args.mask_rate * L)

        generator = torch.Generator()
        generator.manual_seed(seed)

        for i in range(B):
            indices_to_mask = torch.randperm(L, dtype=torch.long, generator=generator)[:num_mask]
            mask[i][indices_to_mask] = 0

        return mask, num_mask



class DatasetKG(Dataset):
    def __init__(self, input_ids=None, attention_mask=None):
        super().__init__()
        self.data = {
            "input_ids": input_ids,
            "att_mask": attention_mask,
        }

    def __len__(self):
        return self.data["input_ids"].size(0)

    def __getitem__(self, index):
        if isinstance(index, torch.Tensor):
            index = index.item()
        batch_data = dict()
        for key in self.data.keys():
            if self.data[key] is not None:
                batch_data[key] = self.data[key][index]
        return batch_data