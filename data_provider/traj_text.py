import time
import torch

from config import global_vars
from transformers import GPT2Tokenizer
from tqdm import tqdm
import os

# 提前离线缓存，轨迹数据每条轨迹对应的text_token_ids
MAX_LEN = 45
TRAJ_CHUNK = 16000      # 每次 16000条轨迹
TEXT_BATCH = 2048       # tokenizer batch size

# 一次性传入16000条
def traj_to_text(batch_road_id, batch_time_id, batch_taxi_id, batch_dis):
    # print("开始轨迹文本化")
    # 遍历每条轨迹
    batch_traj_text = []  # batch中所有轨迹的text
    """
    [TRA] [CHE] At the beginning [Useri] depart from [Roadj] at 2013-07-01 7:00:00;
    [STR] After a time of 30 seconds and a distance of 100 meters [Useri] pass through [Roadj] at 2013-07-01 7:00:30;
    [STR] After a time of 30 seconds and a distance of 100 meters [Useri] pass through [Roadj] at 2013-07-01 7:01:00;
    [CHE] At the end [Useri] arrive at [Roadj] at 2013-07-01 7:30:00;
    """
    # for road_id_list, time_id_list, dis_list, time_list, taxi_id in zip(batch_road_id, batch_time_id, batch_dis,
    #                                                                     batch_time, batch_taxi_id):
    for road_id_list, time_id_list, taxi_id, dis_list in zip(batch_road_id, batch_time_id, batch_taxi_id, batch_dis):

        # 遍历每条轨迹的每个轨迹点
        sentence = []
        head = '[User{}]'.format(taxi_id)
        #head = '[MASK]'
        i = 0  # 第一个轨迹点
        prev_time = None
        # for road_id, time_id, dis_gap, time_gap in zip(road_id_list, time_id_list, dis_list, time_list):
        for road_id, time_id, dis_gap in zip(road_id_list, time_id_list, dis_list):
            time_gap = 0
            word_road = '[Road{}]'.format(road_id)
            # 将时间戳转换为真实时间
            real_time = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time_id))
            #real_time = '[MASK]'
            # 将每个轨迹点信息和taxi_id构成一句文本
            if i == 0:  # 第一个轨迹点
                # word_dis = ''  # 第一个dis设为0
                s = '[TRA] [CHE] At the beginning' + ' ' + head + ' ' + 'depart from' + ' ' + word_road + ' ' + 'at ' + real_time + ';'
            elif i == len(road_id_list) - 1:  # 最后一个轨迹点
                # word_dis = 'and a distance of {} meters'.format(dis_gap)
                s = '[CHE] At the end' + ' ' + head + ' ' + 'arrive at' + ' ' + word_road + ' at ' + real_time + ';'
            else:
                #time_gap = int(time_id - prev_time)
                time_gap = 0
                word_dis = 'and a distance of {} meters'.format(dis_gap)
                s = '[STR] After a time of {} seconds'.format(
                    time_gap) + ' ' + word_dis + ' ' + head + ' ' + 'pass through' + ' ' + word_road + ' at ' + real_time + ';'

            prev_time = time_id
            i = i + 1
            # 将每条轨迹的文本聚合起来
            sentence.append(s)
            # print(s)

        batch_traj_text.append(sentence)  #


    return batch_traj_text

# tokenizer初始化
text_tokenizer = GPT2Tokenizer.from_pretrained("./models/gpt2")
text_tokenizer.pad_token = text_tokenizer.eos_token

os.makedirs("./dataset/porto/test_cache/cached_traj_text_tokens_des", exist_ok=True)

cached_data = torch.load('./dataset/porto/test_cache/cached_destination_prediction_dataset.pth', weights_only=True)
traj_road_id_lists = cached_data['traj_road_id_lists']
traj_time_stamp_lists = cached_data['traj_time_stamp_lists']
#traj_time_features_lists = cached_data['traj_time_features_lists']
traj_road_index_lists = cached_data['traj_road_index_lists']
traj_taxi_id = cached_data['traj_taxi_id']
traj_distance_lists = cached_data['traj_distance_lists']
#traj_label=cached_data['destination_labels']


num_traj = len(traj_road_index_lists)  # 总轨迹条数

print(f"轨迹条数：{num_traj}")


# 每次处理5000条轨迹
for start in tqdm(range(0, num_traj, TRAJ_CHUNK), desc="Offline Tokenizing"):
    end = min(start + TRAJ_CHUNK, num_traj)   # 不满5000条轨迹的最后一个循环

    # ===== 1. 文本构造（chunk 内）=====
    traj_text_list = traj_to_text(
        traj_road_index_lists[start:end],
        traj_time_stamp_lists[start:end],
        traj_taxi_id[start:end],
        traj_distance_lists[start:end]
    )  # [chunk, 64]

    flat_text = [s for traj in traj_text_list for s in traj]  # [chunk*64]

    # ===== 2. tokenizer（分 batch）=====
    token_ids_chunks = []
    # 每次处理2048条文本
    for i in range(0, len(flat_text), TEXT_BATCH):
        batch_text = flat_text[i:i + TEXT_BATCH]

        out = text_tokenizer(
            batch_text,
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt"
        )

        token_ids_chunks.append(out["input_ids"])

    token_ids = torch.cat(token_ids_chunks, dim=0).int()   # 32位
    token_ids = token_ids.view(end - start, 64, MAX_LEN)  # [16000, 64, 45]

    # ===== 3. 保存 =====
    save_path = f"./dataset/porto/test_cache/cached_traj_text_tokens_des/traj_tokens_{start}_{end}.pt"
    torch.save(token_ids, save_path)

    # ===== 4. 及时释放（非常重要）=====
    del traj_text_list, flat_text, token_ids_chunks, token_ids

print(f"成功保存：{save_path}")


