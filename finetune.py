import os
import logging
import traceback
import numpy as np
from tqdm import tqdm
import wandb

import torch
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from config.logging_config import init_logger, make_log_dir
from config.args_config import args

from data_provider.file_loader import file_loader
from data_provider.data_loader import  DatasetTrajRecover, DatasetDestinationPrediction, DatasetTTE

from models.trajkstfinetune import TrajKSTFineTune

from utils.tools import EarlyStopping
from utils.round_iterator import RoundRobinIterator
from utils.plot_losses import save_loss_image, save_losses_to_csv

from transformers import GPT2Tokenizer

import torch.nn.functional as F
from collections import defaultdict


losses = {
    "destination_prediction": [],
    #"traj_recover": [],
    #"tte": []
}

def collate_fn(batch):
    (
        batch_road_id,
        batch_time_id,
        batch_road_index_id,
        #batch_time_features,
        traj_token_ids,
        batch_sub_nodes,
        batch_traj_struct,
        #batch_continuous_feat,
        batch_label,
        #batch_label_mask,
    ) = zip(*batch)

    return (
        torch.stack(batch_road_id),
        torch.stack(batch_time_id),
        torch.stack(batch_road_index_id),
        #torch.stack(batch_time_features),
        torch.stack(traj_token_ids),
        list(batch_sub_nodes),
        list(batch_traj_struct),
        #torch.stack(batch_continuous_feat),
        torch.stack(batch_label),
        #torch.stack(batch_label_mask),
    )

def train(device):
    file_loader.load_all()
    file_loader.load_kg_nx()  # self.G
    
    datasets = {
        #"traj_recover": DatasetTrajRecover(),
        "destination_prediction": DatasetDestinationPrediction(),   # 增加目的地预测
        #"tte": DatasetTTE(),
    }

    dataloaders = {
        name: DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=min(8, os.cpu_count()), pin_memory=True,persistent_workers=True)
        for name, dataset in datasets.items()
    }


    trajkst = TrajKSTFineTune(device, args.ckpt).to(device)
    text_tokenizer = GPT2Tokenizer.from_pretrained("./models/gpt2")
    text_tokenizer.pad_token = text_tokenizer.eos_token
    
    optimizer = torch.optim.Adam(trajkst.parameters(), lr=args.learning_rate, weight_decay=args.weight)


    steps_per_epoch = sum(len(loader) for loader in dataloaders.values())
    total_training_steps = steps_per_epoch * args.train_epochs


    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_training_steps * 0.1),  # 10%  epoch 做线性热身
        num_training_steps=total_training_steps  # 总epoch数
    )

    early_stopping = EarlyStopping("finetune", patience=args.patience, verbose=True)
    
    data_loader_len = file_loader.get_traj_cnt()    # 轨迹条数（样本数）

    global_step = 0
    for epoch in range(1, args.train_epochs + 1):
        iterator = RoundRobinIterator(dataloaders)
        
        progress_bar = tqdm(
            enumerate(iterator),
            total=len(iterator),
            unit="batch"
        )

        epoch_correct = 0
        epoch_total = 0

        for batch_idx, (task_name, (batch_road_id, batch_time_id, batch_road_index_id, traj_token_ids, batch_sub_nodes, batch_traj_struct, batch_label)) in progress_bar:
            progress_bar.set_description(f"Epoch {epoch}/{args.train_epochs} - Task: {task_name: <18}")
            is_dest_pred = (task_name == 'destination_prediction')
            global_step += 1

            # Move data to GPU
            batch_road_id = batch_road_id.to(device)
            batch_time_id = batch_time_id.to(device)
            #batch_time_features = batch_time_features.to(device)
            batch_label = batch_label.to(device)
            batch_road_index_id = batch_road_index_id.to(device)
            traj_token_ids = traj_token_ids.to(device)
            #batch_label_mask = batch_label_mask.to(device)
            #batch_continuous_feat = batch_continuous_feat.to(device)
            B, L, N, Dtf = batch_road_id.shape[0], batch_road_id.shape[1], file_loader.road_cnt, 6

            batch_sub = []
            for sub_nodes in batch_sub_nodes:
                final_sub = file_loader.G.subgraph(sub_nodes)
                batch_sub.append(final_sub)

            mask_entity_list = []
            if task_name == 'traj_recover':
                mask_entity_list=batch_label + 1 # road_index+1=road_entity_id

            flat_texts = []
            traj_window_index = []
            # 遍历batch中的每条轨迹
            for b in range(B):
                traj_struct = batch_traj_struct[b]
                road_seq = traj_struct['road_seq']
                if len(road_seq) != 0 and is_dest_pred:
                    mask_roads = [road_seq[-1]]  # 轨迹中最后一个road eid
                elif task_name == 'traj_recover':
                    mask_roads = mask_entity_list[b]
                else:
                    mask_roads = []

                sentences = []
                for pair in traj_struct["pairs"]:
                    s = file_loader.pair_to_text(pair, mask_roads)
                    sentences.append(s)

                windows = file_loader.sentence_sliding_window(
                    sentences,
                    window_size=5,
                    stride=3
                )
                for w in windows:
                    traj_window_index.append(b)
                    flat_texts.append(w)

            enc = text_tokenizer(
                flat_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(device)

            kg_token_ids = enc["input_ids"]


            # Forward pass (loss has been calculated)
            if task_name == 'traj_recover':
                loss, predict_classify_id = trajkst(
                    task_name, batch_road_id, batch_time_id, batch_sub, batch_road_index_id, traj_token_ids, kg_token_ids,
                    traj_window_index, batch_label
                )

            if task_name == 'tte':
                # kg
                loss, predict_classify_id = trajkst(
                    task_name, batch_road_id, batch_time_id, batch_sub, batch_road_index_id, traj_token_ids, kg_token_ids, traj_window_index, batch_label, batch_label_mask
                )
            else:
                loss, predict_classify_id = trajkst(
                    task_name, batch_road_id, batch_time_id, batch_sub, batch_road_index_id, traj_token_ids, kg_token_ids, traj_window_index,
                    batch_label
                )

            # Record loss
            losses[task_name].append(loss.item())
            
            # Log losses to wandb
            wandb.log({
                "batch": batch_idx,
                f"{task_name}_batch_loss": loss.item()
            })
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            scheduler.step()


        average_losses = {f"{task_name}_epoch_average_loss": np.mean(task_losses[-data_loader_len:]) 
                          for task_name, task_losses in losses.items()}
        average_losses["learning_rate"] = optimizer.param_groups[0]['lr']
        logging.info(f"Epoch:{epoch},{average_losses}")
        wandb.log(average_losses)
        
        #Step the learning rate scheduler
        early_stopping.save_checkpoint(trajkst, optimizer, epoch, f"{epoch}")   # 每个epoch保存一个checkpoint

def main():
    project_name = "trajkst-dev" if args.develop else "trajkst"
    wandb.init(mode=args.wandb_mode, project=project_name, config=args, name=f"finetune-{args.city}")
    
    log_dir = make_log_dir(args.log_path)
    init_logger(log_dir)
    
    device = torch.device("cpu" if args.device == "-1" and torch.cuda.is_available() else f"cuda:{args.device}")
    logging.info(f"Using device: {device}")
    
    try:
        train(device)
    except KeyboardInterrupt:
        logging.info("Training interrupted by user.")
    finally:
        logging.info(f"Saving losses to {log_dir}.")
        save_loss_image(losses, log_dir)
        save_losses_to_csv(losses, log_dir)
        
        logging.info(f"Finishing training.")
        
    wandb.finish()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error("\n" + traceback.format_exc())