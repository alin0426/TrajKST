import os
import logging
import traceback
import numpy as np
from tqdm import tqdm
import wandb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from config.logging_config import init_logger, make_log_dir
from config.args_config import args

from data_provider.file_loader import file_loader
from data_provider.data_loader import DatasetTraj
from data_provider.retriver_sam import sampling_sub

from models.trajkst import TrajKST

from utils.tools import EarlyStopping
from utils.masking import padding_mask
from utils.plot_losses import save_loss_image, save_losses_to_csv

import time
from torch_geometric.data import Batch
from transformers import GPT2Tokenizer


losses = {
    "total": [],
    "road_id": [],
    "time_features": [],
}

def collate_fn(batch):
    (
        batch_road_id,
        batch_time_id,
        batch_road_index_id,
        batch_time_features,
        traj_token_ids,  # Trajectory Textualization
        batch_sub_nodes,  # Topological Subgraph Encoding
        batch_traj_kg_path,  # Semantic Path Reasoning
        #batch_continuous_feat,
    ) = zip(*batch)

    return (
        torch.stack(batch_road_id),
        torch.stack(batch_time_id),
        torch.stack(batch_road_index_id),
        torch.stack(batch_time_features),
        torch.stack(traj_token_ids),
        list(batch_sub_nodes),
        list(batch_traj_kg_path),
        #torch.stack(batch_continuous_feat),
    )

def build_mask_roads(batch_road_id, mask):
    """
    返回：List[Set[int]]，每条轨迹一个 mask_roads
    """
    B, L = batch_road_id.shape
    mask_roads_list = []

    for b in range(B):
        road_ids = batch_road_id[b]      # (L,)
        road_mask = mask[b]              # (L,)
        masked = road_ids[road_mask == 0]
        mask_roads_list.append(set(masked.tolist()))

    return mask_roads_list

def train(device):
    file_loader.load_all()
    file_loader.load_kg_nx()

    dataset = DatasetTraj()
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=min(8, os.cpu_count()), pin_memory=True,persistent_workers=True)

    trajkst = TrajKST(device).to(device)
    mse = nn.MSELoss()
    cross_entropy = nn.CrossEntropyLoss()
    text_tokenizer = GPT2Tokenizer.from_pretrained("./models/gpt2")
    text_tokenizer.pad_token = text_tokenizer.eos_token

    optimizer = torch.optim.Adam(trajkst.parameters(), lr=args.learning_rate, weight_decay=args.weight)

    steps_per_epoch = len(data_loader)
    total_training_steps = steps_per_epoch * args.train_epochs

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_training_steps * 0.1),
        num_training_steps=total_training_steps
    )

    start_epoch = 1

    if args.resume is not None:
        print(f"==> Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)

        trajkst.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"] is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            print("==> Scheduler state loaded")
        else:
            print("==> No scheduler state found, using fresh scheduler")

        start_epoch = checkpoint["epoch"] + 1
        print(f"==> Resumed from epoch {checkpoint['epoch']}")

    early_stopping = EarlyStopping("pretrain", patience=args.patience, verbose=True)

    data_loader_len = len(data_loader)

    for epoch in range(start_epoch, args.train_epochs + 1):

        progress_bar = tqdm(enumerate(data_loader), 
                            total=len(data_loader), 
                            desc=f"Epoch {epoch}/{args.train_epochs}", 
                            unit="batch"
        )

        for batchidx, batch in progress_bar:
            batch_road_id, batch_time_id, batch_road_index_id, batch_time_features, traj_token_ids, batch_sub_nodes, batch_traj_struct = batch

            # Move data to GPU
            batch_road_id = batch_road_id.to(device)
            batch_time_id = batch_time_id.to(device)
            batch_road_index_id = batch_road_index_id.to(device)
            batch_time_features = batch_time_features.to(device)
            traj_token_ids = traj_token_ids.to(device)
            #batch_continuous_feat = batch_continuous_feat.to(device)

            batch_sub = []
            for sub_nodes in batch_sub_nodes:
                final_sub = file_loader.G.subgraph(sub_nodes)
                batch_sub.append(final_sub)

            B, L, N, Dtf = batch_road_id.shape[0], batch_road_id.shape[1], file_loader.road_cnt, 6
            
            # Get mask
            mask, num_mask = padding_mask(B, L)     # num_mask = 0.5*L
            mask = mask.to(device)

            mask_roads_list = build_mask_roads(batch_road_id, mask)
            road2eid = file_loader.get_road2eid()
            mask_entity_list = [[road2eid[oid] for oid in m] for m in mask_roads_list]

            flat_texts = []
            traj_window_index = []
            for b in range(B):
                traj_struct = batch_traj_struct[b]
                mask_roads = mask_entity_list[b]

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

            # Forward pass
            predict_road_id, predict_time_features = trajkst(
                batch_road_id, batch_time_id, mask, num_mask, batch_sub, batch_road_index_id, traj_token_ids, kg_token_ids, traj_window_index
            )

            # Get masked real values
            real_road_id = batch_road_index_id[mask == 0]
            real_time_features = batch_time_features[mask == 0]

            # Calculate individual losses
            road_id_loss = cross_entropy(predict_road_id.view(-1, N), real_road_id)  # 交叉熵
            time_features_loss = mse(predict_time_features.view(-1, Dtf), real_time_features)  # MSE

            # Calculate total loss with scaling factors
            loss = road_id_loss * args.loss_alpha + time_features_loss * args.loss_beta

            # Record the losses for each component
            losses["total"].append(loss.item())
            losses["road_id"].append(road_id_loss.item())
            losses["time_features"].append(time_features_loss.item())

            progress_bar.set_postfix({"loss": f"{loss.item():.2f}"}) 
            
            #Log losses to wandb
            wandb.log({
                "batch": batchidx,
                "batch_total_loss": loss.item(),
                "batch_road_id_loss": road_id_loss.item(),
                "batch_time_features_loss": time_features_loss.item(),
            })
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        # Calculate average training loss for this epoch
        epoch_loss_ave = np.mean(losses["total"][-data_loader_len:])
        epoch_road_id_loss_ave = np.mean(losses["road_id"][-data_loader_len:])
        epoch_time_features_loss_ave = np.mean(losses["time_features"][-data_loader_len:])

        # Log average losses to wandb
        wandb.log({
            "epoch_average_total_loss": epoch_loss_ave,
            "epoch_average_road_id_loss": epoch_road_id_loss_ave,
            "epoch_average_time_features_loss": epoch_time_features_loss_ave,
            "learning_rate": optimizer.param_groups[0]['lr']
        })

        # Early stopping and lr scheduler step
        early_stopping(epoch_loss_ave, trajkst, optimizer, epoch)


def main():
    project_name = "model-dev" if args.develop else "model"
    wandb.init(mode=args.wandb_mode, project=project_name, config=args, name=f"pretrain-{args.city}")

    log_dir = make_log_dir(args.log_path)
    init_logger(log_dir)

    device = torch.device("cpu" if args.device == "-1" and torch.cuda.is_available() else f"cuda:{args.device}")
    logging.info(f"Using device: {device}")

    try:
        train(device)
    except KeyboardInterrupt:
        logging.info("\nTraining interrupted by user.")
    finally:
        # 保存loss
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
