import logging
import traceback
import numpy as np
from tqdm import tqdm
import wandb

import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch.nn.functional as F

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score, top_k_accuracy_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import label_binarize

from config.logging_config import init_logger, make_log_dir
from config.args_config import args

from data_provider.file_loader import file_loader
from data_provider.data_loader import DatasetTrajRecover, DatasetDestinationPrediction,DatasetTTE

from models.trajkstfinetune import TrajKSTFineTune

from transformers import GPT2Tokenizer


from collections import defaultdict


import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm

def bootstrap_eval(eval_func, preds_tensor, labels_tensor, num_bootstrap=1000,
                   mask_tensor=None, seed=42, task_name="unknown"):
    """
    对测试集预测结果做 bootstrap 重采样，估计指标波动。
    """
    rng = np.random.default_rng(seed)
    N = preds_tensor.size(0)

    raw_scores = defaultdict(list)

    boot_bar = tqdm(
        range(num_bootstrap),
        total=num_bootstrap,
        desc=f"Bootstrap [{task_name}]",
        unit="sample"
    )

    for _ in boot_bar:
        # 有放回采样
        boot_idx = rng.choice(N, size=N, replace=True)
        boot_idx = torch.as_tensor(boot_idx, dtype=torch.long)

        boot_preds = preds_tensor[boot_idx]
        boot_labels = labels_tensor[boot_idx]

        if mask_tensor is not None:
            boot_mask = mask_tensor[boot_idx]
            metric_dict = eval_func(boot_preds, boot_labels, boot_mask)
        else:
            metric_dict = eval_func(boot_preds, boot_labels)

        for k, v in metric_dict.items():
            raw_scores[k].append(float(v))

    summary = {}
    for k, values in raw_scores.items():
        values = np.array(values, dtype=np.float64)
        summary[k] = {
            "mean": values.mean(),
            "std": values.std(ddof=1),
            "ci_lower": np.percentile(values, 2.5),
            "ci_upper": np.percentile(values, 97.5),
        }

    return summary, raw_scores

def bootstrap_destination_fast(preds_tensor, labels_tensor, num_bootstrap=200, seed=42, task_name="destination"):
    """
    preds_tensor: [N, C] logits
    labels_tensor: [N]
    """
    rng = np.random.default_rng(seed)

    # 只预处理一次
    labels_np = labels_tensor.detach().cpu().numpy()
    pred_top1 = preds_tensor.argmax(dim=1).detach().cpu().numpy()

    top5_idx = preds_tensor.topk(k=5, dim=1).indices.detach().cpu().numpy()
    top5_hit = (top5_idx == labels_np[:, None]).any(axis=1).astype(np.float32)

    N = len(labels_np)
    all_classes = np.unique(labels_np)

    acc1_list = []
    acc5_list = []
    macro_rec_list = []

    for _ in tqdm(range(num_bootstrap), desc=f"Bootstrap [{task_name}]", total=num_bootstrap, unit="iter"):
        idx = rng.choice(N, size=N, replace=True)

        boot_labels = labels_np[idx]
        boot_top1 = pred_top1[idx]
        boot_top5_hit = top5_hit[idx]

        acc1 = np.mean(boot_top1 == boot_labels)
        acc5 = np.mean(boot_top5_hit)

        recalls = []
        for c in all_classes:
            mask = (boot_labels == c)
            if mask.sum() > 0:
                recalls.append(np.mean(boot_top1[mask] == c))
        macro_rec = float(np.mean(recalls)) if len(recalls) > 0 else 0.0

        acc1_list.append(acc1)
        acc5_list.append(acc5)
        macro_rec_list.append(macro_rec)

    def summarize(values):
        values = np.asarray(values, dtype=np.float64)
        return {
            "mean": values.mean(),
            "std": values.std(ddof=1),
            "ci_lower": np.percentile(values, 2.5),
            "ci_upper": np.percentile(values, 97.5),
        }

    summary = {
        "acc1": summarize(acc1_list),
        "acc5": summarize(acc5_list),
        "macro_recall": summarize(macro_rec_list),
    }

    raw_scores = {
        "acc1": acc1_list,
        "acc5": acc5_list,
        "macro_recall": macro_rec_list,
    }

    return summary, raw_scores

def bootstrap_tte_fast(preds_tensor, labels_tensor, mask_tensor, num_bootstrap=200, seed=42, task_name="tte", eps=1e-6):
    rng = np.random.default_rng(seed)

    preds = preds_tensor
    labels = labels_tensor
    mask = mask_tensor.float()

    if preds.dim() == 3:
        preds = preds.squeeze(-1)
    if labels.dim() == 3:
        labels = labels.squeeze(-1)

    # 只算一次每条轨迹的总预测 / 总真实
    traj_pred = (preds * mask).sum(dim=1).detach().cpu().numpy()
    traj_label = (labels * mask).sum(dim=1).detach().cpu().numpy()

    N = len(traj_label)

    mae_list = []
    rmse_list = []
    mape_list = []

    for _ in tqdm(range(num_bootstrap), desc=f"Bootstrap [{task_name}]", total=num_bootstrap, unit="iter"):
        idx = rng.choice(N, size=N, replace=True)

        boot_pred = traj_pred[idx]
        boot_label = traj_label[idx]

        err = boot_pred - boot_label
        mae = np.mean(np.abs(err))
        rmse = np.sqrt(np.mean(err ** 2))

        valid = boot_label > eps
        if np.any(valid):
            mape = np.mean(np.abs(err[valid] / boot_label[valid])) * 100
        else:
            mape = 0.0

        mae_list.append(mae)
        rmse_list.append(rmse)
        mape_list.append(mape)

    def summarize(values):
        values = np.asarray(values, dtype=np.float64)
        return {
            "mean": values.mean(),
            "std": values.std(ddof=1),
            "ci_lower": np.percentile(values, 2.5),
            "ci_upper": np.percentile(values, 97.5),
        }

    summary = {
        "mae": summarize(mae_list),
        "rmse": summarize(rmse_list),
        "mape": summarize(mape_list),
    }

    raw_scores = {
        "mae": mae_list,
        "rmse": rmse_list,
        "mape": mape_list,
    }

    return summary, raw_scores

def eval_destination_prediction(preds, labels):
    # preds: (N, C) logits
    preds_np = preds.softmax(dim=1).detach().cpu().numpy()  # sklearn 需要概率，且要在 CPU 上
    labels_np = labels.detach().cpu().numpy()

    acc1 = accuracy_score(labels_np, preds_np.argmax(axis=1))

    # 关键修复：显式提供所有类别（顺序要和 preds_np 的列顺序一致）
    all_classes = np.arange(preds_np.shape[1])  # [0..C-1]
    acc5 = top_k_accuracy_score(labels_np, preds_np, k=5, labels=all_classes)

    macro_rec = recall_score(labels_np, preds_np.argmax(axis=1), average='macro',zero_division=0)

    # logging.info(
    #     f"TRAJ_DESTINATION  ACC@1: {acc1:.4f},  ACC@5: {acc5:.4f},  Recall: {macro_rec:.4f}"
    # )
    #return acc1, acc5, macro_rec
    return {
        "acc1": acc1,
        "acc5": acc5,
        "macro_recall": macro_rec
    }


def eval_traj_recover(preds, labels):
    preds = preds.reshape(-1, preds.shape[-1])
    labels = labels.reshape(-1)
    pred_proba = F.softmax(preds, dim=1)
    pred_class = torch.argmax(pred_proba, dim=1)

    acc = accuracy_score(labels, pred_class)

    f1 = f1_score(labels, pred_class, average='macro')

    logging.info(f"TRAJ_RECOVER ACC: {acc:.4f}, F1: {f1:.4f}")

    return acc, f1

def eval_traj_tte(preds, labels, mask, eps=1e-6):
    if preds.dim() == 3:
        preds = preds.squeeze(-1)
    if labels.dim() == 3:
        labels = labels.squeeze(-1)

    mask = mask.float()
    traj_pred = (preds * mask).sum(dim=1)
    traj_label = (labels * mask).sum(dim=1)

    mae = torch.mean(torch.abs(traj_pred - traj_label))
    rmse = torch.sqrt(torch.mean((traj_pred - traj_label) ** 2))

    valid = traj_label > eps
    mape = torch.mean(torch.abs((traj_pred[valid] - traj_label[valid]) / traj_label[valid])) * 100
    #logging.info(f"TRAJ_TTE  MAE: {mae.item():.4f},  RMSE: {rmse.item():.4f},  MAPE: {mape.item():.4f}")

    #return mae.item(), rmse.item(), mape.item()
    return {
        "mae": mae.item(),
        "rmse": rmse.item(),
        "mape": mape.item()
    }


def collate_fn(batch):
    (
        batch_road_id,
        batch_time_id,
        batch_road_index_id,
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
        torch.stack(traj_token_ids),
        list(batch_sub_nodes),
        list(batch_traj_struct),
        #torch.stack(batch_continuous_feat),
        torch.stack(batch_label),
        #torch.stack(batch_label_mask),
    )

def evaluate(device):

    file_loader.load_all()
    file_loader.load_kg_nx()  # self.G
    
    datasets = {
        "destination_prediction": DatasetDestinationPrediction(),  # 增加目的地预测
        #"traj_recover": DatasetTrajRecover(),
        #"tte": DatasetTTE(),
    }


    dataloaders = {
        name: DataLoader(dataset, batch_size=128, shuffle=False, collate_fn=collate_fn)
        for name, dataset in datasets.items()
    }

    # 评估指标
    eval_funcs = {
        "destination_prediction": eval_destination_prediction,
        "traj_recover": eval_traj_recover,
        "tte": eval_traj_tte,
    }
    

    trajkst = TrajKSTFineTune(device, args.ckpt).to(device)
    trajkst.eval()  # 冻结参数

    text_tokenizer = GPT2Tokenizer.from_pretrained("./models/gpt2")
    text_tokenizer.pad_token = text_tokenizer.eos_token

    # 遍历所有下游任务
    for task_name, dataloader in dataloaders.items():
        logging.info(f"Evaluating task: {task_name}")
        
        progress_bar = tqdm(
            enumerate(dataloader), 
            total=len(dataloader), 
            unit="batch"
        )

        all_preds, all_labels = [], []
        all_tte_mask = []

        for batch_idx, (batch_road_id, batch_time_id, batch_road_index_id, traj_token_ids, batch_sub_nodes, batch_traj_struct, batch_label) in progress_bar:
            progress_bar.set_description(f"Task: {task_name: <18}")
            is_dest_pred = (task_name == 'destination_prediction')

            batch_road_id = batch_road_id.to(device)
            batch_time_id = batch_time_id.to(device)
            batch_label = batch_label.to(device)    # 真实值
            batch_road_index_id = batch_road_index_id.to(device)
            traj_token_ids = traj_token_ids.to(device)

            B, L, N, Dtf = batch_road_id.shape[0], batch_road_id.shape[1], file_loader.road_cnt, 6

            batch_sub = []
            for sub_nodes in batch_sub_nodes:
                final_sub = file_loader.G.subgraph(sub_nodes).copy()
                batch_sub.append(final_sub)

            mask_entity_list = []
            if task_name == 'traj_recover':
                mask_entity_list = batch_label + 1

            flat_texts = []
            traj_window_index = []

            for b in range(B):
                traj_struct = batch_traj_struct[b]
                road_seq = traj_struct['road_seq']

                result = list(dict.fromkeys(road_seq))
                mask_roads = []
                if len(road_seq) != 0 and is_dest_pred:

                    mask_roads.append(result[-1])
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


            with torch.no_grad():   # 确保不记录计算图，省显存+提速

                if task_name == 'tte':
                    _, batch_pred = trajkst(
                        task_name, batch_road_id, batch_time_id, batch_road_index_id,traj_token_ids, kg_token_ids, traj_window_index,
                        batch_label, batch_label_mask
                    )
                    all_tte_mask.append(batch_label_mask.cpu())
                else:
                    _, batch_pred = trajkst(
                        task_name, batch_road_id, batch_time_id, batch_sub, batch_road_index_id, traj_token_ids, kg_token_ids, traj_window_index, batch_label
                    )

                all_preds.append(batch_pred.cpu())
                all_labels.append(batch_label.cpu())


        # 拼接
        preds_tensor = torch.cat(all_preds, dim=0)
        labels_tensor = torch.cat(all_labels, dim=0)

        if task_name == 'tte':
            tte_mask_tensor = torch.cat(all_tte_mask, dim=0)

            main_metrics = eval_funcs[task_name](preds_tensor, labels_tensor, tte_mask_tensor)
            logging.info(f"[{task_name}] original metrics: {main_metrics}")

            boot_summary, _ = bootstrap_tte_fast(
                preds_tensor=preds_tensor,
                labels_tensor=labels_tensor,
                mask_tensor=tte_mask_tensor,
                num_bootstrap=1000,
                seed=42,
                task_name=task_name
            )

        elif task_name == 'destination_prediction':
            main_metrics = eval_funcs[task_name](preds_tensor, labels_tensor)
            logging.info(f"[{task_name}] original metrics: {main_metrics}")

            boot_summary, _ = bootstrap_destination_fast(
                preds_tensor=preds_tensor,
                labels_tensor=labels_tensor,
                num_bootstrap=1000,
                seed=42,
                task_name=task_name
            )

        else:
            main_metrics = eval_funcs[task_name](preds_tensor, labels_tensor)
            logging.info(f"[{task_name}] original metrics: {main_metrics}")
            boot_summary = None

        if boot_summary is not None:
            for metric_name, stat in boot_summary.items():
                logging.info(
                    f"[{task_name}] {metric_name}: "
                    f"{stat['mean']:.6f} ± {stat['std']:.6f} "
                    f"(95% CI: {stat['ci_lower']:.4f}, {stat['ci_upper']:.4f})"
                )


def main():
    project_name = "trajkst-dev" if args.develop else "trajkst"
    wandb.init(mode=args.wandb_mode, project=project_name, config=args, name=f"evaluate-{args.city}")
    
    log_dir = make_log_dir(args.log_path)
    init_logger(log_dir)
    
    device = torch.device("cpu" if args.device == "-1" and torch.cuda.is_available() else f"cuda:{args.device}")
    logging.info(f"Using device: {device}")
    
    try:
        evaluate(device)
    except KeyboardInterrupt:
        logging.info("Training interrupted by user.")
    finally:
        logging.info(f"Saving losses to {log_dir}.")
        
        logging.info(f"Finishing training.")
        
    wandb.finish()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error("\n" + traceback.format_exc())