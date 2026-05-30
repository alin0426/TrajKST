from data_provider.traj_text_ddp import build_and_cache_traj_tokens
"""
    1.第一阶段训练数据：cached_traj_dataset.pth -> cached_traj_text_tokens
    2.第二阶段 目的地预测任务：cached_destination_prediction_dataset.pth -> cached_traj_text_tokens_des
              轨迹分类任务：cached_traj_classify_dataset.pth -> cached_traj_text_tokens_nouser

"""

print("开始执行：cached_token_ids.py文件")

build_and_cache_traj_tokens(
    cached_pth='./dataset/porto/cached_traj_dataset.pth',
    save_dir='./dataset/porto/cached_traj_text_tokens/',
    task_name='traj',
)

build_and_cache_traj_tokens(
    cached_pth='./dataset/porto/cached_destination_prediction_dataset.pth',
    save_dir='./dataset/porto/cached_traj_text_tokens_des/',
    task_name='destination',
)

build_and_cache_traj_tokens(
    cached_pth='./dataset/porto/cached_traj_dataset.pth',
    save_dir='./dataset/porto/cached_traj_text_tokens_time/',
    task_name='tte',
)


