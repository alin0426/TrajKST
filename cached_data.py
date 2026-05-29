from data_provider.data_loader import DatasetTraj, DatasetDestinationPrediction, DatasetTTE, DatasetTrajRecover
from data_provider.file_loader import file_loader
from config.logging_config import init_logger, make_log_dir



def main():
    log_dir = make_log_dir()
    init_logger(log_dir)

    file_loader.load_all()

    datasets = {
        "traj": DatasetTraj(),
        "destination_prediction": DatasetDestinationPrediction(),  # 增加目的地预测
        "tte": DatasetTTE(),
        "traj_recover": DatasetTrajRecover(),
    }


if __name__ == "__main__":
    main()