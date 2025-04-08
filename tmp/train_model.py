import torch
from support_web import (
    get_saved_model_path,
    create_signal2features_preprocess,
    create_model,
    get_dataloader,
    find_implementations,
    get_class_by_label,
    single_signal_preprocess_to_matrix_preprocess,
    create_preprocess_chain,
    get_last_dim_size,
)
from abstract import SignalProcessorBase
from signalProcessorInstance import (
    MeanFilterSignalProcessor,
    HampelFilterSignalProcessor,
    HampelAndMeanFilterSignalProcessor,
    FftSignalProcessor,
    WaveletSignalProcessor,
    EmdSignalProcessor,
    RawSignalProcessor,
)
import time
import os


def get_signal_processor(method_name):
    """获取信号处理器"""
    processors = {
        "raw": RawSignalProcessor,
        "mean_filter": MeanFilterSignalProcessor,
        "hampel_filter": HampelFilterSignalProcessor,
        "hampel_then_mean_filter": HampelAndMeanFilterSignalProcessor,
        "fft": FftSignalProcessor,
        "wavelet": WaveletSignalProcessor,
        "emd": EmdSignalProcessor,
    }
    if method_name not in processors:
        raise ValueError(f"不支持的信号处理方法: {method_name}")
    return processors[method_name]()


def train_model(
    model_name,
    signal_process_method,
    feature_type,
    train_dataset,
    test_dataset,
    model_save_filename,
    train_csv_files=None,
    train_csv_labels=None,
    test_csv_files=None,
    test_csv_labels=None,
):
    """
    训练模型的主函数

    参数:
        model_name: 模型名称
        signal_process_method: 信号处理方法名称
        feature_type: 特征类型
        train_dataset: 训练数据集名称
        test_dataset: 测试数据集名称
        model_save_filename: 模型保存文件名
        train_csv_files: 自定义训练CSV文件列表
        train_csv_labels: 自定义训练标签列表
        test_csv_files: 自定义测试CSV文件列表
        test_csv_labels: 自定义测试标签列表
    """
    torch.manual_seed(97)

    print("设置信号处理和特征提取")
    # 设置信号处理和特征提取
    signal_processor = get_signal_processor(signal_process_method)
    signal_preprocess = single_signal_preprocess_to_matrix_preprocess(
        signal_processor.process
    )

    print("特征提取")
    # 特征提取
    features_preprocess = create_signal2features_preprocess(feature_type)
    preprocess = create_preprocess_chain([signal_preprocess, features_preprocess])

    print("创建模型")
    # 创建模型
    last_dim_size = get_last_dim_size(feature_type, 64)
    model = create_model(model_name, last_dim_size)

    print("获取数据加载器")
    # 获取数据加载器
    train_loader, valid_loader = get_dataloader(
        train_dataset,
        preprocess,
        csv_files_with_labels=(train_csv_files, train_csv_labels),
    )

    # 训练模型
    print("开始训练...")
    train_acc, valid_acc = model.fit(train_loader, valid_loader, epoch_size=10, alpha=0.1)
    print(f"训练集准确率: {train_acc:.4f}")
    print(f"验证集准确率: {valid_acc:.4f}")

    # 保存模型
    model.get_model().eval()
    saved_models_path = get_saved_model_path(model_name+'-'+signal_process_method+'-'+model_save_filename)
    torch.save(model.get_model().state_dict(), saved_models_path)
    print(f"模型已保存到: {saved_models_path}")

    # 测试模型
    print("开始测试...")
    test_loader = get_dataloader(
        test_dataset,
        preprocess,
        split=False,
        csv_files_with_labels=(test_csv_files, test_csv_labels),
    )
    confusion_matrix, test_acc = model.test(test_loader)
    print(f"测试集准确率: {test_acc:.4f}")
    print("混淆矩阵:")
    print(confusion_matrix)

    return {
        "saved_path":saved_models_path,
        "train_acc": train_acc,
        "valid_acc": valid_acc,
        "test_acc": test_acc,
        "confusion_matrix": confusion_matrix,
    }


# "FC""RNN""GRU""LSTM""BiLSTM""ResNet18""ResNet34""ResNet50""LP_RNN"
# 信号处理: raw, mean_filter, hampel_filter, hampel_then_mean_filter, fft, wavelet, emd
if __name__ == "__main__":
    # 示例使用
    # 获取当前时间
    now_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time()))
    model_save_filename = f"model_{now_time}.pth"
    config = {
        "model_name": "GRU",  # 选择模型
        "signal_process_method": "wavelet",  # 信号处理
        "feature_type": "振幅",  # 特征类型
        "train_dataset": "自定义",  # 训练数据集
        "test_dataset": "自定义",  # 测试数据集
        "model_save_filename": model_save_filename,  # 保存模型文件名
        "train_csv_files": [
            "./saved_files/data/train/read_from_serial_2025-04-02_22-13-46_0_train.csv",
            "./saved_files/data/train/read_from_serial_2025-04-02_22-29-40_1_train.csv",
        ],
        "train_csv_labels": [0, 1],
        "test_csv_files": [
            "./saved_files/data/test/read_from_serial_2025-04-02_22-13-46_0_test.csv",
            "./saved_files/data/test/read_from_serial_2025-04-02_22-29-40_1_test.csv",
        ],
        "test_csv_labels": [0, 1],
    }

    # 训练并获取结果
    results = train_model(**config)

    # 使用测试集准确率重命名模型文件
    test_acc = results["test_acc"]
    old_path = get_saved_model_path(config["model_name"]+'-'+config["signal_process_method"]+'-'+model_save_filename)
    new_filename = f"model_{now_time}_acc_{test_acc:.4f}.pth"
    new_path = get_saved_model_path(new_filename)

    if os.path.exists(old_path):
        os.rename(old_path, new_path)
        print(f"\n模型已重命名为: {new_filename}")
