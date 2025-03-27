import logging

import numpy as np
import torch
import json

from sktime import split

from models import support


def get_data_loader_from_memory(
    processed_data,
    train_test_rate=0.8,
    step_distance=support.STEP_DISTANCE,
    preprocess=None,
    batch_size=support.BATCH_SIZE,
    split=True,
):
    """直接从内存数据生成DataLoader"""
    from torch.utils.data import TensorDataset, random_split, DataLoader
    import numpy as np
    import torch

    LEN_W = support.LEN_W
    data = []
    labels = []

    # 首先按label分组数据
    grouped_data = {}
    for m, label in zip(processed_data["data"], processed_data["labels"]):
        label = int(label)
        if label not in grouped_data:
            grouped_data[label] = []
        grouped_data[label].append(m)

    # 对每个label的数据进行拼接并提取窗口
    for label, matrices in grouped_data.items():
        # 将同一label的所有矩阵在时间维度上拼接
        combined_matrix = np.concatenate(matrices, axis=0)
        print(f"Label {label} 拼接后的矩阵形状: {combined_matrix.shape}")

        # 提取滑动窗口
        for i in range(0, combined_matrix.shape[0] - LEN_W + 1, step_distance):
            matrix = combined_matrix[i : i + LEN_W, :]
            data.append(matrix)
            labels.append(label)

    # 如果有预处理函数，则对数据进行预处理
    if preprocess is not None:
        data = [preprocess(x) for x in data]

    # 检查是否有有效的样本
    if len(data) == 0:
        print("没有提取到有效的样本，请检查数据维度和LEN_W的值。")
        return None, None

    # 将数据转换为tensor
    data = torch.tensor(data, dtype=torch.float32)
    print(f"最终数据形状: {data.shape}")
    labels = torch.tensor(labels, dtype=torch.long)
    print(f"标签分布: {np.bincount(labels.numpy())}")

    # 创建数据集
    dataset = TensorDataset(data, labels)

    # 如果需要划分训练集和测试集
    if split:
        train_size = int(train_test_rate * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        # 创建 DataLoader
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        # 打印数据形状信息
        for train_batch in train_dataloader:
            train_data_batch, train_labels_batch = train_batch
            print(f"训练集一个批次的数据形状: {train_data_batch.shape}")
            print(f"训练集一个批次的标签形状: {train_labels_batch.shape}")
            break

        return train_dataloader, test_dataloader
    else:
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def extract_csi_processed(json_obj):
    """
    从Flask请求的JSON数据中提取csiData和processed字段
    :param json_obj: 解析后的JSON对象（字典）
    :return: 包含(csi_data, processed)的列表
    """
    data_list = []
    try:
        # 提取原始数据列表
        raw_data_list = json_obj["train_set"]["data"]["rawDataList"]

        for item in raw_data_list:
            # 解析嵌套的csiData字符串为字典
            csi_data = json.loads(item["csiData"])
            processed = item["processed"]

            data_list.append({"csiData": csi_data, "processed": processed})

        return data_list
    except KeyError as e:
        print(f"字段缺失错误: {e}")
        return []
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {e}")
        return []


import numpy as np
import logging


def process_data_string(train_set):
    """将客户端发送的JSON数据转换为内存中的训练数据格式"""
    processed = {"data": [], "labels": []}

    for item in train_set:
        try:
            # 直接使用解析后的csiData（已通过extract_csi_processed解析）
            csi_data = item["csiData"]

            # 转换为复数数组
            real = np.array(csi_data["real"], dtype=np.complex64)
            imag = np.array(csi_data["imag"], dtype=np.complex64)

            # 组合成复数矩阵（这里假设子载波数为64）
            complex_data = real + 1j * imag
            time_steps = len(complex_data) // 64  # 计算时间步数

            # 重塑为(time_steps, 64)的矩阵
            csi_matrix = complex_data.reshape(time_steps, 64)

            processed["data"].append(csi_matrix)
            processed["labels"].append(item["processed"])  # 使用processed作为标签

        except Exception as e:
            logging.error(f"数据处理失败: {str(e)}")
            continue

    return processed


def print_loader(loader):
    for batch in loader:
        print(batch)
