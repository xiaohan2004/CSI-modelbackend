## 用于web页面的支持函数以及全局变量
import inspect
import os
import typing

import numpy as np
import streamlit as st
import torch

import models.LP_RNN
import utils
from abstract import SignalReaderBase
from models import support
from models.All import Model

import logging

# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# 常量定义
SAVED_MODELS_PATH = "./saved_models"
SAVED_CSV_PATH = "./saved_csv"
SUPPORTED_MODELS = ["FC", "RNN", "GRU", "LSTM", "BiLSTM", "ResNet18", "ResNet34", "ResNet50", "LP_RNN"]
SUPPORTED_FEATURES = ['CSI功率', '振幅', 'CSI相位', "DFS", 'CSI功率+振幅', 'CSI功率+相位']
N_CLASS = 2
SUPPORTED_DATASET = ["环境1", "环境2", "自定义1", "自定义2"] + ["自定义"]


class _GLOBAL_STATE:
    import utils

    RECEIVE_S = utils.CsiTime(max_size=50)
    HAS_MAN = False
    PRODUCER = None

    SELECTED_MODEL = None


GLOBAL_STATE = _GLOBAL_STATE()


def get_saved_model_path(file_name):
    if not os.path.exists(SAVED_MODELS_PATH):
        os.makedirs(SAVED_MODELS_PATH)
    return f"{SAVED_MODELS_PATH}/{support.LEN_W}-{support.STEP_DISTANCE}-{file_name}"


def get_saved_csv_path(file_name):
    if not os.path.exists(SAVED_CSV_PATH):
        os.makedirs(SAVED_CSV_PATH)
    return f"{SAVED_CSV_PATH}/{file_name}"


def get_producer(cls, *args, **kwargs):
    if not hasattr(get_producer, "RUNNING_MAP"):
        get_producer.RUNNING_MAP = {}

    if cls in get_producer.RUNNING_MAP and not get_producer.RUNNING_MAP[cls].is_stopped():
        get_producer.RUNNING_MAP[cls].stop()
        while not get_producer.RUNNING_MAP[cls].is_stopped():
            print("waiting for producer to stop", cls)
            pass

    get_producer.RUNNING_MAP[cls] = cls(*args, **kwargs)
    return get_producer.RUNNING_MAP[cls]


def get_class_by_label(classes, label):
    for cls in classes:
        if cls.get_label() == label:
            return cls
    return None


def single_signal_preprocess_to_matrix_preprocess(preprocess):
    def multi_signal_preprocess(signals, *args, **kwargs):
        return np.apply_along_axis(preprocess, 0, signals, *args, **kwargs)

    return multi_signal_preprocess


def create_signal2features_preprocess(features):
    def small_wave_preprocess(signal):
        import pywt
        fs = 100  # 采样频率
        care_fs = 2  # 关心的频率上限
        base_len = 40  # 返回的长度
        # Continuous wavelet transform
        wavename = 'cmor3-3'
        totalscal = base_len * fs / care_fs
        Fc = pywt.central_frequency(wavename)  # Center frequency of the wavelet
        c = 2 * Fc * totalscal
        scales = c / np.arange(1, totalscal + 1)
        coefs, _ = pywt.cwt(signal, scales, wavename, 1 / fs)
        return coefs[:base_len].T

    def small_wave_preprocess_for_m(m):
        m = [small_wave_preprocess(m[:, i]) for i in range(m.shape[1])]
        ret = np.zeros((m[0].shape[0], m[0].shape[1] * len(m)))
        for i in range(len(m)):
            ret[:, i * m[0].shape[1]: (i + 1) * m[0].shape[1]] = m[i]
        return ret

    funcs = {
        "CSI功率": lambda x: utils.CSI.get_amplitude_db_unit(x),
        "振幅": lambda x: utils.CSI.get_amplitude(x),
        "DFS": lambda x: small_wave_preprocess_for_m(utils.CSI.get_amplitude(x)),
        "CSI相位": lambda x: utils.CSI.get_phase(x),
        "CSI功率+振幅": lambda x: np.concatenate([utils.CSI.get_amplitude_db_unit(x), utils.CSI.get_amplitude(x)],
                                                 axis=1),
        "CSI功率+相位": lambda x: np.concatenate([utils.CSI.get_amplitude_db_unit(x), utils.CSI.get_phase(x)],
                                                 axis=1)
    }
    if features in funcs:
        return funcs[features]
    elif features in SUPPORTED_FEATURES:
        raise NotImplementedError("Not implemented for :" + features)
    else:
        raise ValueError("Unknown features")


@st.cache_data
def get_last_dim_size(features, common_size):
    if features in ["CSI功率", "振幅", "CSI相位"]:
        return common_size * 1
    elif features in ["CSI功率+振幅", "CSI功率+相位"]:
        return common_size * 2
    elif features in ["DFS"]:
        return common_size * 20
    elif features in SUPPORTED_FEATURES:
        raise NotImplementedError("Not implemented for :" + features)
    else:
        raise ValueError("Unknown features")


def create_model(model_name, last_dim_size) -> Model:
    global N_CLASS
    n_classes = N_CLASS
    import models.support as support
    from models.All import SimpleMLP, SimpleLSTM, SimpleBiLSTM, SimpleGRU, SimpleRNN, SimpleResNet18, SimpleResNet34, \
        SimpleResNet50
    if model_name == "FC":
        model = SimpleMLP(last_dim_size=last_dim_size, num_classes=n_classes)
    elif model_name == "RNN":
        model = SimpleRNN(last_dim_size=last_dim_size, num_classes=n_classes)
    elif model_name == "GRU":
        model = SimpleGRU(last_dim_size=last_dim_size, num_classes=n_classes)
    elif model_name == "LSTM":
        model = SimpleLSTM(last_dim_size, n_classes, 64)
    elif model_name == "BiLSTM":
        model = SimpleBiLSTM(last_dim_size, n_classes, 64)
    elif model_name == "ResNet18":
        model = SimpleResNet18(support.LEN_W)
    elif model_name == "ResNet34":
        model = SimpleResNet34(support.LEN_W)
    elif model_name == "ResNet50":
        model = SimpleResNet50(support.LEN_W)
    elif model_name == "LP_RNN":
        model = models.LP_RNN.SimpleLP_RNN(last_dim_size=last_dim_size, num_classes=n_classes)
    elif model_name in SUPPORTED_MODELS:
        raise NotImplementedError("Not implemented for :" + model_name)
    else:
        raise ValueError("Unknown model")
    return model


def get_dataloader(domain, preprocess, split=True, csv_files_with_labels=None):
    domain_map = {
        "环境1": 2,
        "环境2": 3,
    }
    custom_domain = {
        "自定义1": 1,
        "自定义2": 2,
    }
    import models.support as support
    if domain in domain_map:
        return support.get_dataloader(domain=domain_map[domain],
                                      split=split,
                                      preprocess=preprocess,
                                      data_path_prefix=".")
    elif domain == "自定义1":
        return get_dataloader_from_csv((
            ["saved_csv/restroom_no_people_at_near.csv",
             "saved_csv/restroom_no_people_at_near.csv",
             "saved_csv/restroom_no_people_at_near.csv",
             "saved_csv/restroom_no_people_at_near.csv",
             "saved_csv/restroom_no_people_at_near.csv",
             "saved_csv/restroom_has_people_at_near.csv",
             "saved_csv/has_people-read_from_serial_2024-11-28_09-18-29.csv",
             "saved_csv/has_people-read_from_serial_2024-11-28_09-25-42.csv"],
            [0, 0, 0, 0, 0, 1, 1, 1]),
            split=split,
            preprocess=preprocess)
    elif domain == "自定义2":
        return get_dataloader_from_csv(
            (["saved_csv/no_people-read_from_serial_2024-11-28_18-30-44.csv",
              "saved_csv/has_people-read_from_serial_2024-11-28_09-18-29.csv",
              "saved_csv/has_people-read_from_serial_2024-11-28_09-25-42.csv"],
             [0, 1, 1]),
            split=split,
            preprocess=preprocess)
    elif domain == SUPPORTED_DATASET[-1]:
        return get_dataloader_from_csv(csv_files_with_labels, split=split, preprocess=preprocess)
    elif domain in SUPPORTED_DATASET:
        raise NotImplementedError("Not implemented for :" + domain)
    else:
        raise ValueError("Unknown domain")


def get_dataloader_from_csv(csv_files_with_labels, split=True, step_distance=support.STEP_DISTANCE, preprocess=None,
                            batch_size=support.BATCH_SIZE,
                            data_path_prefix="."):
    import scipy
    import numpy as np
    from torch.utils.data import TensorDataset, random_split, DataLoader

    LEN_W = support.LEN_W
    # 从csv文件中读取数据到numpy矩阵
    raw_data = (
        [np.loadtxt(f"{data_path_prefix}/{csv_file}", delimiter=',', dtype=np.complex64) for csv_file in
         csv_files_with_labels[0]],
        csv_files_with_labels[1])

    data = []
    labels = []

    for (m, label) in zip(raw_data[0], raw_data[1]):
        for i in range(0, m.shape[0] - LEN_W, step_distance):
            matrix = m[i:i + LEN_W:, :]
            data.append(matrix)
            labels.append(int(label))

    if preprocess is not None:
        data = [preprocess(x) for x in data]
    # print(f"data:{data}")
    # _ = input("pause")
    data = torch.tensor(data, dtype=torch.float32)
    print(data.shape)
    labels = torch.tensor(labels, dtype=torch.long)

    dataset = TensorDataset(data, labels)

    if split:
        # Split the dataset into training and testing sets
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        # 创建 DataLoader
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        return train_dataloader, test_dataloader
    else:
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)


class SimpleSignalReader(SignalReaderBase):
    def __init__(self, receiver):
        self.receiver = receiver

    def read(self, signal: np.ndarray, *args: typing.Any, **kwargs: typing.Any):
        self.receiver(signal, *args, **kwargs)


def show_receive_s(matrix=None):
    import matplotlib.pyplot as plt
    import seaborn as sns
    if matrix is None:
        matrix = GLOBAL_STATE.RECEIVE_S.try_get_corrected_csi_matrix()

    # Create a heatmap using seaborn
    fig, ax = plt.subplots()
    matrix = np.abs(matrix)
    # 对matrix进行归一化
    matrix = np.where(np.isnan(matrix), 0, matrix)
    matrix = 255 * (matrix - matrix.min()) / (matrix.max() - matrix.min() + 1)
    sns.heatmap(np.abs(matrix), ax=ax, cmap="coolwarm", cbar=False, xticklabels=False, yticklabels=False)
    # Display the heatmap in Streamlit
    st.pyplot(fig)


def create_simple_reader_to_write_global_receive_s():
    def receiver(signal, *args, **kwargs):
        packet = kwargs.get('packet')
        if packet is not None:
            tsf = utils.CSI.get_tsf_from_pack(packet)
            GLOBAL_STATE.RECEIVE_S.append_with_time(signal, tsf)
        else:
            GLOBAL_STATE.RECEIVE_S.append(signal)

    return SimpleSignalReader(receiver=receiver)


def find_implementations(abstraction: typing.Type) -> [typing.Type]:
    implementations = []
    for subclass in abstraction.__subclasses__():
        if not inspect.isabstract(subclass):
            implementations.append(subclass)
        implementations.extend(find_implementations(subclass))
    return implementations


def is_same_label_of_key_in_session(key, label):
    return key in st.session_state and st.session_state[key].get_label() == label


def create_preprocess_chain(processors):
    def preprocess_chain(signal):
        for p in processors:
            if p is not None:
                signal = p(signal)
        return signal

    return preprocess_chain


if __name__ == '__main__':
    import signalProcessorInstance

    #
    # impl = find_implementations(SignalReaderBase)
    # print(impl)
    # print([i.get_label() for i in impl])
    # pass
    get_dataloader("自定义", lambda x: x,
                   csv_files_with_labels=(["./saved_csv/read_from_serial_2024-11-11_10-42-52.csv"], [1]))
