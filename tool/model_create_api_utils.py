import uuid

from flask import jsonify
from signalProcessorInstance import *
from support_web import SUPPORTED_MODELS, find_implementations, SUPPORTED_FEATURES

import os

from tmp.train_model import train_model
from tool.my_dp import db_tool


def check_create_model_form_data_valid(form_data):
    """
    检查 form-data 中是否存在需要的参数：
    :return:
    """
    # 检查 form_data 是否为 None，以及是否包含所需的键值对
    if (form_data is None or
            form_data.get("model_name") is None or
            form_data.get("signal_processing_method") is None or
            form_data.get("model_feature") is None or
            form_data.get("model_saved_name") is None):
        return False
    return True

def check_model_name(model_name):
    """
    检查 model_name 是否符合要求
    :param model_name:
    :return:
    """
    if model_name not in SUPPORTED_MODELS:
        return jsonify({'error': 'Unsupported model name','supported_models': SUPPORTED_MODELS})
    return True, None


def check_signal_processing_method(signal_processing_method):
    """
    检查 signal_processing_method 是否符合要求
    :param signal_processing_method:
    :return:
    """
    support_signal_processing_methods = find_implementations(SignalProcessorBase)
    support_signal_processing_labels = [x.get_label() for x in support_signal_processing_methods]
    if signal_processing_method not in support_signal_processing_labels:
        return False, jsonify({'error': 'Unsupported signal processing method','supported_signal_processing_methods': support_signal_processing_labels})
    return True, None

def check_model_feature(model_feature):
    """
    检查 model_feature 是否符合要求
    :param model_feature:
    :return:
    """
    if model_feature not in SUPPORTED_FEATURES:
        return False, jsonify({'error': 'Unsupported model feature','supported_features': SUPPORTED_FEATURES})
    return True, None

def save_train_file(file, file_label, file_path, file_index):
    """
    保存训练数据文件
    :param file:
    :param file_label:
    :param file_path:
    :return:
    """
    if file is None:
        return False, jsonify({'error': 'No file provided'}), None
    if file_label not in [0, 1]:
        return False, jsonify({'error': 'Invalid file label'}), None
    # 按照标签构造对应文件地址
    save_file_path = f"{file_path}/{file_label}-{file_index}.csv"
    try:
        # 文件存在则直接替换
        if os.path.exists(save_file_path):
            os.remove(save_file_path)
        file.save(save_file_path)
        return True, None, save_file_path
    except Exception as e:
        print(e)
        return False, jsonify({'error': f'Failed to save file: {str(e)}'}), None

def get_csv_files_info(dir_path):
    csv_files = [f for f in os.listdir(dir_path) if f.endswith('.csv')]
    csv_file_paths = [os.path.join(dir_path, f) for f in csv_files]
    csv_file_count = len(csv_files)
    return csv_file_paths, csv_file_count


def collect_files_and_labels(dir_paths, labels):
    file_paths = []
    file_labels = []
    for dir_path, label in zip(dir_paths, labels):
        csv_file_paths, csv_file_count = get_csv_files_info(dir_path)
        file_paths.extend(csv_file_paths)
        file_labels.extend([label] * csv_file_count)
    return file_paths, file_labels

def utils_create_model(model_name,
                       signal_processing_method,
                       model_feature,
                       model_saved_name,
                       train_file_0_dir_path,
                       train_file_1_dir_path,
                       test_file_0_dir_path,
                       test_file_1_dir_path):

    train_file_paths, train_labels = collect_files_and_labels(
        [train_file_0_dir_path, train_file_1_dir_path], [0, 1]
    )

    test_file_paths, test_labels = collect_files_and_labels(
        [test_file_0_dir_path, test_file_1_dir_path], [0, 1]
    )

    model_uuid = str(uuid.uuid4())
    model_saved_filename = f"{model_saved_name}_{model_uuid}"

    res = train_model(
        model_name=model_name,
        signal_process_method=signal_processing_method,
        feature_type=model_feature,
        model_save_filename=model_saved_filename,
        train_dataset="自定义",
        test_dataset="自定义",
        train_csv_files=train_file_paths,
        train_csv_labels=train_labels,
        test_csv_files=test_file_paths,
        test_csv_labels=test_labels,
    )

    if res is None:
        return jsonify({"error": "模型训练失败"})

    try:
        db_tool.insert_record(model_uuid, model_name, res["saved_path"])
    except Exception as e:
        print(e)
        return jsonify({"error": "数据库插入失败"})

    json = jsonify(
        {
            "msg": "模型训练成功",
            "model_uuid": model_uuid,
            "model_name": model_name,
            "model_basic_test": {
                "train_acc": res['train_acc'],
                "valid_acc": res["valid_acc"],
                "test_acc": res["test_acc"],
                "confusion_matrix": res["confusion_matrix"].tolist(),
            }
        }
    )

    return json


def main():
    json = utils_create_model(
        model_name="GRU",
        signal_processing_method="wavelet",
        model_feature="振幅",
        model_saved_name="model_saved_name",
        train_file_0_dir_path=r"label/train/label_0",
        train_file_1_dir_path=r"label/train/label_1",
        test_file_0_dir_path=r"label/test/label_0",
        test_file_1_dir_path="label/test/label_1",
    )
    print(json)

if __name__ == "__main__":
    main()

