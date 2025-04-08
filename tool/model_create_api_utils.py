import tempfile

from flask import jsonify
import torch
import support_web
from signalProcessorInstance import *
from support_web import SUPPORTED_MODELS, find_implementations, SUPPORTED_FEATURES
from tool.csi_tools import load_data_from_csv
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

def save_train_file(file, file_label, file_path):
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
    save_file_path = f"{file_path}/{file_label}.csv"
    try:
        # 文件存在则直接替换
        if os.path.exists(save_file_path):
            os.remove(save_file_path)
        file.save(save_file_path)
        return True, None, save_file_path
    except Exception as e:
        print(e)
        return False, jsonify({'error': f'Failed to save file: {str(e)}'}), None


def utils_create_model(model_name,
                       signal_processing_method,
                       model_feature,
                       model_saved_name,
                       file_0_path,
                       file_1_path):
    train_set, valid_set = load_data_from_csv(file_0_path, file_1_path)
    if train_set is None or valid_set is None:
        return jsonify({'error': 'Error loading data from CSV'})

    support_signal_processing_methods = find_implementations(SignalProcessorBase)
    support_signal_processing_labels = [x.get_label() for x in support_signal_processing_methods]

    # 去噪算法
    denoise_process_cls = (support_web.get_class_by_label
                           (support_signal_processing_methods, support_signal_processing_labels))
    denoise_preprocesses = [denoise_process_cls()] if denoise_process_cls is not None else []

    denoise_preprocesses = [support_web.single_signal_preprocess_to_matrix_preprocess(x) for x in
                            denoise_preprocesses]
    # 特征提取算法
    features_preprocess = support_web.create_signal2features_preprocess(model_feature)

    # 生成预处理过程链
    support_web.create_preprocess_chain(denoise_preprocesses + [features_preprocess])

    # 获取最后一维大小
    last_dim_size = support_web.get_last_dim_size(model_feature, 64)

    # 创建模型
    model = support_web.create_model(model_name, last_dim_size)

    if model is None:
        return jsonify({'error': 'Error creating model'})


    try:
        temp_dir = tempfile.gettempdir()
        temp_model_path = tempfile.mktemp(suffix='.pth', dir=temp_dir)
        last_acc, last_valid_acc = model.fit(train_set, valid_set)
    except Exception as e:
        print(e)

    model.get_model().eval()

    import uuid

    model_uuid = str(uuid.uuid4())

    model_unique_name = f"{model_name}_{model_uuid}"

    model_save_path = support_web.get_saved_model_path(model_unique_name)


    try:
        torch.save(model.get_model().state_dict(), model_save_path)
        model.get_model().load_state_dict(torch.load(model_save_path, weights_only=True))
    except Exception as e:
        print(e)
        return jsonify({'error': f'Error saving model: {str(e)}'})

    try:
        db_tool.insert_record(model_uuid, model_saved_name, model_save_path)
    except Exception as e:
        print(e)
        return jsonify({'error': f'Error inserting record to database: {str(e)}'})

    return jsonify({'msg': 'Model created successfully', 'model_uuid': model_uuid})
