import tempfile

import torch
from flask import Blueprint, request, jsonify, send_file
from tool.my_dp import db_tool
import support_web
from signalProcessorInstance import *
from support_web import SUPPORTED_FEATURES, SUPPORTED_MODELS, find_implementations
from tool.csi_tools import read_raw_data_between_time, saved_csi_data_with_label_to_different_file, load_data_from_csv

create_model_bp = Blueprint('create_model', __name__ )

@create_model_bp.route('/api/get_model', methods=['GET'])
def get_model_api():
    model_uuid = request.args.get("uuid")
    if not model_uuid:
        return "Missing 'uuid' parameter", 400
    try:
        model_list = db_tool.select_record(model_uuid)
        if not model_list:
            return "Model not found", 404
        # 取列表的第一个元素，这是一个字典
        model = model_list[0]
        model_path = model["model_saved_path"]
        return send_file(model_path, as_attachment=True)
    except Exception as e:
        return f"Error: {str(e)}", 500

@create_model_bp.route('/api/delete_model', methods=['GET'])
def delete_model_api():
    msg = "success delete model"
    delete_uuid = request.args.get("uuid")
    try:
        model_list = db_tool.select_record(delete_uuid)
        if not model_list:
            msg = "model not exist"
        else:
            model = model_list[0]
            model_path = model["model_saved_path"]
            if os.path.exists(model_path):
                os.remove(model_path)
                db_tool.delete_record(delete_uuid)
    except Exception as e:
        msg = f"error delete model {e}"
    return jsonify({"msg": msg})
@create_model_bp.route('/api/create_model', methods=['POST'])
def create_model_api():
    msg = "success create model"
    tips = []
    json = request.get_json()
    # 获取模型名称
    model_name = json.get("model_name", "FC")
    if model_name not in SUPPORTED_MODELS:
        msg = "model name not supported"
        tips.append(f"model name not supported : {SUPPORTED_MODELS}")
        return jsonify({"msg": msg, "tips": tips})

    # 获取信号处理方法
    signal_processing_method = json.get("signal_processing_method", "none")
    # 修改为查找 SignalProcessorBase 的实现类
    exist_signal_processing_method = find_implementations(SignalProcessorBase)
    exist_signal_processing_method_name = [x.get_label() for x in exist_signal_processing_method]
    if signal_processing_method not in exist_signal_processing_method_name:
        msg = "signal processing method not supported"
        tips.append(f"signal processing method not supported : {exist_signal_processing_method_name}")
        return jsonify({"msg": msg, "tips": tips})
    # 处理特征
    model_feature = json.get("model_feature", "none")
    if model_feature not in SUPPORTED_FEATURES:
        msg = "model feature not supported"
        tips.append(f"model feature not supported : {SUPPORTED_FEATURES}")
        return jsonify({"msg": msg, "tips": tips})
    # 获取数据
    read_result = read_raw_data_between_time(json)
    if "error" in read_result:
        msg = "Error reading raw data"
        tips.append(read_result["error"])
    else:
        saved_csi_data_with_label_to_different_file(read_result, clean=True)

    # 加载数据
    train_set, valid_set = load_data_from_csv()
    if train_set is None or valid_set is None:
        msg = "Error loading data from CSV"
        tips.append("数据加载失败，请检查文件路径和文件内容。")
        return jsonify({"msg": msg, "tips": tips})
    # 去噪算法
    denoise_process_cls = (support_web.get_class_by_label
                           (exist_signal_processing_method, exist_signal_processing_method_name))
    denoise_preprocesses = [denoise_process_cls()] if denoise_process_cls is not None else []
    denoise_preprocesses = [support_web.single_signal_preprocess_to_matrix_preprocess(x) for x in
                            denoise_preprocesses]

    features_preprocess = support_web.create_signal2features_preprocess(model_feature)

    support_web.create_preprocess_chain(denoise_preprocesses + [features_preprocess])

    last_dim_size = support_web.get_last_dim_size(model_feature, 64)

    model = support_web.create_model(model_name, last_dim_size)

    model_save_name = json.get("model_saved_name", "undefined")

    saved_models_path = support_web.get_saved_model_path(model_save_name)

    tips.append(f"saved_if saved_models_path successfully: {saved_models_path}")

    try:
        # 获取系统临时目录
        temp_dir = tempfile.gettempdir()
        # 生成一个唯一的临时文件路径
        temp_model_path = tempfile.mktemp(suffix='.pth', dir=temp_dir)

        # 假设 model.fit 方法可以接受一个保存路径的参数
        last_acc, last_valid_acc = model.fit(train_set, valid_set, temp_model_path)
        tips.append(f"last_acc: {last_acc}")
        tips.append(f"last_valid_acc: {last_valid_acc}")
    except Exception as e:
        msg = "Error fitting the model"
        tips.append(str(e))

    model.get_model().eval()
    saved_models_path = support_web.get_saved_model_path(model_save_name)
    import uuid
    model_uuid = str(uuid.uuid4())

    torch.save(model.get_model().state_dict(), saved_models_path)
    model.get_model().load_state_dict(torch.load(saved_models_path, weights_only=True))

    if db_tool is not None:
        db_tool.insert_record(model_uuid, model_name,saved_models_path)

    # 返回 json
    return jsonify(
        {
            "msg": msg,
            "tips": tips,
            "model_uuid": model_uuid
        }
    )