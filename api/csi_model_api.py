from flask import request, jsonify
from flask import Blueprint

from tool.model_create_api_utils import check_create_model_form_data_valid, check_model_name, \
    check_signal_processing_method, check_model_feature, save_train_file, utils_create_model
from tool.model_signal_process_api_utils import check_signal_processing_json_data, check_signal_producer, \
    handle_signal_producer

import time
from datetime import datetime

import numpy as np
import streamlit as st
from abstract import SignalProcessorBase, SignalProducerBase

import support_web
import streamlit_extra_global_data
from support_web import find_implementations
from signalProducer import *

from signalProcessorInstance import *

from tool.my_dp import db_tool
from tool.model_predict_api_utils import check_predict_json_form_data_valid, Predictor

csi_model_bp = Blueprint('csi_model_bp', __name__)

@csi_model_bp.route('/api/create-new-model', methods=['POST'])
def create_model():
    # 检查 form-data 中是否存在需要的参数：
    if not check_create_model_form_data_valid(request.form):
        return jsonify({'error': 'Missing parameters in form-data'})
    # 从 from-data 中获取参数
    model_name = request.form.get("model_name")
    signal_processing_method = request.form.get("signal_processing_method")
    model_feature = request.form.get("model_feature")
    model_saved_name = request.form.get("model_saved_name")
    file_0 = request.files.get("file_0")
    file_1 = request.files.get("file_1")

    flag, json = check_model_name(model_name)
    if not flag:
        return json

    flag, json = check_signal_processing_method(signal_processing_method)
    if not flag:
        return json

    flag, json = check_model_feature(model_feature)
    if not flag:
        return json

    # TODO: 根据标签保存文件
    flag, json, file_0_path = save_train_file(file_0, 0,r"tool/label")
    if not flag:
        return json
    flag, json, file_1_path = save_train_file(file_1, 1,r"tool/label")
    if not flag:
        return json

    json = utils_create_model(
        model_name=model_name,
        signal_processing_method=signal_processing_method,
        model_feature=model_feature,
        model_saved_name=model_saved_name,
        file_0_path=file_0_path,
        file_1_path=file_1_path
    )

    return json

@csi_model_bp.route('/api/predict-model', methods=['POST'])
def predict_model():
    flag, json = check_predict_json_form_data_valid(request.form)
    if not flag:
        return json
    model_uuid = request.form.get('model_uuid')
    signal_process_method = request.form.get('signal_process_method')
    feature_type = request.form.get('feature_type')
    file = request.files.get('predict_file')

    try:
        models = db_tool.select_record(model_uuid)
        if models is None or len(models) == 0:
            return jsonify({"error": "未找到指定UUID的模型"})
        model = models[0]
    except Exception as e:
        print(e)
        return jsonify({"error": "数据库查询错误"})
    model_name = model["model_name"]
    model_path = model["model_saved_path"]

    flag, json = check_signal_processing_method(signal_process_method)
    if not flag:
        return json

    flag, json = check_model_feature(feature_type)
    if not flag:
        return json

    predictor = Predictor(
        model_name=model_name,
        model_path=model_path,
        signal_process_method=signal_process_method,
        feature_type=feature_type,
    )

    if file is None:
        return jsonify({"error": "未提供文件"})
    tmp_predict_file_path = 'tool/train_csv_tmp/temp_predict.csv'
    file.save(tmp_predict_file_path)
    result = predictor.predict(tmp_predict_file_path)
    return jsonify({"msg":"Predict Success", "result":result})

@csi_model_bp.route('/api/process-signal', methods=['POST'])
def process_signal():
    try:
        flag, json = check_signal_processing_json_data(request.form)
        if not flag:
            return json
        signal_producer = request.form.get("signal_producer")
        signal_process_method = request.form.get("signal_process_method")
        feature_type = request.form.get("feature_type")
        signal_producer_file = request.files.get("signal_producer_file")

        signal_producer_file_path = r'tool/signal_producer_file/short_no_people.txt'
        signal_tmp_file_path = None
        if signal_producer == SignalProducerFromFile.get_label():
            if signal_producer_file is None:
                return jsonify({"error": "未提供文件"})
            try:
                signal_producer_file.save(signal_producer_file_path)
            except Exception as e:
                print(e)
                return jsonify({"error": "文件保存失败"})
        else:
            signal_tmp_file_path = 'tool/signal_file_tmp/temp_signal.csv'

        flag, json = check_signal_producer(signal_producer)
        if not flag:
            return json

        flag, json = check_signal_processing_method(signal_process_method)
        if not flag:
            return json
        flag, json = check_model_feature(feature_type)
        if not flag:
            return json

        json = handle_signal_producer(
            signal_producer,
            signal_process_method,
            signal_producer_file_path,
            signal_tmp_file_path
        )
        return json
    except Exception as e:
        print(e)
        return jsonify({"error": "处理失败"})











