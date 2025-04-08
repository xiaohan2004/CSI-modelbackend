import os
import zipfile

import numpy as np
from flask import Blueprint, jsonify, request, send_file
import streamlit as st
import streamlit_extra_global_data
from tool.model_api_utils import check_train_csv_json, delete_temp_zip
from tool.model_create_api_utils import check_model_feature
from tool.my_dp import db_tool
from tool.csi_tools import read_raw_data_between_time, saved_csi_data_with_label_to_different_file

model_api_bp = Blueprint('model_api', __name__)

@model_api_bp.route('/api/get-model-list', methods=['GET'])
def get_model_list():
    res_json = []
    try:
        model_list = db_tool.select_record()
        if model_list is None:
            return jsonify({'message': 'No models found'})
        for model in model_list:
            model_uuid = model["uuid"]
            model_name = model["model_name"]
            model_saved_path = model["model_saved_path"]
            model_insert_time = model["model_insert_time"]
            # 组合每个模型的信息为一个字典
            model_info = {
                "uuid": model_uuid,
                "model_name": model_name,
                "model_saved_path": model_saved_path,
                "model_insert_time": model_insert_time
            }
            # 将模型信息字典添加到结果列表中
            res_json.append(model_info)
        # 返回包含所有模型信息的列表
        return jsonify(res_json)
    except Exception as e:
        print(e)
        return jsonify({'message': 'An error occurred while retrieving the model list'})

@model_api_bp.route('/api/get-model-info', methods=['GET'])
def get_model_info():
    try:
        model_uuid = request.args.get('uuid')
        if model_uuid is None:
            return jsonify({'message': 'No model UUID provided'})
        models = db_tool.select_record(model_uuid)
        if models is None:
            return jsonify({'message': 'No model found'})
        else:
            model = models[0]
            model_uuid = model["uuid"]
            model_name = model["model_name"]
            model_saved_path = model["model_saved_path"]
            model_insert_time = model["model_insert_time"]
            # 组合每个模型的信息为一个字典
            model_info = {
                "uuid": model_uuid,
                "model_name": model_name,
                "model_saved_path": model_saved_path,
                "model_insert_time": model_insert_time
            }
            return jsonify(model_info)
    except Exception as e:
        print(e)
        return jsonify({'message': 'An error occurred while retrieving the model info'})

@model_api_bp.route('/api/delete-model', methods=['GET'])
def delete_model():
    try:
        model_uuid = request.args.get('uuid')
        if model_uuid is None:
            return jsonify({'message': 'No model UUID provided'})
        db_tool.delete_record(model_uuid)
        return jsonify({'message': 'Model deleted successfully'})
    except Exception as e:
        print(e)
        return jsonify({'message': 'An error occurred while deleting the model'})

@model_api_bp.route('/api/get-train-csv-file', methods=['POST'])
def get_train_csv_file():
    flag, json_data = check_train_csv_json(request.json)
    if not flag:
        return json_data
    read_result = read_raw_data_between_time(request.json)
    if read_result is None:
        return jsonify({"error": "raw data is not provided"})
    path_0 = r'tool/train_csv_tmp/label_0.csv'
    path_1 = r'tool/train_csv_tmp/label_1.csv'
    tmp_zip_file_path = r'tool/train_csv_tmp/train_files.zip'
    flag = saved_csi_data_with_label_to_different_file(read_result,
                                                       clean=True,
                                                       path_0=path_0,
                                                       path_1=path_1)
    if not flag:
        return jsonify({"error": "save raw data to csv failed"})

    if os.path.exists(path_0) and os.path.exists(path_1):
        # 创建一个临时的压缩文件
        zip_file_path = r'tool/train_csv_tmp/train_files.zip'
        with zipfile.ZipFile(zip_file_path, 'w') as zipf:
            zipf.write(path_0, os.path.basename(path_0))
            zipf.write(path_1, os.path.basename(path_1))

        # 返回压缩文件
        response = send_file(zip_file_path, as_attachment=True)

        # 在响应发送完成后删除临时压缩文件
        response.call_on_close(lambda: delete_temp_zip(response, zip_file_path))

        return response
    else:
        return jsonify({"error": "save raw data to csv failed"})

@model_api_bp.route('/api/query-process-signal', methods=['POST'])
def query_process_signal():
    try:
        channel = request.json.get("channel")
        if channel is None or channel not in range(-1, 65):
            return jsonify({'message': 'No channel provided or channel is not in range(-1, 65)'})
        select_feature = request.json.get("select_feature")
        flag, json = check_model_feature(select_feature)
        if not flag:
            return json
        s = streamlit_extra_global_data.RECEIVE_S
        s = s.get_csi_matrix().copy()
        s = np.where(np.isnan(s),0,s)
        sel_care_feature = lambda x: np.abs(x) if select_feature == "振幅" else 20 * np.log(np.abs(x) + 1e-7)
        if channel != -1:
            data = sel_care_feature(
                np.apply_along_axis(st.session_state.processor, 0, s)[:, channel])
        else:
            data = sel_care_feature(
                np.apply_along_axis(st.session_state.processor, 0, s)[:, :])
        data_list = data.tolist()
        return jsonify({'data': data_list})
    except Exception as e:
        print(e)
        return jsonify({'message': 'An error occurred while retrieving the process signal'})


