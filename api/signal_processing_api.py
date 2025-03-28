import os.path
from flask import request, jsonify, Blueprint, current_app
from datetime import datetime
import streamlit as st
import support_web
from abstract import SignalProcessorBase, SignalProducerBase
from signalProcessorInstance import *
from signalProducer import *
from signalProducer_完整版本 import *
import streamlit_extra_global_data
from support_web import find_implementations
import numpy as np

# 创建蓝图
signal_processing_bp = Blueprint('signal_processing', __name__)

# 模拟 streamlit_extra_global_data
class StreamlitExtraGlobalData:
    def __init__(self):
        self.PRODUCER = None
        self.RECEIVE_S = []


streamlit_extra_global_data = StreamlitExtraGlobalData()


def get_class_by_label(classes, label):
    for cls in classes:
        if str(cls.get_label()) == label:
            return cls
    return None


def create_reader():
    def reader(signal, *args, **kwargs):
        streamlit_extra_global_data.RECEIVE_S.append(signal)
    return reader


def create_csv_writer(file_path):
    parent_dir = os.path.dirname(file_path)
    file_name = os.path.basename(file_path)
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    real_file_name = file_name.split(".")[0] + "_" + current_time + ".csv"
    real_file_path = os.path.join(parent_dir, real_file_name)

    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    def writer(signal, *args, **kwargs):
        with open(real_file_path, 'a') as f:
            import csv
            writer = csv.writer(f)
            writer.writerow(signal)
    return writer


import os
from flask import request


def handle_file_upload():
    if 'file' not in request.files:
        return None, "ERROR no file uploaded", "Please upload a file."
    file = request.files['file']
    if file.filename == '':
        return None, "ERROR empty file name", "Please select a valid file."
    try:
        file_content = file.read().decode('utf-8')
        if request.form.get('select_producer') == SignalProducerFromFile.get_label():
            file_path = 'signal_tmp/signal_process_tmp.txt'
        else:
            file_path = 'signal_tmp/signal_process_csv_tmp.csv'

        # 获取文件所在的目录
        directory = os.path.dirname(file_path)

        # 检查目录是否存在，如果不存在则创建
        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(file_path, 'w') as f:
            f.write(file_content)
        return file_path, None, None
    except Exception as e:
        return None, f"ERROR processing file: {str(e)}", "Please check the file format."


def validate_data(data):
    response = {
        "msg": [],
        "tips": [],
        "res": []
    }
    # 验证信号来源
    select_producer = data.get('select_producer')
    supported_producer_class = find_implementations(SignalProducerBase)
    supported_producer_labels = [cls.get_label() for cls in supported_producer_class]
    if select_producer is None or select_producer not in supported_producer_labels:
        response["msg"].append("ERROR unknown producer")
        response["tips"].append(f"Supported producers : {supported_producer_labels}")
        return response, None, None

    # 验证子载波
    select_channel = data.get('select_channel')
    if select_channel is None or not isinstance(select_channel, int) or select_channel not in range(-1, 64):
        response["msg"].append("ERROR invalid select_channel value")
        response["tips"].append("The 'select_channel' must be an integer between -1 and 63.")

    # 验证信号处理方法
    signal_processor_methods = data.get('signal_processor_methods')
    select_processor_methods = find_implementations(SignalProcessorBase)
    select_method_name = [x.get_label() for x in select_processor_methods]
    if signal_processor_methods is None or signal_processor_methods not in select_method_name:
        response["msg"].append("ERROR invalid signal_processor_methods value")
        response["tips"].append(f"The 'signal_processor_methods' must be a valid signal processor method. {select_method_name}")

    # 验证信号特征
    select_feature = data.get('select_feature')
    if select_feature is None or select_feature not in ['振幅', 'CSI功率']:
        response["msg"].append("ERROR invalid select_feature value")
        response["tips"].append("The 'select_feature' must be either '振幅' or 'CSI功率'.")

    return response, select_producer, select_processor_methods


@signal_processing_bp.route('/api/process_signal', methods=['POST'])
def process_signal():
    # 初始化消息和提示列表
    response = {
        "msg": [],
        "tips": [],
        "res": []
    }
    # 从 form-data 中获取 JSON 数据
    json_data_str = request.form.get('json_data')
    if not json_data_str:
        response["msg"].append("ERROR no valid JSON data in request")
        response["tips"].append("Please send valid JSON data in the 'json_data' field of form-data.")
        return jsonify(response)

    try:
        data = eval(json_data_str)
    except Exception as e:
        response["msg"].append(f"ERROR parsing JSON data: {str(e)}")
        response["tips"].append("Please ensure the 'json_data' field contains valid JSON data.")
        return jsonify(response)

    # 处理文件上传
    file_path, file_error_msg, file_error_tip = handle_file_upload()
    if file_error_msg:
        response["msg"].append(file_error_msg)
        response["tips"].append(file_error_tip)
        return jsonify(response)

    # 验证数据
    validation_response, select_producer, select_processor_methods = validate_data(data)
    response["msg"].extend(validation_response["msg"])
    response["tips"].extend(validation_response["tips"])
    if response["msg"]:
        return jsonify(response)

    streamlit_extra_global_data.selected_producer_label = select_producer

    producer = streamlit_extra_global_data.PRODUCER
    if producer is None or producer.get_label() != select_producer or (
            producer.get_label() == SignalProducerFromFile.get_label() and producer.file != file_path):
        cls = get_class_by_label(find_implementations(SignalProducerBase), select_producer)
        if producer is not None:
            producer.stop()
        if cls == SignalProducerFromFile:
            if os.path.exists(file_path):
                streamlit_extra_global_data.PRODUCER = support_web.get_producer(cls, file_path)
            else:
                response["msg"].append("ERROR file not exist")
                return jsonify(response)
        else:
            streamlit_extra_global_data.PRODUCER = support_web.get_producer(cls)
        streamlit_extra_global_data.PRODUCER.start()
        producer = streamlit_extra_global_data.PRODUCER
        if "processor" in st.session_state:
            st.session_state.__delitem__("processor")

    select_signal_processor = get_class_by_label(select_processor_methods, data.get('signal_processor_methods'))
    if select_signal_processor is None:
        response["msg"].append(f"ERROR unknown method")
        response["tips"].append(f"Supported methods : {[x.get_label() for x in select_processor_methods]}")
        return jsonify(response)

    if "processor" not in st.session_state or st.session_state.processor.get_label() != data.get('signal_processor_methods'):
        st.session_state.processor = select_signal_processor()
        producer.clear_readers()
        producer.register_reader(create_reader())
        if file_path:
            producer.register_reader(create_csv_writer(support_web.get_saved_csv_path(file_path)))
        print("REGISTER", producer.readers)

        # 处理信号
        s = streamlit_extra_global_data.RECEIVE_S
        if isinstance(s, list):
            s = np.array(s)
        s = s.get_csi_matrix().copy() if hasattr(s, 'get_csi_matrix') else s
        s = np.where(np.isnan(s), 0, s)
        select_feature = data.get('select_feature')
        sel_care_feature = lambda x: np.abs(x) if select_feature == "振幅" else 20 * np.log(np.abs(x) + 1e-7)

        result = np.apply_along_axis(st.session_state.processor, 0, s)
        select_channel = data.get('select_channel')
        if select_channel != -1:
            result = sel_care_feature(result[:, select_channel] if result.ndim == 2 else result).tolist()
        else:
            result = sel_care_feature(result if result.ndim == 1 else result[:, :]).tolist()

        response["res"] = result

    return jsonify(response)