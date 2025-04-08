import csv
import os.path
from datetime import datetime

import streamlit as st
from flask import jsonify

import support_web
from abstract import SignalProcessorBase, SignalProducerBase
from signalProcessorInstance import *
from signalProducer import *
import streamlit_extra_global_data
from support_web import find_implementations
from tmp.train_model import get_signal_processor


def check_signal_processing_json_data(json_data):
    if (json_data is None or
        json_data.get("signal_producer") is None or
        json_data.get("signal_process_method") is None or
        json_data.get("feature_type") is None):
        return False, jsonify({'error': 'Missing parameters in json'})
    return True, None

def check_signal_producer(signal_producer):
    supported_producers_class = find_implementations(SignalProducerBase)
    supported_producers_labels = [cls.get_label() for cls in supported_producers_class]
    if signal_producer not in supported_producers_labels:
        return False, jsonify({'error': 'Invalid signal producer', 'supported_producers': supported_producers_labels})
    else:
        return True, None


@st.cache_data
def get_signal_class_by_label(classes, label):
    for cls in classes:
        if cls.get_label() == label:
            return cls
    return None

def create_signal_reader():
    def reader(signal, *args, **kwargs):
        streamlit_extra_global_data.RECEIVE_S.append(signal)

    return reader

def create_signal_csv_writer(file_path):
    parent_dir = os.path.dirname(file_path)
    file_name = os.path.basename(file_path)
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    real_file_name = file_name.split(".")[0] + "_" + current_time + ".csv"

    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    real_file_path = os.path.join(parent_dir, real_file_name)

    def writer(signal, *args, **kwargs):
        with open(real_file_path, "a") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(signal)

    return writer


def parse_complex_string(s):
    """安全解析复数字符串"""
    try:
        if s == '0j':
            return 0 + 0j
        return complex(s)
    except (ValueError, SyntaxError):
        return 0 + 0j  # 解析失败时返回0


def process_csi_file(file_path, signal_process_method):
    """处理CSI文件的核心函数（针对特定格式优化）"""
    try:
        # 读取文件数据
        if file_path.endswith('.csv'):
            # 直接读取整个文件内容（假设每行是一个完整的CSI样本）
            with open(file_path, 'r') as f:
                lines = f.readlines()

            # 解析每行的CSI数据
            csi_data = []
            for line in lines:
                # 移除换行符和空格，分割字符串
                items = line.strip().replace(' ', '').split(',')
                # 解析每个复数
                complex_row = [parse_complex_string(item) for item in items]
                csi_data.append(complex_row)

            csi_matrix = np.array(csi_data, dtype=np.complex64)
        else:
            raise ValueError("仅支持CSV格式文件")

        # 验证数据维度
        if csi_matrix.shape[1] != 64:
            raise ValueError(f"CSI数据应包含64个子载波，实际得到{csi_matrix.shape[1]}个")

        # 获取信号处理器
        signal_processor = get_signal_processor(signal_process_method)
        signal_preprocess = support_web.single_signal_preprocess_to_matrix_preprocess(
            signal_processor.process
        )

        # 处理数据（取绝对值或其他处理）
        processed_matrix = signal_preprocess(np.abs(csi_matrix))  # 根据需求可能需要修改

        return processed_matrix
    except Exception as e:
        raise ValueError(f"文件处理失败: {str(e)}")

def handle_signal_producer(signal_producer,
        signal_process_method,
        signal_producer_file_path,
        signal_file_path):
    supported_producers_class = find_implementations(SignalProducerBase)

    producer = streamlit_extra_global_data.PRODUCER
    if producer is None or producer.get_label() != signal_producer or (
            producer.get_label() == SignalProducerFromFile.get_label() and producer.file != signal_producer_file_path):
        cls = get_signal_class_by_label(supported_producers_class, signal_producer)
        if producer is not None:
            producer.stop()
        if cls == SignalProducerFromFile:
            if os.path.exists(signal_producer_file_path):
                streamlit_extra_global_data.PRODUCER = support_web.get_producer(cls, signal_producer_file_path)
            else:
                return jsonify({'error': 'ERROR file not exist'})
        else:
            streamlit_extra_global_data.PRODUCER = support_web.get_producer(cls)

        streamlit_extra_global_data.PRODUCER.start()
        producer = streamlit_extra_global_data.PRODUCER

        # 清除之前的处理器
        if "processor" in st.session_state:
            st.session_state.__delitem__("processor")

    signal_processor_methods = find_implementations(SignalProcessorBase)

    signal_processor = get_signal_class_by_label(signal_processor_methods, signal_process_method)
    if signal_processor is None:
        return jsonify({'error': 'Invalid signal process method'})

    if ("processor" not in st.session_state or
            st.session_state.processor.get_label() != signal_process_method):
        st.session_state.processor = signal_processor()
        producer.clear_readers()
        producer.register_reader(create_signal_reader())
        if signal_file_path is not None:
            producer.register_reader(create_signal_csv_writer(signal_file_path))
        print("REGISTER", producer.readers)

    return jsonify({'success': 'Signal producer and processor registered'})


