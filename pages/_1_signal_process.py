import time
from datetime import datetime

import numpy as np
import streamlit as st

import support_web
from abstract import SignalProcessorBase, SignalProducerBase
from signalProcessorInstance import *
from signalProducer import *
import streamlit_extra_global_data
from support_web import find_implementations


@st.cache_data
def get_class_by_label(classes, label):
    for cls in classes:
        if cls.get_label() == label:
            return cls
    return None


def create_reader():
    def reader(signal, *args, **kwargs):
        streamlit_extra_global_data.RECEIVE_S.append(signal)

    return reader


# @st.cache_data
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
            writer = csv.writer(f)
            writer.writerow(signal)

    return writer


def main():
    # 页面内容布局
    st.title("信号处理")
    supported_producers_class = find_implementations(SignalProducerBase)
    select_producer = st.sidebar.selectbox("信号来源", [x.get_label() for x in supported_producers_class])
    select_producer_file_path = None
    if select_producer == SignalProducerFromFile.get_label():
        select_producer_file_path = st.sidebar.text_input("文件路径", "./data/short_no_people.txt")
    select_channel = st.sidebar.selectbox("显示的子载波", range(-1, 64))
    signal_processor_methods = find_implementations(SignalProcessorBase)
    select_method_name = st.sidebar.selectbox("信号处理算法", [x.get_label() for x in signal_processor_methods])
    select_feature = st.sidebar.selectbox("显示的信号特征", ['振幅', 'CSI功率'])
    select_data_record_filename = None
    if select_producer != SignalProducerFromFile.get_label():
        select_data_record_filename = st.sidebar.text_input("原始信号记录文件",
                                                            select_producer.replace(" ", "_") + ".csv")
    graph = st.empty()

    # 创建信号来源对象
    producer = streamlit_extra_global_data.PRODUCER
    if producer is None or producer.get_label() != select_producer or (
            producer.get_label() == SignalProducerFromFile.get_label() and producer.file != select_producer_file_path):
        cls = get_class_by_label(supported_producers_class, select_producer)
        if producer is not None:
            producer.stop()

        if cls == SignalProducerFromFile:
            if os.path.exists(select_producer_file_path):
                streamlit_extra_global_data.PRODUCER = support_web.get_producer(cls, select_producer_file_path)
            else:
                st.write("ERROR file not exist")
                return
        else:
            streamlit_extra_global_data.PRODUCER = support_web.get_producer(cls)
        streamlit_extra_global_data.PRODUCER.start()
        producer = streamlit_extra_global_data.PRODUCER
        # 清除之前的处理器
        # 重新执行处理器创建注册，便于注册新的reader
        if "processor" in st.session_state:
            st.session_state.__delitem__("processor")

    # 创建信号处理对象
    select_signal_processor = get_class_by_label(signal_processor_methods, select_method_name)
    if select_signal_processor is None:
        st.write("ERROR unknow method" + select_method_name)
        return

    # if not support_web.is_same_label_of_key_in_session("processor", select_method_name):
    if "processor" not in st.session_state or st.session_state.processor.get_label() != select_method_name:
        st.session_state.processor = select_signal_processor()
        producer.clear_readers()
        producer.register_reader(create_reader())
        if select_data_record_filename is not None:
            producer.register_reader(create_csv_writer(support_web.get_saved_csv_path(select_data_record_filename)))
        print("REGISTER", producer.readers)

    # 不断更新图像
    for _ in range(1):
        # while True:
        s = streamlit_extra_global_data.RECEIVE_S
        s = s.get_csi_matrix().copy()
        s = np.where(np.isnan(s), 0, s)
        sel_care_feature = lambda x: np.abs(x) if select_feature == "振幅" else 20 * np.log(np.abs(x) + 1e-7)
        if select_channel != -1:
            graph.line_chart(sel_care_feature(
                np.apply_along_axis(st.session_state.processor, 0, s)[:, select_channel]))
        else:
            graph.line_chart(sel_care_feature(
                np.apply_along_axis(st.session_state.processor, 0, s)[:, :]))
        time.sleep(2)
    # 因为不停的循环，无法消除上一次的图像，所以采取循环一段时间，然后重新加载页面的方式
    # 实现不间断的图像更新st
    st.rerun()


if __name__ == '__main__':
    main()
