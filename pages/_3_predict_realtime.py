import os.path

import streamlit as st
import torch

import streamlit_extra_global_data
import support_web
from abstract import SignalProcessorBase, SignalProducerBase
# 导入所有信号处理算法的类，否则无法通过反射找到这些类
from signalProcessorInstance import *
from signalProducer import *
from models import support

from support_web import get_saved_model_path, create_model, get_last_dim_size, create_signal2features_preprocess, \
    SUPPORTED_MODELS, \
    SUPPORTED_FEATURES


def get_reader():
    return support_web.create_simple_reader_to_write_global_receive_s()


def clear_state():
    if "model" in st.session_state:
        del st.session_state["model"]
    if "preprocess" in st.session_state:
        del st.session_state["preprocess"]
    if support_web.GLOBAL_STATE.PRODUCER is not None:
        support_web.GLOBAL_STATE.PRODUCER.stop()
        support_web.GLOBAL_STATE.PRODUCER = None


def main():
    st.title("实时预测")
    if 'predicting' not in st.session_state:
        st.session_state.predicting = False
    supported_producers_classes = support_web.find_implementations(SignalProducerBase)
    select_producer = st.sidebar.selectbox("信号来源", [x.get_label() for x in supported_producers_classes if
                                                        x.get_label() != SignalProducerFromFile.get_label()])
    select_model = st.sidebar.selectbox("选择模型", SUPPORTED_MODELS)
    signal_processor_methods = support_web.find_implementations(SignalProcessorBase)
    select_method_name = st.sidebar.selectbox("信号处理算法", [x.get_label() for x in signal_processor_methods])
    select_features = st.sidebar.selectbox("选择特征", SUPPORTED_FEATURES)
    select_model_save_filename = st.sidebar.text_input("待加载模型文件名", select_model + ".pth")
    if st.session_state.predicting:
        if st.sidebar.button("STOP"):
            st.session_state.predicting = False
            clear_state()
            st.rerun()
    else:
        if st.sidebar.button("开始预测"):
            if not os.path.exists(get_saved_model_path(select_model_save_filename)):
                st.sidebar.error(f"{get_saved_model_path(select_model_save_filename)}模型文件不存在")
                return
            st.session_state.predicting = True
            st.rerun()

    # webrtc_streamer(
    #     key="example",
    #     mode=WebRtcMode.SENDRECV,
    #     media_stream_constraints={"video": True, "audio": False},
    # )
    st.write("因为跳板机的限制，无法使用摄像头")
    # st.line_chart(streamlit_extra_global_data.RECEIVE_S.try_get_corrected_csi_matrix()[-support.LEN_W:, 0])

    if st.session_state.predicting:
        if "model" not in st.session_state:
            select_model_save_path = get_saved_model_path(select_model_save_filename)
            st.session_state.model = create_model(select_model, get_last_dim_size(select_features, 64))
            st.session_state.model.get_model().load_state_dict(torch.load(select_model_save_path, weights_only=True))
            st.session_state.model.get_model().eval()
        if "preprocess" not in st.session_state:
            # 去噪算法
            denoise_process_cls = support_web.get_class_by_label(signal_processor_methods, select_method_name)
            denoise_preprocesses = [denoise_process_cls()] if denoise_process_cls is not None else []
            denoise_preprocesses = [support_web.single_signal_preprocess_to_matrix_preprocess(x) for x in
                                    denoise_preprocesses]
            # 特征提取算法
            features_preprocess = create_signal2features_preprocess(select_features)
            # 生成预处理过程链
            st.session_state.preprocess = support_web.create_preprocess_chain(
                denoise_preprocesses + [features_preprocess])
        if streamlit_extra_global_data.PRODUCER is None or (
                streamlit_extra_global_data.PRODUCER.get_label() != select_producer):
            if streamlit_extra_global_data.PRODUCER is not None:
                streamlit_extra_global_data.PRODUCER.stop()
            cls = support_web.get_class_by_label(supported_producers_classes, select_producer)
            streamlit_extra_global_data.PRODUCER = support_web.get_producer(cls)
            streamlit_extra_global_data.PRODUCER.register_reader(get_reader())
            streamlit_extra_global_data.PRODUCER.start()
        # predict
        model = st.session_state.model
        preprocess = st.session_state.preprocess
        s = support_web.GLOBAL_STATE.RECEIVE_S.try_get_corrected_csi_matrix()
        s = s[-support.LEN_W::]
        s = preprocess(s)
        s = s.reshape(1, *s.shape)
        s = torch.tensor(s).float()
        result = model.get_model()(s).detach().numpy()
        support_web.GLOBAL_STATE.predicted_result = result
        st.write("预测结果", result)
        print(result)
        support_web.show_receive_s()
        time.sleep(0.4)
        st.rerun()


if __name__ == '__main__':
    main()
