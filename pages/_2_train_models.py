import os

import streamlit as st
import torch

import support_web
from abstract import SignalProcessorBase
from signalProcessorInstance import *
from support_web import get_saved_model_path, SUPPORTED_FEATURES, SUPPORTED_MODELS, get_last_dim_size, \
    create_signal2features_preprocess, create_model, SUPPORTED_DATASET, get_dataloader, find_implementations


def main():
    torch.manual_seed(123)

    st.title("模型训练")
    if 'training' not in st.session_state:
        st.session_state.training = False
    if 'trained' not in st.session_state:
        st.session_state.trained = False

    # UI
    select_model = st.selectbox("选择模型", SUPPORTED_MODELS)
    signal_processor_methods = find_implementations(SignalProcessorBase)
    select_method_name = st.selectbox("信号处理算法", [x.get_label() for x in signal_processor_methods])
    select_features = st.selectbox("选择特征", SUPPORTED_FEATURES)
    select_train_data = st.selectbox("选择训练数据", SUPPORTED_DATASET)
    select_train_csv_files = []
    select_train_csv_labels = []
    if select_train_data == SUPPORTED_DATASET[-1]:
        support_web.get_saved_csv_path(".")  # 让 saved_csv_path 一定存在
        # 这里提供多个选择框，用于选择csv文件，然后每个选择框后面还有一个输入框用于输入文件的label数字，这样可以方便的选择多个文件
        if "need_train_csv_file_num" not in st.session_state:
            st.session_state.need_train_csv_file_num = 2
        for i in range(st.session_state.need_train_csv_file_num):
            col1, col2 = st.columns(2)
            select_train_csv_files.append(support_web.SAVED_CSV_PATH + "/" +
                                          col1.selectbox(f"选择第{i + 1}个训练csv文件",
                                                         os.listdir(support_web.SAVED_CSV_PATH)))
            select_train_csv_labels.append(col2.text_input(f"第{i + 1}个训练csv文件的label", 0))

        col1, col2 = st.columns(2)
        if col1.button("Add", key="add-test"):
            st.session_state.need_train_csv_file_num += 1
            st.rerun()
        if col2.button("Sub", key="sub-test"):
            st.session_state.need_train_csv_file_num -= 1
            st.rerun()
    select_test_data = st.selectbox("选择测试数据", SUPPORTED_DATASET)
    select_test_csv_files = []
    select_test_csv_labels = []
    if select_test_data == SUPPORTED_DATASET[-1]:
        support_web.get_saved_csv_path(".")  # 让 saved_csv_path 一定存在
        if "need_test_csv_file_num" not in st.session_state:
            st.session_state.need_test_csv_file_num = 2
        for i in range(st.session_state.need_test_csv_file_num):
            col1, col2 = st.columns(2)
            select_test_csv_files.append(support_web.SAVED_CSV_PATH + "/" +
                                         col1.selectbox(f"选择第{i + 1}个测试csv文件",
                                                        os.listdir(support_web.SAVED_CSV_PATH)))
            select_test_csv_labels.append(int(col2.text_input(f"第{i + 1}个测试csv文件的label", 0)))

        col1, col2 = st.columns(2)
        if col1.button("Add"):
            st.session_state.need_test_csv_file_num += 1
            st.rerun()
        if col2.button("Sub"):
            st.session_state.need_test_csv_file_num -= 1
            st.rerun()
    select_model_save_filename = st.text_input("模型保存文件名", select_model + ".pth")
    if st.button("开始训练"):
        st.session_state.training = True
        st.session_state.trained = False
        st.rerun()

    # real training
    if st.session_state.training:
        with st.spinner("PS: 此处为了避免性能消耗，已经将全部的训练简化为1轮，训练后性能无参考意义。\n"+"开始训练，这个过程可能需要一段时间，请耐心等待。中途请勿操作页面"):
            info = st.empty()
            # 去噪算法
            denoise_process_cls = support_web.get_class_by_label(signal_processor_methods, select_method_name)
            denoise_preprocesses = [denoise_process_cls()] if denoise_process_cls is not None else []
            denoise_preprocesses = [support_web.single_signal_preprocess_to_matrix_preprocess(x) for x in
                                    denoise_preprocesses]
            # 特征提取算法
            features_preprocess = create_signal2features_preprocess(select_features)
            # 生成预处理链
            preprocess = support_web.create_preprocess_chain(denoise_preprocesses + [features_preprocess])

            last_dim_size = get_last_dim_size(select_features, 64)
            model = create_model(select_model, last_dim_size)

            # train
            train_loader, valid_loader = get_dataloader(select_train_data, preprocess, csv_files_with_labels=(
                select_train_csv_files, select_train_csv_labels))

            def info_print(x):
                buffer = getattr(info, "buffer", "")
                buffer = f"{buffer}\n{x}"
                info.write(buffer + x)
                info.write(x)

            last_acc, last_valid_acc = model.fit(train_loader, valid_loader)
            # save the model
            model.get_model().eval()
            saved_models_path = get_saved_model_path(select_model_save_filename)
            torch.save(model.get_model().state_dict(), saved_models_path)
            model.get_model().load_state_dict(torch.load(saved_models_path, weights_only=True))
            # test
            info.write("开始预测测试集")
            test_loader = get_dataloader(select_test_data, preprocess, split=False,
                                         csv_files_with_labels=(select_test_csv_files, select_test_csv_labels))
            cm, last_test_acc = model.test(test_loader)

            # record the results
            st.session_state.last_acc = last_acc
            st.session_state.last_valid_acc = last_valid_acc
            st.session_state.last_test_acc = last_test_acc
            st.session_state.cm = cm
            st.session_state.trained = True
            st.session_state.training = False
            st.rerun()
    if st.session_state.trained:
        st.write("训练完成")
        st.write("训练集准确率", st.session_state.last_acc)
        st.write("验证集准确率", st.session_state.last_valid_acc)
        st.write("测试集准确率", st.session_state.last_test_acc)
        st.write("测试集混淆矩阵", st.session_state.cm)
        if st.button("返回"):
            st.session_state.trained = False
            st.rerun()


if __name__ == '__main__':
    main()
