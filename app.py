import uuid

import torch
from flask import Flask, request, jsonify
from sklearn.model_selection import train_test_split
import traceback

import support_web


from tools import extract_csi_processed, process_data_string, get_data_loader_from_memory
from signalProcessorInstance import *
from support_web import SUPPORTED_MODELS, find_implementations, SUPPORTED_FEATURES

app = Flask(__name__)

@app.route('/api', methods=['POST'])
def api():
    try:
        data = request.get_json()
        # 在这里处理数据，并返回结果
        return jsonify({'result': 'success'})
    except Exception as e:
        error_message = f"An unexpected error occurred: {str(e)}"
        trace = traceback.format_exc()
        print(trace)
        return jsonify({'result': 'error', 'message': error_message})

@app.route('/api/create-model-114514', methods=['POST'])
def create_model_api():
    try:
        required_fields = ['model_name', 'signal_process_method', 'feature_type', 'model_definite_name', 'train_test_rate',
                           'train_set']
        data = request.get_json()
        # 校验必要字段是否存在
        for field in required_fields:
            if field not in data:
                return jsonify({'result': 'error', 'message': f'Missing required field: {field}'})

        model_name = data['model_name']
        signal_process_method = data['signal_process_method']

        # 校验模型名称
        if model_name not in SUPPORTED_MODELS:
            return jsonify({'result': 'error', 'message': 'Model not found'})

        # 校验信号处理方法
        signal_processor_methods = find_implementations(SignalProcessorBase)
        valid_methods = [x.get_label() for x in signal_processor_methods]
        if signal_process_method not in valid_methods:
            return jsonify({
                'result': 'error',
                'message': f'Invalid signal process method. Valid methods: {valid_methods}'
            })

        feature_type = data['feature_type']
        # 校验特征类型
        if feature_type not in SUPPORTED_FEATURES:
            return jsonify({'result': 'error', 'message': 'Feature type not found'})

        # 保存的代码名称
        model_definite_name = data['model_definite_name']

        # 训练集划分参数
        train_test_rate = data['train_test_rate']

        # 训练集
        train_set = extract_csi_processed(data)

        # 处理训练集

        processed_data = process_data_string(train_set)

        if len(processed_data['data']) == 0:
            return jsonify({'result': 'error', 'message': 'No valid training data'})

        train_loader, valid_loader = get_data_loader_from_memory(processed_data, train_test_rate)

        # 创建模型
        denoise_process_cls = support_web.get_class_by_label(signal_processor_methods, signal_process_method)
        denoise_preprocesses = [denoise_process_cls()] if denoise_process_cls is not None else []
        denoise_preprocesses = [support_web.single_signal_preprocess_to_matrix_preprocess(x) for x in
                                denoise_preprocesses]

        features_preprocess = support_web.create_signal2features_preprocess(feature_type)

        preprocess = support_web.create_preprocess_chain(denoise_preprocesses + [features_preprocess])

        last_dim_size = support_web.get_last_dim_size(feature_type, 64)

        model = support_web.create_model(model_name, last_dim_size)


        # 模型能够创建，但是不能进行验证，原因是维度不匹配
        # last_acc, last_valid_acc = model.fit(train_loader, valid_loader)

        model.get_model().eval()

        saved_models_path = support_web.get_saved_model_path(model_definite_name)
        torch.save({
            'model_state_dict': model.get_model().state_dict(),
        }, saved_models_path)

        # 从保存的字典中提取实际的state_dict
        checkpoint = torch.load(saved_models_path)
        model.get_model().load_state_dict(checkpoint['model_state_dict'])

        # 手动生成model uuid
        model_uuid = str(uuid.uuid4())
        return jsonify({'result': 'success',
                        'message': 'Model created successfully',
                        'model-save-path': f"{saved_models_path}",
                        'model-id': model_uuid})



    except ValueError as ve:
        error_message = f"Value error: {str(ve)}"
        trace = traceback.format_exc()
        print(trace)
        return jsonify({'result': 'error', 'message': error_message})
    except KeyError as ke:
        error_message = f"Key error: {str(ke)}"
        trace = traceback.format_exc()
        print(trace)
        return jsonify({'result': 'error', 'message': error_message})
    except Exception as e:
        error_message = f"An unexpected error occurred: {str(e)}"
        trace = traceback.format_exc()
        print(trace)
        return jsonify({'result': 'error', 'message': error_message})


if __name__ == '__main__':
    app.run(debug=True, port=5000)
