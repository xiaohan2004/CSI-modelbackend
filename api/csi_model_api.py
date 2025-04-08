import os
import tempfile

from flask import Blueprint, send_file
from flask import request, jsonify
from werkzeug.utils import secure_filename

from signalProducer import *
from support_web import single_signal_preprocess_to_matrix_preprocess
from tmp.train_model import get_signal_processor
from tool.model_create_api_utils import check_create_model_form_data_valid, check_model_name, \
    check_signal_processing_method, check_model_feature, save_train_file, utils_create_model
from tool.model_predict_api_utils import check_predict_json_form_data_valid, Predictor
from tool.model_signal_process_api_utils import check_signal_processing_json_data, check_signal_producer, \
    handle_signal_producer, process_csi_file
from tool.my_dp import db_tool

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
    file_0_list = request.files.getlist("train_file_0")
    file_1_list = request.files.getlist("train_file_1")
    test_file_0_list = request.files.getlist("test_file_0")
    test_file_1_list = request.files.getlist("test_file_1")

    print(len(file_0_list))
    print(len(file_1_list))
    print(len(test_file_0_list))
    print(len(test_file_1_list))



    flag, json = check_model_name(model_name)
    if not flag:
        return json

    flag, json = check_signal_processing_method(signal_processing_method)
    if not flag:
        return json

    flag, json = check_model_feature(model_feature)
    if not flag:
        return json

    # TODO: 根据标签保存训练文件
    for index, file_0 in enumerate(file_0_list):
        flag, json, file_0_path = save_train_file(file_0, 0, r"tool/label/train/label_0", index + 1)
        if not flag:
            return json
    for index, file_1 in enumerate(file_1_list):
        flag, json, file_1_path = save_train_file(file_1, 1, r"tool/label/train/label_1", index + 1)
        if not flag:
            return json


    # TODO: 根据标签保存测试文件
    for index, file_0 in enumerate(test_file_0_list):
        flag, json, file_0_path = save_train_file(file_0, 0, r"tool/label/test/label_0", index + 1)
        if not flag:
            return json
    for index, file_1 in enumerate(test_file_1_list):
        flag, json, file_1_path = save_train_file(file_1, 1, r"tool/label/test/label_1", index + 1)
        if not flag:
            return json


    json = utils_create_model(
        model_name=model_name,
        signal_processing_method=signal_processing_method,
        model_feature=model_feature,
        model_saved_name=model_saved_name,
        train_file_0_dir_path=r"tool/label/train/label_0",
        train_file_1_dir_path=r"tool/label/train/label_1",
        test_file_0_dir_path=r"tool/label/test/label_0",
        test_file_1_dir_path=r"tool/label/test/label_1"
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
        # 检查必需的表单字段
        required_fields = ['signal_process_method']
        for field in required_fields:
            if field not in request.form:
                return jsonify({"error": f"缺少必需字段: {field}"}), 400

        # 获取处理方法
        signal_process_method = request.form['signal_process_method']

        # 检查文件
        if 'file' not in request.files:
            return jsonify({"error": "未提供文件"}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({"error": "未选择文件"}), 400

        if not file.filename.endswith('.csv'):
            return jsonify({"error": "仅支持CSV格式文件"}), 400

        # 保存上传的文件到临时目录
        temp_dir = tempfile.mkdtemp()
        filename = secure_filename(file.filename)
        file_path = os.path.join(temp_dir, filename)
        file.save(file_path)

        # 处理文件
        processed_data = process_csi_file(file_path, signal_process_method)

        # 创建结果文件
        result_filename = f"processed_{filename}"
        result_path = os.path.join(temp_dir, result_filename)

        # 保存处理结果
        np.savetxt(result_path, processed_data, delimiter=',', fmt='%.4f')

        # 返回文件
        return send_file(
            result_path,
            as_attachment=True,
            download_name=result_filename,
            mimetype='text/csv'
        )

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"处理失败: {str(e)}"}), 500
    finally:
        # 清理临时文件
        if 'temp_dir' in locals():
            for filename in os.listdir(temp_dir):
                file_path = os.path.join(temp_dir, filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(f"无法删除临时文件 {file_path}: {e}")
            try:
                os.rmdir(temp_dir)
            except Exception as e:
                print(f"无法删除临时目录 {temp_dir}: {e}")











