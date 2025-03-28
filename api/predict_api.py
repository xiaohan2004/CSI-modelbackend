import torch
from flask import request, jsonify, Blueprint

from models import support
from receiveCSI import LLTF_VALID_INDEX
from signalProcessorInstance import *
from signalProducer_完整版本 import *
from support_web import (
    create_signal2features_preprocess,
    create_model,
    get_last_dim_size,
    create_preprocess_chain,
    single_signal_preprocess_to_matrix_preprocess,
    get_class_by_label,
    find_implementations,
    SUPPORTED_FEATURES,
)
from tool.my_dp import db_tool

predict_bp = Blueprint('predict_model', __name__)


class Predictor:
    def __init__(
        self,
        model_name,
        model_path,
        signal_process_method,
        feature_type,
        window_size=support.LEN_W,
    ):
        """
        初始化文件预测器

        参数:
            model_name: 模型名称
            model_path: 已训练模型的文件路径
            signal_process_method: 信号处理方法名称
            feature_type: 特征类型
            window_size: 信号窗口大小
        """
        self.window_size = window_size

        # 加载模型
        self.model = create_model(model_name, get_last_dim_size(feature_type, 64))
        self.model.get_model().load_state_dict(
            torch.load(model_path, weights_only=True)
        )
        self.model.get_model().eval()

        # 设置信号处理和特征提取
        signal_processor_methods = find_implementations(SignalProcessorBase)
        denoise_process_cls = get_class_by_label(
            signal_processor_methods, signal_process_method
        )
        denoise_preprocesses = (
            [denoise_process_cls()] if denoise_process_cls is not None else []
        )
        denoise_preprocesses = [
            single_signal_preprocess_to_matrix_preprocess(x)
            for x in denoise_preprocesses
        ]

        # 特征提取
        features_preprocess = create_signal2features_preprocess(feature_type)
        self.preprocess = create_preprocess_chain(
            denoise_preprocesses + [features_preprocess]
        )

    def read_csi_from_file(self, file_path):
        """从CSV文件读取CSI数据"""
        csi_data = []
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:  # 跳过空行
                    # 解析复数字符串，格式如："1.23+4.56j"
                    complex_numbers = [complex(x) for x in line.split(",")]
                    csi_data.append(complex_numbers)
        return np.array(csi_data)

    def predict(self, file_path):
        """预测指定文件中的CSI数据"""
        # 读取数据
        csi_data = self.read_csi_from_file(file_path)

        # 处理无效子载波
        for i in range(len(csi_data)):
            invalid_index = [j for j in range(0, 64) if j not in LLTF_VALID_INDEX]
            csi_data[i][invalid_index] = 0

        # 如果数据量不足window_size，则无法预测
        if len(csi_data) < self.window_size:
            return {
                "error": f"数据量不足，需要至少 {self.window_size} 组数据，当前只有 {len(csi_data)} 组"
            }

        predictions = []
        # 使用滑动窗口进行预测
        for i in range(0, len(csi_data) - self.window_size + 1):
            window_data = csi_data[i: i + self.window_size]

            # 预处理信号
            processed_signal = self.preprocess(window_data)
            processed_signal = processed_signal.reshape(1, *processed_signal.shape)

            # 转换为tensor并预测
            tensor_signal = torch.tensor(processed_signal).float()
            with torch.no_grad():
                prediction = self.model.get_model()(tensor_signal).numpy()

            # 记录预测结果
            predicted_class = int(np.argmax(prediction[0]))  # 转换为int类型
            confidence = float(prediction[0][predicted_class])  # 确保可以被JSON序列化
            full_prediction = [float(x) for x in prediction[0]]  # 转换为float列表
            predictions.append(
                {
                    "class": predicted_class,
                    "confidence": confidence,
                    "full_prediction": full_prediction,
                }
            )

        # 统计各类别的预测次数
        class_counts = {}
        for p in predictions:
            class_counts[p["class"]] = class_counts.get(p["class"], 0) + 1

        # 将 class_counts 的键转换为 int 类型
        class_counts = {int(k): v for k, v in class_counts.items()}

        # 输出最终预测结果（投票结果）
        most_common_class = int(max(class_counts.items(), key=lambda x: x[1])[0])  # 转换为int类型

        return {
            "predictions": predictions,
            "final_prediction": most_common_class,
            "class_counts": class_counts
        }

    def predict_from_array(self, csi_data):
        """
        直接从CSI数组进行预测

        参数:
            csi_data: numpy数组，形状为(n, 64)的CSI数据，每行是一个时间点的64个子载波数据

        返回:
            predictions: 预测结果列表
        """
        # 处理无效子载波
        for i in range(len(csi_data)):
            invalid_index = [j for j in range(0, 64) if j not in LLTF_VALID_INDEX]
            csi_data[i][invalid_index] = 0

        # 如果数据量不足window_size，则无法预测
        if len(csi_data) < self.window_size:
            return {
                "error": f"数据量不足，需要至少 {self.window_size} 组数据，当前只有 {len(csi_data)} 组"
            }

        predictions = []
        # 使用滑动窗口进行预测
        for i in range(0, len(csi_data) - self.window_size + 1):
            window_data = csi_data[i: i + self.window_size]

            # 预处理信号
            processed_signal = self.preprocess(window_data)
            processed_signal = processed_signal.reshape(1, *processed_signal.shape)

            # 转换为tensor并预测
            tensor_signal = torch.tensor(processed_signal).float()
            with torch.no_grad():
                prediction = self.model.get_model()(tensor_signal).numpy()

            # 记录预测结果
            predicted_class = int(np.argmax(prediction[0]))  # 转换为int类型
            confidence = float(prediction[0][predicted_class])  # 确保可以被JSON序列化
            full_prediction = [float(x) for x in prediction[0]]  # 转换为float列表
            predictions.append(
                {
                    "class": predicted_class,
                    "confidence": confidence,
                    "full_prediction": full_prediction,
                }
            )

        # 统计各类别的预测次数
        class_counts = {}
        for p in predictions:
            class_counts[p["class"]] = class_counts.get(p["class"], 0) + 1

        # 将 class_counts 的键转换为 int 类型
        class_counts = {int(k): v for k, v in class_counts.items()}

        # 输出最终预测结果（投票结果）
        most_common_class = int(max(class_counts.items(), key=lambda x: x[1])[0])  # 转换为int类型

        return {
            "predictions": predictions,
            "final_prediction": most_common_class,
            "class_counts": class_counts
        }


@predict_bp.route('/api/predict', methods=['POST'])
def predict_api():
    # 从 form-data 中获取 JSON 数据
    json_data_str = request.form.get('json_data')
    if not json_data_str:
        return jsonify({"error": "未提供JSON数据"})
    try:
        data = eval(json_data_str)
    except Exception:
        return jsonify({"error": "JSON数据格式错误"})

    # 从JSON数据中获取配置信息
    model_uuid = data.get('model_uuid', '')
    if model_uuid is None or model_uuid == '':
        return jsonify({"error": "未提供模型UUID"})
    models = db_tool.select_record(model_uuid)
    if models is None:
        return jsonify({"error": "未找到指定UUID的模型"})
    model = models[0]
    model_name = model["model_name"]
    model_path = model["model_saved_path"]

    signal_process_method = data.get('signal_process_method', 'mean_filter')
    select_processor_methods = find_implementations(SignalProcessorBase)
    select_method_name = [x.get_label() for x in select_processor_methods]
    if signal_process_method is None or signal_process_method not in select_method_name:
        return jsonify({"error": "未提供有效的信号处理方法", "tips": f"支持的信号处理方法有：{select_method_name}"})
    feature_type = data.get('feature_type', '振幅')
    if feature_type not in SUPPORTED_FEATURES:
        return jsonify({"error": "未提供有效的特征类型", "tips": f"支持的特征类型有：{SUPPORTED_FEATURES}"})

    # 创建预测器实例
    predictor = Predictor(
        model_name=model_name,
        model_path=model_path,
        signal_process_method=signal_process_method,
        feature_type=feature_type,
    )

    file = request.files.get('file')
    if file:
        # 保存文件
        file_path = 'temp_predict.csv'
        file.save(file_path)
        result = predictor.predict(file_path)
        return jsonify(result)
    else:
        return jsonify({"error": "未提供文件"})
