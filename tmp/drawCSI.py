import abc
import socket
import struct
import threading
from collections import deque
from typing import Callable, Optional

import numpy as np
import matplotlib.pyplot as plt

from train_model import get_signal_processor
from support_web import single_signal_preprocess_to_matrix_preprocess


class ReceiveCSI(abc.ABC):
    """
    用于接收CSI数据的基类。
    基本使用方法是：
    1. 创建一个ReceiveCSI的实现子类的实例
    2. 设置处理CSI数据的回调函数
    3. 调用start方法开始接收CSI数据
    4. 调用stop方法停止接收CSI数据
    """

    def __init__(self):
        self.handle_csi = None

    def set_handle_csi(self, handle_csi):
        """
        设置处理CSI数据的回调函数。
        给外部调用，用于设置处理CSI数据的回调函数。
        :param handle_csi: 处理CSI数据的回调函数，函数原型为handle_csi(csi)，csi为CSI数据，
            csi的数据为一个字典，包含如下键值：{
            timestamp: int, 时间戳，单位为microsecond
            csi: np.array, CSI数据，为一个一维数组，形状为(子载波数量[64]), 数据类型为complex64
            }
        """
        self.handle_csi = handle_csi

    @abc.abstractmethod
    def start(self):
        """
        开始接收CSI数据，给方法实现子类实现。要求不能阻塞线程。可以使用多线程。
        """
        pass

    @abc.abstractmethod
    def stop(self):
        """
        停止接收CSI数据，给方法实现子类实现。要求不能阻塞线程。
        """
        pass

    def _send_one_csi(self, csi):
        """
        内部接口，用于提供给实现子类调用，传递一个CSI数据
        """
        if self.handle_csi:
            self.handle_csi(csi)


class ReadFromUDP(ReceiveCSI):
    def __init__(self, port, ip=""):
        """
        port: int, 接收CSI数据的端口号
        ip: str, 接收CSI数据的本机IP地址，主要针对多网卡设备。默认为空字符串，一般不需要设置。
        """
        super().__init__()
        self.port = port
        self.ip = ip
        self.running = False

    def start(self):
        """
        开始接收CSI数据
        """

        def receive():
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            client_address = (self.ip, self.port)
            sock.bind(client_address)
            print("start receiving")
            try:
                while self.running:
                    data, address = sock.recvfrom(4096)
                    if data != b"HEART" and len(data) == 132:
                        timestamp = struct.unpack("<I", data[:4])[0]
                        csi_raw = struct.unpack("128b", data[4:])
                        csi = np.zeros(64, dtype=np.complex64)
                        for i in range(0, 128, 2):
                            csi[i // 2] = csi_raw[i + 1] + csi_raw[i] * 1j
                        self._send_one_csi({"timestamp": timestamp, "csi": csi})
            finally:
                sock.close()
                print("stop receiving")

        self.running = True
        threading.Thread(target=receive).start()

    def stop(self):
        """
        停止接收CSI数据
        """
        self.running = False


if __name__ == "__main__":
    # 创建一个ReadFromUDP的实例，端口号为1234
    receiver = ReadFromUDP(1234)

    # 初始化存储最近30次CSI数据的队列
    csi_history = deque(maxlen=100)

    # 获取信号处理器
    signal_process_method = "wavelet"
    # signal_process_method = "mean_filter"
    signal_processor = get_signal_processor(signal_process_method)
    signal_preprocess = single_signal_preprocess_to_matrix_preprocess(
        signal_processor.process
    )

    # 绘制CSI处理后的数据随时间变化的函数
    def plot_csi_history():
        plt.clf()  # 清空当前图形

        if not csi_history:
            return
        if len(csi_history) < 100:
            return

        # 预处理数据
        csidata = list(csi_history)
        csi_matrix = np.array([np.abs(item["csi"]) for item in csidata])
        processed_matrix = signal_preprocess(csi_matrix.copy())
        print(f"processed_matrix shape: {processed_matrix.shape}")
        # print(processed_matrix)
        # processed_matrix = csi_matrix

        # 获取时间戳作为x轴
        timestamps = [item["timestamp"] for item in csi_history]
        # 转换为相对时间（微秒为单位）
        timestamps = np.array(timestamps) - timestamps[0]
        # 转换为秒
        timestamps = timestamps / 1e6

        # 绘制64条线，每条线代表一个子载波的幅度随时间变化
        for subcarrier in range(64):
            plt.plot(
                timestamps, processed_matrix[:, subcarrier], label=f"Sub {subcarrier}"
            )

        plt.title(f"CSI {signal_process_method} over Time (30 samples)")
        plt.ylim(0, 40)
        plt.xlabel("Time (s)")
        plt.ylabel(signal_process_method.capitalize())
        plt.grid(True)

        plt.tight_layout()
        plt.pause(0.01)  # 短暂暂停以更新图形

    # 设置处理CSI数据的回调函数
    def handle_csi(csi):
        # 将新数据添加到历史队列
        csi_history.append(csi)
        # 绘制历史数据
        plot_csi_history()
        print(f"Received CSI: {csi['timestamp']}")

    receiver.set_handle_csi(handle_csi)

    # 开始接收CSI数据
    receiver.start()

    # 等待用户输入，用户输入任意字符后停止接收CSI数据
    input()

    # 停止接收CSI数据
    receiver.stop()
    print("stop")
