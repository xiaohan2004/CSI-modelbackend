import csv
import json
import socket
import time
from io import StringIO

import numpy as np

import abstract
import utils


# 从文件中读取csi信号，并按照0.1s的间隔进行生成，模拟实时抓包。实现SignalProducerBase类
class SignalProducerFromFile(abstract.SignalProducerBase):
    def __init__(self, file):
        super().__init__()
        self.file = file
        packets = utils.read_udp_data_txt_to_bytes(file)
        self.csi = [utils.CSI.get_csi_vector_from_packet(x) for x in packets]
        self.index = 0
        self.stopped = False

    def _produce(self):
        while self.running and self.index < len(self.csi):
            self.send_signal(self.csi[self.index])
            self.index += 1
            time.sleep(0.09)
        self.stopped = True

    @staticmethod
    def get_label():
        return "read from file"


class SignalProducerFromSocket(abstract.SignalProducerBase):
    def __init__(self, host='', port=5500):
        super().__init__()
        self.host = host
        self.port = port

    def _produce(self):
        # 创建 UDP 套接字
        udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_socket = udp_socket

        # 允许发送广播数据
        udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

        # 在 Linux 下绑定到 eth0 网卡
        udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BINDTODEVICE, b'enx5414a7150450')

        # 绑定到特定的网络接口和端口
        udp_socket.bind((self.host, self.port))  # 绑定到网络接口的5500端口

        while self.running:
            # 接收数据
            data, addr = udp_socket.recvfrom(1024)

            # 打印接收到的数据和发送者地址
            # print(f"Received message from {addr}")

            raw_data = utils.CSI.get_data_from_pack(data)
            # print(f"RAW_DATA: {raw_data}")

            csi_data = utils.CSI.get_csi_from_data(raw_data)
            # print(f"CSI_DATA: {csi_data}")

            cis_matrix = utils.CSI.get_vector_from_csi(csi_data)
            # print(f"CSI_MATRIX: {cis_matrix}")

            self.send_signal(cis_matrix, packet=data)

        self.udp_socket.close()
        self.stopped = True

    @staticmethod
    def get_label():
        return "read from socket"


class SignalProducerFromSerial(abstract.SignalProducerBase):
    """
    从指定串口读取数据
    """

    DATA_COLUMNS_NAMES = ["type", "id", "mac", "rssi", "rate", "sig_mode", "mcs", "bandwidth", "smoothing",
                          "not_sounding", "aggregation", "stbc", "fec_coding",
                          "sgi", "noise_floor", "ampdu_cnt", "channel", "secondary_channel", "local_timestamp", "ant",
                          "sig_len", "rx_state", "len", "first_word", "data"]

    def __init__(self, port="/dev/ttyUSB0", baudrate=921600):
        super().__init__()
        self.port = port
        self.baudrate = baudrate
        self.serial = None

    def _produce(self):
        import serial
        ser = serial.Serial(port=self.port, baudrate=self.baudrate,
                            bytesize=8, parity='N', stopbits=1)

        # Remove invalid subcarriers
        csi_vaid_subcarrier_index = []

        # LLTF: 52
        csi_vaid_subcarrier_index += [i for i in range(6, 32)]  # 26  red
        csi_vaid_subcarrier_index += [i for i in range(33, 59)]  # 26  green
        CSI_DATA_LLFT_COLUMNS = len(csi_vaid_subcarrier_index)

        CSI_DATA_COLUMNS = len(csi_vaid_subcarrier_index)
        csi_data_array = np.zeros(64, dtype=np.complex64)

        if not ser.is_open:
            try:
                ser.open()
            except serial.SerialException as e:
                print(e)
                return

        try:
            while self.running:
                strings = str(ser.readline())
                if not strings:
                    break

                strings = strings.lstrip('b\'').rstrip('\\r\\n\'')
                index = strings.find('CSI_DATA')

                if index == -1:
                    continue

                csv_reader = csv.reader(StringIO(strings))
                csi_data = next(csv_reader)

                if len(csi_data) != len(self.DATA_COLUMNS_NAMES):
                    print("element number is not equal")
                    print("csi_data", csi_data)
                    continue

                try:
                    csi_raw_data = json.loads(csi_data[-1])
                except json.JSONDecodeError:
                    print("data is incomplete")
                    continue

                # Reference on the length of CSI data and usable subcarriers
                # https://docs.espressif.com/projects/esp-idf/en/stable/esp32/api-guides/wifi.html#wi-fi-channel-state-information
                if len(csi_raw_data) != 128 and len(csi_raw_data) != 256 and len(csi_raw_data) != 384:
                    print(f"element number is not equal: {len(csi_raw_data)}")
                    continue

                if len(csi_raw_data) == 128:
                    csi_vaid_subcarrier_len = CSI_DATA_LLFT_COLUMNS
                else:
                    csi_vaid_subcarrier_len = CSI_DATA_COLUMNS

                csi_data_array = csi_data_array * 0
                for i in range(csi_vaid_subcarrier_len):
                    csi_data_array[csi_vaid_subcarrier_index[i]] = complex(
                        csi_raw_data[csi_vaid_subcarrier_index[i] * 2 + 1],
                        csi_raw_data[csi_vaid_subcarrier_index[i] * 2])
                self.send_signal(csi_data_array)
        finally:
            ser.close()
            self.stopped = True
        return

    @staticmethod
    def get_label():
        return "read from serial"

    def __del__(self):
        super().__del__()
        if self.serial is not None:
            self.serial.close()
        self.serial = None


# 从文件中读取csi信号，并按照0.1s的间隔进行生成，模拟实时抓包。实现SignalProducerBase类
class SignalProducerFromUDP(abstract.SignalProducerBase):
    def __init__(self):
        import receiveCSI
        super().__init__()
        self.receiver = receiveCSI.ReadFromUDP(1234)

    def _produce(self):
        import receiveCSI
        def handle_csi(csi):
            print(csi)
            raw_csi = np.copy(csi["csi"])
            INVALID_INDEX = [i for i in range(0, 64) if i not in receiveCSI.LLTF_VALID_INDEX]
            raw_csi[INVALID_INDEX] = 0  # 将无效的子载波的CSI数据置为0
            print("csi 信号:", raw_csi)
            self.send_signal(raw_csi)

        self.receiver.set_handle_csi(handle_csi)
        self.receiver.start()
        while self.running:
            time.sleep(1)
        self.receiver.stop()
        self.stopped = True

    @staticmethod
    def get_label():
        return "read from UDP"


if __name__ == '__main__':
    p = SignalProducerFromSocket()
    reader = lambda s: print(f"read")
    p.register_reader(reader)
    p.start()
