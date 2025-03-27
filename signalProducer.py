import csv
import json
import socket
import time
from io import StringIO

import numpy as np

import abstract
import utils


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
        # self.port = port
        # self.baudrate = baudrate
        # self.serial = None

    def _produce(self):
        import numpy as np
        a = utils.read_csi_vector_from_csv_file("./saved_files_of_9_floor/2024-12-27-01-10_1x/0_10.csv")
        b = utils.read_csi_vector_from_csv_file("./saved_files_of_9_floor/2024-12-27-01-10_1x/1_10.csv")
        data = np.concatenate([a, b], axis=0)
        size = data.shape[0]
        index = 0
        try:
            while self.running:
                csi_data_array = data[index]
                index = (index + 1) % size
                self.send_signal(csi_data_array)
        finally:
            self.stopped = True
        return

    @staticmethod
    def get_label():
        return "read from serial"

    def __del__(self):
        super().__del__()
        # if self.serial is not None:
        #     self.serial.close()
        # self.serial = None


class SignalProducerFromFile:
    @staticmethod
    def get_label():
        return "Dummy"