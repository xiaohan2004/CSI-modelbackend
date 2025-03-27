import os
import typing
import numpy as np
import pywt
import scipy.signal

import streamlit_extra_global_data
import utils
from abstract import SignalProcessorBase
from sktime.libs.vmdpy import VMD


# 输出原始信号，不做处理，继承自SignalProcessorBase
class RawSignalProcessor(SignalProcessorBase):
    def process(self, signal: np.ndarray, *args, **kwargs) -> np.ndarray:
        return signal

    @staticmethod
    def get_label() -> str:
        return 'raw'

    def get_method_params(self) -> dict:
        return {}


# 均值滤波信号处理算法，继承自SignalProcessorBase
class MeanFilterSignalProcessor(SignalProcessorBase):
    def process(self, signal: np.ndarray, *args, **kwargs) -> np.ndarray:
        window_size = kwargs.get('window_size', 9)
        if np.iscomplexobj(signal) or signal.dtype == np.complex64:
            amplitude = utils.CSI.get_amplitude(signal)
            angle = utils.CSI.get_phase(signal)
        else:
            amplitude = signal
            angle = None
        # print("amplitude shape:", amplitude.shape)
        # print("angle shape:", angle)
        amplitude = utils.Signal.mean_filter(amplitude, kernel_size=window_size, pad_mode='reflect')
        amplitude = utils.Signal.mean_filter(amplitude, kernel_size=3, pad_mode='reflect')

        # print("mean processed amplitude shape:", amplitude.shape)
        ret = utils.CSI.rebuild_complex(amplitude, angle)
        # print(("mean final csi shape:", ret.shape))
        return ret

    @staticmethod
    def get_label() -> str:
        return 'mean_filter'

    def get_method_params(self) -> dict:
        return {
            'window_size': {
                'type': 'number',
                'default': 7,
            }
        }


# vmd信号处理算法，继承自SignalProcessorBase
# class VmdSignalProcessor(SignalProcessorBase):
#     def process(self, signal: np.ndarray, *args, **kwargs) -> np.ndarray:
#         alpha = kwargs.get('alpha', 2000)
#         tau = kwargs.get('tau', 0.0)
#         K = kwargs.get('K', 10)
#         DC = kwargs.get('DC', 0)
#         init = kwargs.get('init', 1)
#         tol = kwargs.get('tol', 1e-5)
#
#         # 去除直流分量
#         signal = signal - signal.mean()
#         # nan 置0
#         signal = np.where(np.isnan(signal), 0, signal)
#         # 只关心振幅
#         signal = np.abs(signal)
#
#         # Run VMD
#         u, u_hat, omega = VMD(signal, alpha, tau, K, DC, init, tol)
#
#         return np.array(u)
#
#     @staticmethod
#     def get_label() -> str:
#         return 'vmd'
#
#     def get_method_params(self) -> typing.Dict[str, typing.Any]:
#         return {
#             'alpha': {
#                 'type': 'number',
#                 'default': 2000,
#             },
#             'tau': {
#                 'type': 'number',
#                 'default': 0.0,
#             },
#             'K': {
#                 'type': 'number',
#                 'default': 10,
#             },
#             'DC': {
#                 'type': 'number',
#                 'default': 0,
#             },
#             'init': {
#                 'type': 'number',
#                 'default': 1,
#             },
#             'tol': {
#                 'type': 'number',
#                 'default': 1e-5,
#             }
#         }


class HampelFilterSignalProcessor(SignalProcessorBase):
    def process(self, signal: np.ndarray, *args, **kwargs) -> np.ndarray:
        from scipy.signal import medfilt

        window_size = kwargs.get('window_size', 7)
        n_sigmas = 2

        amplitude = utils.CSI.get_amplitude(signal)
        input_series = amplitude

        new_series = utils.Signal.hampel_filter(input_series, window_size, n_sigmas)

        angle = utils.CSI.get_phase(signal)
        amplitude = new_series
        return utils.CSI.rebuild_complex(amplitude, angle)

    @staticmethod
    def get_label() -> str:
        return 'hampel_filter'

    def get_method_params(self) -> dict:
        return {
            'window_size': {
                'type': 'number',
                'default': 9,
            }
        }


class HampelAndMeanFilterSignalProcessor(SignalProcessorBase):
    def process(self, signal: np.ndarray, *args, **kwargs) -> np.ndarray:
        from scipy.signal import medfilt

        window_size = kwargs.get('window_size', 11)
        n_sigmas = 2

        amplitude = utils.CSI.get_amplitude(signal)
        input_series = amplitude

        new_series = utils.Signal.hampel_filter(input_series, window_size, n_sigmas)
        # new_series = utils.Signal.mean_filter(new_series, kernel_size=7, pad_mode='reflect')
        # new_series = utils.Signal.mean_filter(new_series, kernel_size=5, pad_mode='reflect')
        new_series = utils.Signal.mean_filter(new_series, kernel_size=3, pad_mode='reflect')

        angle = utils.CSI.get_phase(signal)
        amplitude = new_series
        return utils.CSI.rebuild_complex(amplitude, angle)

    @staticmethod
    def get_label() -> str:
        return 'hampel_then_mean_filter'

    def get_method_params(self) -> dict:
        return {
            'window_size': {
                'type': 'number',
                'default': 7,
            }
        }


# 傅里叶变换信号处理算法，继承自SignalProcessorBase
class FftSignalProcessor(SignalProcessorBase):
    """
    傅里叶变换信号处理算法,通过将信号转换到频域后，筛选指定频率范围的信号，然后再转换回时域，得到处理后的信号
    """

    def process(self, signal: np.ndarray, *args, **kwargs) -> np.ndarray:
        low_freq = kwargs.get('low_freq', 0.5)
        high_freq = kwargs.get('high_freq', 1000)
        sampling_rate = kwargs.get('sampling_rate', 100)  # 采样率
        signal = MeanFilterSignalProcessor().process(signal)
        # 傅里叶变换
        fft_signal = np.fft.fft(signal)
        # 频率
        freqs = np.fft.fftfreq(len(signal), 1 / sampling_rate)
        # 筛选指定频率范围的信号
        fft_signal[(np.abs(freqs) < low_freq) | (np.abs(freqs) > high_freq)] = 0
        # 傅里叶逆变换
        # return np.fft.ifft(fft_signal).real
        signal = np.fft.ifft(fft_signal)
        return signal  # utils.CSI.rebuild_complex(np.abs(signal), phase)

    @staticmethod
    def get_label() -> str:
        return 'fft'

    def get_method_params(self) -> typing.Dict[str, typing.Any]:
        return {
            'low_freq': {
                'type': 'number',
                'default': 0.1,
            },
            'high_freq': {
                'type': 'number',
                'default': 2,
            },
            'sampling_rate': {
                'type': 'number',
                'default': 10,
            }
        }


# 小波变换信号处理算法，继承自SignalProcessorBase
class WaveletSignalProcessor(SignalProcessorBase):
    def process(self, signal: np.ndarray, *args, **kwargs) -> np.ndarray:
        signal = utils.CSI.get_amplitude(signal)
        wavelet = kwargs.get('wavelet', 'db4')
        # 小波变换的层数计算方式为： 因为第n层的CD频率范围是[fs/2^(n+1), fs/2^n],CA是[0,fs/2^(n+1)]，n从1开始，
        # 所以为了获取到频率小于freq的信号，层数为log2(fs/freq)-1，其中fs为采样频率，freq为信号中的最高频率
        level = kwargs.get('level', 2)

        mode = kwargs.get('mode', 'symmetric')
        coeffs = pywt.wavedec(signal, wavelet, level=level, mode=mode)
        # coeffs[1:] = np.zeros_like(coeffs[1:])  #  保留低频信号
        for i in range(1, len(coeffs)):
            coeffs[i] = np.zeros_like(coeffs[i])
        signal = pywt.waverec(coeffs, wavelet, mode=mode)
        return signal
        # approximation = pywt.upcoef('a', coeffs[0], wavelet, level=level, take=len(signal))
        # return approximation

    @staticmethod
    def get_label() -> str:
        return 'wavelet'

    def get_method_params(self) -> typing.Dict[str, typing.Any]:
        return {
            'wavelet': {
                'type': 'string',
                'default': 'db4',
            },
            'level': {
                'type': 'number',
                'default': 3,
            },
            'mode': {
                'type': 'string',
                'default': 'symmetric',
            }
        }


# EMD信号处理算法，继承自SignalProcessorBase
class EmdSignalProcessor(SignalProcessorBase):
    def process(self, signal: np.ndarray, *args, **kwargs) -> np.ndarray:
        drop_percent = kwargs.get("drop_percent", 0.5)
        spline_kind = kwargs.get('spline_kind', 'cubic')
        energy_ratio_thr = kwargs.get('energy_ratio_thr', 0.2)
        std_thr = kwargs.get('std_thr', 0.2)
        svar_thr = kwargs.get('svar_thr', 0.001)
        total_power_thr = kwargs.get('total_power_thr', 0.005)
        range_thr = kwargs.get('range_thr', 0.001)
        extrema_detection = kwargs.get('extrema_detection', 'simple')

        from PyEMD import EMD
        emd = EMD(spline_kind=spline_kind,
                  extrema_detection=extrema_detection,
                  energy_ratio_thr=energy_ratio_thr,
                  std_thr=std_thr,
                  svar_thr=svar_thr,
                  total_power_thr=total_power_thr,
                  range_thr=range_thr)
        imfs = emd(utils.CSI.get_amplitude(signal))
        drop_size = int(len(imfs) * drop_percent)
        return np.sum(imfs[drop_size:], axis=0)

    @staticmethod
    def get_label() -> str:
        return 'emd'

    def get_method_params(self) -> typing.Dict[str, typing.Any]:
        return {
            'drop_percent': {
                'type': 'number',
                'default': 0.5
            },
            'spline_kind': {
                'type': 'string',
                'default': 'cubic',
            },
            'energy_ratio_thr': {
                'type': 'number',
                'default': 0.2,
            },
            'std_thr': {
                'type': 'number',
                'default': 0.2,
            },
            'svar_thr': {
                'type': 'number',
                'default': 0.001,
            },
            'total_power_thr': {
                'type': 'number',
                'default': 0.005,
            },
            'range_thr': {
                'type': 'number',
                'default': 0.001,
            },
            'extrema_detection': {
                'type': 'string',
                'default': 'simple',
            }
        }


# EEMD信号处理算法，继承自SignalProcessorBase
# class EemdSignalProcessor(SignalProcessorBase):
#     def process(self, signal: np.ndarray, *args, **kwargs) -> np.ndarray:
#         drop_percent = kwargs.get("drop_percent", 0.5)
#         trials = kwargs.get('trial', 100)
#         parallel = kwargs.get('parallel', True)
#         ensemble_size = kwargs.get('ensemble_size', 50)
#         noise_width = kwargs.get('noise_width', 0.05)
#         processes = os.cpu_count() // 2 if parallel else 1
#         from PyEMD import EEMD
#         eemd = EEMD(trials=trials, parallel=parallel, processes=processes, ensemble_size=ensemble_size,
#                     noise_width=noise_width)
#         eIMFs = eemd(utils.CSI.get_amplitude(signal))
#         drop_size = int(len(eIMFs) * drop_percent)
#         return np.sum(eIMFs[drop_size:], axis=0)
#
#     @staticmethod
#     def get_label() -> str:
#         return 'eemd'
#
#     def get_method_params(self) -> typing.Dict[str, typing.Any]:
#         return {
#             'drop_percent': {
#                 'type': 'number',
#                 'default': 0.5
#             },
#             'trial': {
#                 'type': 'number',
#                 'default': 100,
#             },
#             'parallel': {
#                 'type': 'boolean',
#                 'default': True,
#             },
#             'ensemble_size': {
#                 'type': 'number',
#                 'default': 50,
#             },
#             'noise_width': {
#                 'type': 'number',
#                 'default': 0.05,
#             }
#         }
#

if __name__ == '__main__':
    # emdf = EmdSignalProcessor()
    # streamlit_extra_global_data.RECEIVE_S.append(np.random.randint(0, 100, 64))
    # m = streamlit_extra_global_data.RECEIVE_S.get_csi_matrix()
    # x = np.apply_along_axis(emdf, 0, m)
    # print()
    streamlit_extra_global_data.RECEIVE_S.append(np.random.randint(0, 120, 64))
    m = streamlit_extra_global_data.RECEIVE_S.get_csi_matrix()
    x = np.apply_along_axis(MeanFilterSignalProcessor(), 0, m)
    print()
