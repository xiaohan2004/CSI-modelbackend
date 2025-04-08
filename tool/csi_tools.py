import json

import numpy as np

from support_web import get_dataloader_from_csv

# 定义缓存大小
CACHE_SIZE = 100


def read_raw_data_between_time(json_data):
    try:
        data = json_data.get('data')
        if isinstance(data, dict):  # 如果 data 是字典，将其转换为列表
            data = [data]
        complex_csi_list = []
        label_list = []
        cache = []
        for item in data:
            csi_data_str = item.get('csiData')
            # 如果 json 没有 label 标签则自定义 label = 0
            label = item.get('label', 0)
            if csi_data_str:
                try:
                    csi_data = json.loads(csi_data_str)
                except json.JSONDecodeError as e:
                    return {
                        "error": f"Failed to decode JSON in csiData: {str(e)}",
                        "csi_data": [],
                        "label": []
                    }
                try:
                    complex_csi = handle_raw_csi_data_to_complex(csi_data)
                except Exception as e:
                    return {
                        "error": f"Error processing CSI data: {str(e)}",
                        "csi_data": [],
                        "label": []
                    }
                cache.append((complex_csi, label))
                if len(cache) >= CACHE_SIZE:
                    for csi, lab in cache:
                        complex_csi_list.append(csi)
                        label_list.append(lab)
                    cache = []
        # 处理剩余的缓存数据
        for csi, lab in cache:
            complex_csi_list.append(csi)
            label_list.append(lab)
        result = {
            "csi_data": complex_csi_list,
            "label": label_list
        }
        return result
    except Exception as e:
        return {
            "error": f"An unexpected error occurred: {str(e)}",
            "csi_data": [],
            "label": []
        }


def load_data_from_csv(path_0 = r'tool/label/label_0.csv',path_1 = r'tool/label/label_1.csv'):
    flag_0 = 1
    flag_1 = 1
    try:
        # 检查文件是否为空
        with open(path_0, 'r') as file_0:
            if len(file_0.readlines()) == 0:
                flag_0 = 0
        with open(path_1, 'r') as file_1:
            if len(file_1.readlines()) == 0:
                flag_1 = 0

        if flag_0 == 0 and flag_1 == 0:
            return None, None
        elif flag_0 == 0 and flag_1 == 1:
            csv_with_label = ([path_1], [1])
        elif flag_0 == 1 and flag_1 == 0:
            csv_with_label = ([path_0], [0])
        else:
            csv_with_label = ([path_0,path_1], [0,1])

        # 调用 get_dataloader_from_csv 函数
        train_set, test_set = get_dataloader_from_csv(csv_with_label)
        return train_set, test_set
    except FileNotFoundError:
        return None, None
    except Exception as e:
        print(f"数据加载出错: {e}")
        return None, None

CACHE_SIZE = 100  # 假设 CACHE_SIZE 是一个全局变量

def saved_csi_data_with_label_to_different_file(csi_with_label, clean=True,
                                                path_0 = r'tool/label/label_0.csv',
                                                path_1 = r'tool/label/label_1.csv'):
    success = True  # 初始化标志变量为 True
    try:
        # 先清空需要保存的文件
        if clean:
            try:
                with open(path_0, 'w') as f:
                    f.write('')
            except FileNotFoundError:
                print(f"文件 '{path_0}' 未找到。")
                success = False  # 发生异常，标志变量设为 False
            except PermissionError:
                print(f"没有权限清空文件 '{path_0}'。")
                success = False  # 发生异常，标志变量设为 False

            try:
                with open(path_1, 'w') as f:
                    f.write('')
            except FileNotFoundError:
                print(f"文件 '{path_1}' 未找到。")
                success = False  # 发生异常，标志变量设为 False
            except PermissionError:
                print(f"没有权限清空文件 '{path_1}'。")
                success = False  # 发生异常，标志变量设为 False

        cache = []
        for csi_data, label in zip(csi_with_label["csi_data"], csi_with_label["label"]):
            if csi_data is None or label is None:
                print("数据项缺少 'csi_data' 或 'label' 键。")
                continue
            cache.append((csi_data, label))
            if len(cache) >= CACHE_SIZE:
                for csi, lab in cache:
                    if lab == 0:
                        try:
                            with open(path_0, 'a') as f:
                                f.write(csi + '\n')
                        except FileNotFoundError:
                            print(f"文件 '{path_0}' 未找到，无法写入数据。")
                            success = False  # 发生异常，标志变量设为 False
                        except PermissionError:
                            print(f"没有权限写入文件 '{path_0}'。")
                            success = False  # 发生异常，标志变量设为 False
                    elif lab == 1:
                        try:
                            with open(path_1, 'a') as f:
                                f.write(csi + '\n')
                        except FileNotFoundError:
                            print(f"文件 '{path_1}' 未找到，无法写入数据。")
                            success = False  # 发生异常，标志变量设为 False
                        except PermissionError:
                            print(f"没有权限写入文件 '{path_1}'。")
                            success = False  # 发生异常，标志变量设为 False
                cache = []
        # 处理剩余的缓存数据
        for csi, lab in cache:
            if lab == 0:
                try:
                    with open(path_0, 'a') as f:
                        f.write(csi + '\n')
                except FileNotFoundError:
                    print(f"文件 '{path_0}' 未找到，无法写入数据。")
                    success = False  # 发生异常，标志变量设为 False
                except PermissionError:
                    print(f"没有权限写入文件 '{path_0}'。")
                    success = False  # 发生异常，标志变量设为 False
            elif lab == 1:
                try:
                    with open(path_1, 'a') as f:
                        f.write(csi + '\n')
                except FileNotFoundError:
                    print(f"文件 '{path_1}' 未找到，无法写入数据。")
                    success = False  # 发生异常，标志变量设为 False
                except PermissionError:
                    print(f"没有权限写入文件 '{path_1}'。")
                    success = False  # 发生异常，标志变量设为 False
    except Exception as e:
        print(f"发生未知异常: {e}")
        success = False  # 发生异常，标志变量设为 False

    return success



def handle_raw_csi_data_to_complex(csi_data):
    real_part = np.array(csi_data['real'], dtype=np.float32)
    imag_part = np.array(csi_data['imag'], dtype=np.float32)
    complex_str_list = []
    for r, i in zip(real_part, imag_part):
        if r == 0 and i != 0:
            complex_str = f"{i}j"
        elif r != 0 and i == 0:
            complex_str = str(r)
        elif r != 0 and i > 0:
            complex_str = f"({r}+{i}j)"
        elif r != 0 and i < 0:
            complex_str = f"({r}{i}j)"
        else:
            complex_str = "0j"
        complex_str_list.append(complex_str)
    result_str = ','.join(complex_str_list)
    return result_str