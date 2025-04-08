import os

from flask import jsonify



def check_train_csv_json(request_json):
    """
    检查 request_json 中是否存在需要的参数：
    :param request_json:
    :return:
    """
    if (request_json is None or
            request_json.get("data") is None):
        return False, jsonify({"error": "raw data is not provided"})
    return True, None

def delete_temp_zip(response, zip_file_path=r'tool/train_csv_tmp/train_files.zip'):
    try:
        if os.path.exists(zip_file_path):
            os.remove(zip_file_path)
    except Exception as e:
        print(f"Error deleting temporary zip file: {e}")
    return response

