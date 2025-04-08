import requests
from flask import jsonify

from tool.csi_tools import read_raw_data_between_time, saved_csi_data_with_label_to_different_file
from tool.mysql_tools import MySQLTools


def main():
    url = "http://10.100.164.48:8080/api/rawData/between"
    params = {
        "startTime": 1742898507961,
        "endTime": 1742898607961
    }
    response = requests.get(url, params=params)
    json_data = response.json()
    if json_data is None or json_data.get('msg') != 'success':
        return jsonify({'error': 'No data available'})
    complex_list = read_raw_data_between_time(json_data)
    saved_csi_data_with_label_to_different_file(complex_list, path_0="label/label_0.csv", path_1="label/label_1.csv")

    tool = MySQLTools()

    tool.insert_record('123e4567-e89b-12d3-a456-426614174000', 'test_model', '/path/to/model')

    all_records = tool.select_record()
    print(all_records)

    tool.close_connection()


if __name__ == '__main__':
    main()