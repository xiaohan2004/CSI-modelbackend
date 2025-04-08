import mysql.connector
from mysql.connector import Error
import time
import os
import json
import configparser


class MySQLTools:
    def __init__(self, config_path=None, max_retries=3):
        """
        初始化MySQL工具类

        参数:
            config_path: 配置文件路径，如果为None则尝试使用默认路径
            max_retries: 最大重试次数，默认3次
        """
        self.host = None
        self.user = None
        self.password = None
        self.database = None
        self.port = 3306
        self.max_retries = max_retries
        self.connection = None

        # 从配置文件加载
        self._load_config(config_path)

        # 初始化时尝试连接数据库
        if self.host and self.user and self.password:
            self.connect()
        else:
            print("数据库配置不完整，连接失败")

    def _load_config(self, config_path=None):
        """从配置文件加载数据库连接信息"""
        try:
            # 如果未指定配置文件，则尝试使用默认路径
            if config_path is None:
                # 尝试多个可能的位置
                possible_paths = [
                    "config/db_config.ini",  # 项目根目录下的config文件夹
                    "db_config.ini",  # 项目根目录
                    os.path.join(os.path.dirname(__file__), "db_config.ini"),  # 当前脚本同目录
                ]

                for path in possible_paths:
                    if os.path.exists(path):
                        config_path = path
                        break

            if config_path and os.path.exists(config_path):
                # 根据文件扩展名选择解析方式
                if config_path.endswith(".ini") or config_path.endswith(".conf"):
                    self._load_from_ini(config_path)
                elif config_path.endswith(".json"):
                    self._load_from_json(config_path)
                else:
                    print(f"不支持的配置文件格式: {config_path}")
                    print("使用默认配置")
            else:
                print("未找到配置文件，使用默认配置")
        except Exception as e:
            print(f"加载配置文件时出错: {e}")
            print("使用默认配置")

    def _load_from_ini(self, config_path):
        """从INI文件加载配置"""
        config = configparser.ConfigParser()
        config.read(config_path)

        if "mysql" in config:
            mysql_config = config["mysql"]
            self.host = mysql_config.get("host", self.host)
            self.user = mysql_config.get("user", self.user)
            self.password = mysql_config.get("password", self.password)
            self.database = mysql_config.get("database", self.database)
            self.port = mysql_config.getint("port", self.port)
            print(f"从 {config_path} 加载了MySQL配置")

    def _load_from_json(self, config_path):
        """从JSON文件加载配置"""
        with open(config_path, "r") as f:
            config = json.load(f)

        if "mysql" in config:
            mysql_config = config["mysql"]
            self.host = mysql_config.get("host", self.host)
            self.user = mysql_config.get("user", self.user)
            self.password = mysql_config.get("password", self.password)
            self.database = mysql_config.get("database", self.database)
            self.port = mysql_config.get("port", self.port)
            print(f"从 {config_path} 加载了MySQL配置")

    def connect(self):
        """建立数据库连接，如果失败会重试几次"""
        retry_count = 0
        while retry_count < self.max_retries:
            try:
                self.connection = mysql.connector.connect(
                    host=self.host,
                    user=self.user,
                    password=self.password,
                    port=self.port,
                    auth_plugin="mysql_native_password",
                )
                # 确保数据库存在
                self._ensure_database_exists()
                # 选择数据库
                self.connection.database = self.database
                print("数据库连接成功")
                return True
            except Error as e:
                retry_count += 1
                print(f"连接失败 (尝试 {retry_count}/{self.max_retries}): {e}")
                if retry_count < self.max_retries:
                    time.sleep(2)  # 等待2秒后重试
                else:
                    print("达到最大重试次数，连接失败")
                    raise

    def _ensure_database_exists(self):
        """确保数据库存在，如果不存在则创建"""
        try:
            cursor = self.connection.cursor()
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {self.database}")
            cursor.close()
        except Error as e:
            print(f"创建数据库时出错: {e}")
            raise

    def _ensure_connection(self):
        """确保数据库连接是活跃的"""
        try:
            if self.connection is None or not self.connection.is_connected():
                self.connect()
        except Error as e:
            print(f"重新连接数据库时出错: {e}")
            raise

    def create_table(self):
        """创建模型保存表"""
        try:
            self._ensure_connection()
            cursor = self.connection.cursor()
            create_table_query = f"""
            CREATE TABLE IF NOT EXISTS {self.database}.csi_model_saved_table (
                uuid VARCHAR(36) PRIMARY KEY,
                model_name VARCHAR(255) NOT NULL,
                model_saved_path VARCHAR(255) NOT NULL,
                model_insert_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
            cursor.execute(create_table_query)
            self.connection.commit()
            cursor.close()
            print("Table csi_model_saved_table created or already exists.")
        except Error as e:
            print(f"创建表时出错: {e}")
            raise

    def close(self):
        """关闭数据库连接"""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            print("数据库连接已关闭")

    def __del__(self):
        """析构函数，确保连接被正确关闭"""
        self.close()

    # 示例使用方法
    def insert_model_info(self, uuid, model_name, model_saved_path):
        """插入模型信息"""
        try:
            self._ensure_connection()
            cursor = self.connection.cursor()
            insert_query = f"""
            INSERT INTO {self.database}.csi_model_saved_table 
            (uuid, model_name, model_saved_path) 
            VALUES (%s, %s, %s)
            """
            cursor.execute(insert_query, (uuid, model_name, model_saved_path))
            self.connection.commit()
            cursor.close()
            print("模型信息插入成功")
        except Error as e:
            print(f"插入模型信息时出错: {e}")
            raise

    def insert_record(self, uuid, model_name, model_saved_path):
        try:
            cursor = self.connection.cursor()
            insert_query = f"""
            INSERT INTO {self.database}.csi_model_saved_table (uuid, model_name, model_saved_path)
            VALUES (%s, %s, %s)
            """
            cursor.execute(insert_query, (uuid, model_name, model_saved_path))
            self.connection.commit()
            print("Record inserted successfully.")
        except Error as e:
            print(f"Error while inserting record: {e}")

    def delete_record(self, uuid):
        try:
            cursor = self.connection.cursor()
            delete_query = (
                f"DELETE FROM {self.database}.csi_model_saved_table WHERE uuid = %s"
            )
            cursor.execute(delete_query, (uuid,))
            self.connection.commit()
            print("Record deleted successfully.")
        except Error as e:
            print(f"Error while deleting record: {e}")

    def update_record(self, uuid, model_name=None, model_saved_path=None):
        try:
            update_query = f"UPDATE {self.database}.csi_model_saved_table SET "
            values = []
            if model_name:
                update_query += "model_name = %s, "
                values.append(model_name)
            if model_saved_path:
                update_query += "model_saved_path = %s, "
                values.append(model_saved_path)
            update_query = update_query.rstrip(", ")
            update_query += " WHERE uuid = %s"
            values.append(uuid)
            cursor = self.connection.cursor()
            cursor.execute(update_query, tuple(values))
            self.connection.commit()
            print("Record updated successfully.")
        except Error as e:
            print(f"Error while updating record: {e}")

    def select_record(self, uuid=None):
        try:
            cursor = self.connection.cursor(dictionary=True)
            if uuid:
                select_query = f"SELECT * FROM {self.database}.csi_model_saved_table WHERE uuid = %s"
                cursor.execute(select_query, (uuid,))
            else:
                select_query = f"SELECT * FROM {self.database}.csi_model_saved_table"
                cursor.execute(select_query)
            records = cursor.fetchall()
            return records
        except Error as e:
            print(f"Error while selecting record: {e}")
            return []


if __name__ == "__main__":
    # 使用示例
    tool = MySQLTools(config_path="config/db_config.ini")
    # 创建表
    tool.create_table()
    # 插入记录
    tool.insert_record(
        "123e4567-e89b-12d3-a456-426614174000", "test_model", "/path/to/model"
    )
    # 查询所有记录
    all_records = tool.select_record()
    print(all_records)
    # 查询指定记录
    specific_record = tool.select_record("123e4567-e89b-12d3-a456-426614174000")
    print(specific_record)
    # 更新记录
    tool.update_record(
        "123e4567-e89b-12d3-a456-426614174000", model_name="new_model_name"
    )
    # 删除记录
    tool.delete_record("123e4567-e89b-12d3-a456-426614174000")
    tool.close()
