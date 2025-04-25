from tool.mysql_tools import MySQLTools
import threading
import time

# 使用配置文件初始化数据库工具，不再硬编码连接参数
db_tool = MySQLTools()


# 创建一个定期检查数据库连接的函数
def check_db_connection():
    """检查数据库连接状态，如果断开则自动重连"""
    while True:
        try:
            # 每隔5分钟检查一次连接状态
            time.sleep(300)

            # 检查连接是否有效
            if not db_tool.is_connected():
                print("检测到数据库连接已断开，正在尝试重新连接...")
                db_tool.connect()
                print("数据库连接检查完成")
            else:
                print("数据库连接正常")

        except Exception as e:
            print(f"检查数据库连接时出错: {e}")
            # 发生错误时等待短暂时间后重试
            time.sleep(10)


# 创建并启动后台线程
db_connection_thread = threading.Thread(target=check_db_connection, daemon=True)
db_connection_thread.start()
print("数据库连接监控线程已启动")
