import os

def create_folder_if_not_exists(folder_path):
    """
    检测给定路径的文件夹是否存在，如果不存在则递归创建文件夹。

    :param folder_path: 需要检测和创建的文件夹路径
    """
    try:
        os.makedirs(folder_path, exist_ok=True)
        print(f"文件夹 '{folder_path}' 已存在或已创建成功。")
    except Exception as e:
        print(f"创建文件夹 '{folder_path}' 时出错: {e}")