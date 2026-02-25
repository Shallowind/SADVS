# 查验导入数据是否是图片
import os
from datetime import datetime

import cv2
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QTextCursor, QColor


def file_is_pic(suffix):
    if suffix == "png" or suffix == "jpg" or suffix == "bmp":
        return True
    return False


def create_incremental_folder(base_path, prefix='ep'):
    # 获取 base_path 下已存在的文件夹列表
    existing_folders = [name for name in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, name))]

    # 找到最大的数字后缀
    max_suffix = 0
    for folder in existing_folders:
        if folder.startswith(prefix):
            try:
                suffix = int(folder[len(prefix):])
                max_suffix = max(max_suffix, suffix)
            except ValueError:
                pass

    # 构造新文件夹的名称
    new_folder_name = f'{prefix}{max_suffix + 1}'
    new_folder_path = os.path.join(base_path, new_folder_name)

    # 创建新文件夹
    os.makedirs(new_folder_path)

    return new_folder_path


# 获得视频信息
def get_video_info(selected_video_path):
    try:
        # 获取视频文件的基本信息
        video_info = os.stat(selected_video_path)
        file_size = video_info.st_size  # 文件大小（字节）

        modified_time = os.path.getmtime(selected_video_path)  # 文件修改日期时间
        formatted_date = datetime.fromtimestamp(modified_time).strftime("%Y-%m-%d %H:%M:%S")
        # 打开视频文件
        cap = cv2.VideoCapture(selected_video_path)
        # 获取视频的总帧数
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # 获取视频的帧率
        frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
        # 获取视频的时长（以秒为单位）
        duration_seconds = total_frames / frame_rate
        # 改成时分秒格式
        hours = int(duration_seconds / 3600)
        minutes = int((duration_seconds % 3600) / 60)
        seconds = int(duration_seconds % 60)
        # 获取视频的分辨率
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # 关闭视频文件
        cap.release()
        file_size = convertBytesToReadable(file_size)
        return total_frames, frame_rate, hours, minutes, seconds, file_size, formatted_date, width, height
    except Exception as e:
        print(f"Error: {e}")
        return None


def convertBytesToReadable(size_in_bytes):
    units = ["B", "KB", "MB", "GB", "TB"]
    unit_index = 0
    # 初始化待转换的字节数
    size = size_in_bytes
    # 循环条件为 size大于等于1024且单位索引小于单位列表长度减1
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1
    formatted_size = "{:.2f}".format(size)
    # 返回格式化后的结果，包括单位
    return f"{formatted_size} {units[unit_index]}"


# 全局变量
class Globals:
    user = None
    settings = {}
    camera_running = False
    dict_text = {}
    select_labels = []
    select_objects = []
    detection_run = False
    yolo_slowfast_dict = {}
    yolov5_dict = {}
    yolov8_dict = {}
    Identify_use_time = "0"


class ConsoleRedirector:
    def __init__(self, parent, text_widget, color=QColor(255, 255, 255)):
        self.parent = parent
        self.text_widget = text_widget
        self.color = color

    def write(self, text):
        # 在控制台中显示文本
        cursor = self.text_widget.textCursor()
        cursor.movePosition(QTextCursor.End)
        if self.color == QColor(255, 0, 0):
            text_with_br = text.replace('\n', '<br>')
            cursor.insertHtml(f'<font color="{self.color.name()}">{text_with_br}</font>')
        else:
            text_with_br = text.replace('\n', '<br>')
            cursor.insertHtml(f'<font color="">{text_with_br}</font>')
        self.text_widget.setTextCursor(cursor)
        self.text_widget.ensureCursorVisible()

    def flush(self):
        pass
