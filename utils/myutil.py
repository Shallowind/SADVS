# 查验导入数据是否是图片
import os

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


# 全局变量
class Globals:
    settings = {}
    camera_running = False
    dict_text = {}
    select_labels = []
    detection_run = False
    yolo_slowfast_dict = {}
    yolov5_dict = {}


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
            cursor.insertText('\n')
        else:
            text_with_br = text.replace('\n', '<br>')
            cursor.insertHtml(f'<font color="{self.color.name()}">{text_with_br}</font>')
        self.text_widget.setTextCursor(cursor)
        self.text_widget.ensureCursorVisible()

    def flush(self):
        pass
