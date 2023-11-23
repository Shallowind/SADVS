# 查验导入数据是否是图片
from PyQt5.QtGui import QTextCursor, QColor


def file_is_pic(suffix):
    if suffix == "png" or suffix == "jpg" or suffix == "bmp":
        return True
    return False


# 全局变量
class Globals:
    settings = {}
    camera_running = False
    dict_text = {}
    select_labels = []
    detection_run = False
    yolo_slowfast_dict = {}
    yolov5_dict = {}
    apple_num =[]
    apple_xy = []


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