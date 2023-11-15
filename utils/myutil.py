# 查验导入数据是否是图片
from PyQt5.QtGui import QTextCursor


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


class ConsoleRedirector:
    def __init__(self, parent, text_widget):
        self.parent = parent
        self.text_widget = text_widget

    def write(self, text):
        # 在控制台中显示文本
        cursor = self.text_widget.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text)
        self.text_widget.setTextCursor(cursor)
        self.text_widget.ensureCursorVisible()

    def flush(self):
        pass
