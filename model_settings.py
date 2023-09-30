import sys

from PyQt5.QtWidgets import QApplication
from PyQt5 import uic


class ModelSettings:
    def __init__(self):
        self.app = QApplication(sys.argv)  # 创建Qt应用程序对象
        self.ui = uic.loadUi('model_settings.ui')
        self.ui.setWindowTitle("识别设置")
        self.ui.show()  # 显示窗口


if __name__ == "__main__":
    app = QApplication([])
    modelsettings = ModelSettings()
    modelsettings.ui.show()
    app.exec()
