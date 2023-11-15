from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QMessageBox
from PyQt5.QtCore import Qt
from PyQt5 import uic
import qdarkstyle
from qdarkstyle import LightPalette


class User_management(QWidget):
    def __init__(self):
        super().__init__()
        self.ui = uic.loadUi("user_management.ui")
        self.ui.resize(1000, 600)
        self.ui.setWindowTitle("用户登录")
        self.ui.show()  # 显示窗口
        self.ui.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())

        # 登录
        self.ui.login.clicked.connect(self.Login)
        # 退出
        self.ui.quit.clicked.connect(self.close)

    def close(self):
        del self.ui
        return 0

    def begin(self):
        del self.ui
        return 1
    def Login(self):
        if self.ui.username.text()  and self.ui.password.text():
            self.begin()
        else:
            QMessageBox.warning(self, "警告", "用户名或密码不能为空")

if __name__ == "__main__":
    app = QApplication([])
    user_management = User_management()
    user_management.ui.show()
    app.exec()