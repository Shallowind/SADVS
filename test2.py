import sys

import qdarkstyle
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QPainter, QBitmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox
from PyQt5.uic import loadUi


class FramelessWindow(QMainWindow):
    def __init__(self):
        super(FramelessWindow, self).__init__()

        # 加载UI文件
        self.dragPos = None
        loadUi('user_management.ui', self)

        # 设置窗口标志，去掉窗口边框
        self.setWindowFlags(Qt.FramelessWindowHint)

        # 登录
        self.signin.clicked.connect(self.Login)
        # 退出
        # self.quit.clicked.connect(self.close)
        self.exit.clicked.connect(self.close)
        self.widget_2.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        # self.widget.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())

        # 设置窗口背景颜色为白色
        self.setStyleSheet("background-color: #19232d;")

        # 设置窗口大小
        # self.setGeometry(100, 100, 400, 200)

    def mousePressEvent(self, event):
        # 实现窗口拖动
        if event.button() == Qt.LeftButton:
            self.dragPos = event.globalPos()
            event.accept()

    def Login(self):
        if self.username.text() and self.password.text():
            self.begin()
        else:
            QMessageBox.warning(self, "警告", "用户名或密码不能为空")

    def close(self):
        super(FramelessWindow, self).close()

    def mouseMoveEvent(self, event):
        # 实现窗口拖动
        if hasattr(self, 'dragPos'):
            if self.dragPos is not None:
                newPos = self.mapToGlobal(event.pos() - self.dragPos)
            else:
                # 处理 self.dragPos 为 None 的情况
                return

            self.move(self.mapToGlobal(newPos))
            self.dragPos = event.globalPos()
            event.accept()


if __name__ == '__main__':
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
    app = QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    window = FramelessWindow()
    window.show()
    sys.exit(app.exec_())
