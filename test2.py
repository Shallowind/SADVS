import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QPlainTextEdit, QPushButton, QWidget
from PyQt5.QtCore import Qt, QTextStream
from PyQt5.QtGui import QPalette, QColor, QTextCursor

from utils.myutil import ConsoleRedirector


class ConsoleWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        # 设置主窗口
        self.setWindowTitle('Console Window')
        self.setGeometry(100, 100, 800, 600)

        # 创建控制台输出区域
        self.console_output = QPlainTextEdit(self)
        self.console_output.setReadOnly(True)  # 设置为只读
        # self.console_output.setStyleSheet("background-color: black; color: white;")  # 设置背景颜色和文本颜色

        # 创建清除按钮
        clear_button = QPushButton('Clear Console', self)
        clear_button.clicked.connect(self.clearConsole)

        # 创建布局并添加控件
        layout = QVBoxLayout()
        layout.addWidget(self.console_output)
        layout.addWidget(clear_button)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # 重定向标准输出到控制台
        sys.stdout = ConsoleRedirector(self, self.console_output)

    def clearConsole(self):
        # self.console_output.clear()
        print("123")




if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = ConsoleWindow()
    mainWindow.show()
    sys.exit(app.exec_())
