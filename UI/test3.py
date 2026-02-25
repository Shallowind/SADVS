import sys
import subprocess
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLineEdit, QTextEdit, QPushButton

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Embedded Terminal")
        self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout()

        # 创建一个QPlainTextEdit控件用作终端输出
        self.terminal_output = QTextEdit()
        self.terminal_output.setReadOnly(True)
        layout.addWidget(self.terminal_output)

        # 创建一个QLineEdit控件用作终端输入
        self.terminal_input = QLineEdit()
        self.terminal_input.returnPressed.connect(self.execute_command)
        layout.addWidget(self.terminal_input)

        # 创建一个QPushButton控件用于执行命令
        self.execute_button = QPushButton("Execute")
        self.execute_button.clicked.connect(self.execute_command)
        layout.addWidget(self.execute_button)

        # 创建一个QWidget作为主窗口的中央控件，并设置布局
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def execute_command(self):
        command = self.terminal_input.text()
        self.terminal_output.append(f">>> {command}")
        self.terminal_input.clear()

        # 使用subprocess模块执行命令并将结果添加到终端输出
        result = subprocess.run(command, capture_output=True, shell=True, text=True)
        output = result.stdout.strip()
        self.terminal_output.append(output)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
