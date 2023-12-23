from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QLabel, QApplication


class VideoLabel(QLabel):
    doubleClicked = pyqtSignal()  # 定义一个双击信号

    def __init__(self, parent=None):
        super(VideoLabel, self).__init__(parent)
        self.setAlignment(Qt.AlignCenter)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.doubleClicked.emit()  # 发射双击信号


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    label = VideoLabel('123')


    def handle_double_click():
        print("Label double-clicked")


    label.doubleClicked.connect(handle_double_click)  # 连接信号到槽函数

    label.show()
    sys.exit(app.exec_())
