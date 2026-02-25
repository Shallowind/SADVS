from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout
from PyQt5.QtCore import Qt, pyqtSignal


class VideoLabel(QLabel):
    doubleClicked = pyqtSignal()

    def __init__(self, parent=None):
        super(VideoLabel, self).__init__(parent)
        self.setAlignment(Qt.AlignCenter)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.doubleClicked.emit()  # 发射双击信号
