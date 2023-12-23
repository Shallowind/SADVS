from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout
from PyQt5.QtCore import Qt, pyqtSignal


class VdWidget(QWidget):
    doubleClicked = pyqtSignal()
    mousePressed = pyqtSignal()

    def __init__(self, parent=None):
        super(VdWidget, self).__init__(parent)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.doubleClicked.emit()  # 发射双击信号
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.mousePressed.emit()
