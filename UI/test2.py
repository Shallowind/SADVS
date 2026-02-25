import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QPushButton, QWidget
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen
from PyQt5.QtCore import Qt, QRect


class SelectAreaLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.startPoint = None
        self.endPoint = None
        self.rect = QRect()

    def mousePressEvent(self, event):
        self.startPoint = event.pos()
        self.endPoint = self.startPoint
        self.update()

    def mouseMoveEvent(self, event):
        self.endPoint = event.pos()
        self.update()

    def mouseReleaseEvent(self, event):
        self.endPoint = event.pos()
        self.rect = QRect(self.startPoint, self.endPoint)
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.startPoint or not self.endPoint:
            return
        painter = QPainter(self)
        painter.setPen(QPen(Qt.red, 2))
        painter.drawRect(QRect(self.startPoint, self.endPoint))


class MainWindow(QMainWindow):
    def __init__(self, img_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Select Area')
        self.imageLabel = SelectAreaLabel()
        img = cv2.imread(img_path)
        height, width, channel = img.shape
        bytesPerLine = 3 * width
        qImg = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
        self.imageLabel.setPixmap(QPixmap.fromImage(qImg))

        self.okButton = QPushButton('OK')
        self.okButton.clicked.connect(self.close)

        layout = QVBoxLayout()
        layout.addWidget(self.imageLabel)
        layout.addWidget(self.okButton)

        mainWidget = QWidget()
        mainWidget.setLayout(layout)
        self.setCentralWidget(mainWidget)

    def get_coordinates(self):
        rect = self.imageLabel.rect
        return [(rect.topLeft().x(), rect.topLeft().y()),
                (rect.topRight().x(), rect.topRight().y()),
                (rect.bottomRight().x(), rect.bottomRight().y()),
                (rect.bottomLeft().x(), rect.bottomLeft().y())]


def get_rectangle_coordinates(img_path):
    app = QApplication([])
    window = MainWindow(img_path)
    window.show()
    app.exec_()
    return window.get_coordinates()


if __name__ == '__main__':
    coordinates = get_rectangle_coordinates('D:/0000\sadvs\data\show.png')
    print(coordinates)
