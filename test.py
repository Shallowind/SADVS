from PyQt5.QtWidgets import QWidget, QApplication, QGraphicsDropShadowEffect
from PyQt5.QtGui import QPixmap, QPainter
from PyQt5.QtCore import Qt

class RoundedWidget(QWidget):
    def __init__(self):
        super(RoundedWidget, self).__init__()

        pixmap = QPixmap(self.size())
        pixmap.setDevicePixelRatio(self.devicePixelRatioF())

        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.HighQualityAntialiasing)

        # Use QGraphicsDropShadowEffect for smoother edges
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(10)
        self.setGraphicsEffect(shadow)

        painter.setBrush(Qt.white)
        painter.drawRoundedRect(self.rect(), 20, 20)
        self.setMask(pixmap.mask())
        painter.end()

if __name__ == '__main__':
    import sys

    app = QApplication(sys.argv)
    widget = RoundedWidget()
    widget.show()
    sys.exit(app.exec_())
