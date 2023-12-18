from PyQt5.QtCore import QObject, pyqtSignal, Qt

class YourClass(QObject):
    # 为了发射信号，我们需要创建一个信号对象
    playerSizeChanged = pyqtSignal()

    def __init__(self):
        super().__init__()

        # 假设 self.player_2 是一个QWidget对象，连接其大小变化信号到槽函数
        self.player_2.installEventFilter(self)

    def eventFilter(self, obj, event):
        if obj == self.player_2 and event.type() == Qt.Resize:
            # 当 self.player_2 大小变化时，发射信号
            self.playerSizeChanged.emit()

        return super().eventFilter(obj, event)

    def handlePlayerSizeChange(self):
        # 在这里放上述的缩放计算和设置新的最大大小的代码
        scale_factor = min(self.player_2.width() / showImage.width(), self.player_2.height() / showImage.height())
        new_width = int(showImage.width() * scale_factor)
        new_height = int(showImage.height() * scale_factor)
        self.camera_2.setMaximumSize(new_width, new_height)

# 在你的初始化代码中
your_instance = YourClass()
your_instance.playerSizeChanged.connect(your_instance.handlePlayerSizeChange)
