from PyQt5.QtCore import pyqtSignal, QObject

class MyObject(QObject):
    signal = pyqtSignal()

    def __init__(self):
        super().__init__()

    def connect_and_emit(self):
        self.signal.connect(self.box)
        self.signal.emit()

    def box(self):
        print("接收到信号")

my_object = MyObject()
my_object.connect_and_emit()
