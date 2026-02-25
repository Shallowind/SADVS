from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication
from UI.Controller import Controller

if __name__ == "__main__":
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
    app = QApplication([])
    # app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    controller = Controller()
    controller.show_user_management()
    app.exec()
