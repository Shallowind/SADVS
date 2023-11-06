from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import sys

# 按钮高度
BUTTON_HEIGHT = 30
# 按钮宽度
BUTTON_WIDTH = 30
# 标题栏高度
TITLE_HEIGHT = 30


class TitleWidget(QWidget):
    def __init__(self):
        super().__init__()
        # self.setStyleSheet("background-color:blue")
        titleIcon = QPixmap(".\icon.png")
        Icon = QLabel()
        Icon.setPixmap(titleIcon.scaled(25, 25))
        titleContent = QLabel("标题内容")
        titleContent.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        titleContent.setFixedHeight(TITLE_HEIGHT)
        titleContent.setObjectName("TitleContent")
        self.ButtonMin = QPushButton()
        self.ButtonMin.setFixedSize(QSize(BUTTON_WIDTH, BUTTON_HEIGHT))
        self.ButtonMin.setObjectName("ButtonMin")
        self.ButtonMax = QPushButton()
        self.ButtonMax.setFixedSize(QSize(BUTTON_WIDTH, BUTTON_HEIGHT))
        self.ButtonMax.setObjectName("ButtonMax")
        self.ButtonRestore = QPushButton()
        self.ButtonRestore.setFixedSize(QSize(BUTTON_WIDTH, BUTTON_HEIGHT))
        self.ButtonRestore.setObjectName("ButtonRestore")
        self.ButtonRestore.setVisible(False)
        self.ButtonClose = QPushButton()
        self.ButtonClose.setFixedSize(QSize(BUTTON_WIDTH, BUTTON_HEIGHT))
        self.ButtonClose.setObjectName("ButtonClose")
        mylayout = QHBoxLayout()
        mylayout.setSpacing(0)
        mylayout.setContentsMargins(0, 0, 0, 0)
        mylayout.addWidget(Icon)

        mylayout.addWidget(titleContent)
        mylayout.addWidget(self.ButtonMin)
        mylayout.addWidget(self.ButtonMax)
        mylayout.addWidget(self.ButtonRestore)
        mylayout.addWidget(self.ButtonClose)

        self.setLayout(mylayout)
        # QSS可写在文件中 读文件使用 这里方便大家使用直接写在代码里吧
        Qss = '''

            QLabel#TitleContent
            {
                color: #FFFFFF;
            }

            QPushButton#ButtonMin
            {
                border-image:url(./min.png) 0 81 0 0 ;

            }

            QPushButton#ButtonMin:hover
            {
                border-image:url(./min.png) 0 54 0 27 ;
            }

            QPushButton#ButtonMin:pressed
            {
                border-image:url(./min.png) 0 27 0 54 ;
            }

            QPushButton#ButtonMax
            {
                border-image:url(./max.png) 0 81 0 0 ;
            }

            QPushButton#ButtonMax:hover
            {
                border-image:url(./max.png) 0 54 0 27 ;
            }

            QPushButton#ButtonMax:pressed
            {
                border-image:url(./max.png) 0 27 0 54 ;
            }

            QPushButton#ButtonRestore
            {
                border-image:url(./restore.png) 0 81 0 0 ;
            }

            QPushButton#ButtonRestore:hover
            {
                border-image:url(./restore.png) 0 54 0 27 ;
            }

            QPushButton#ButtonRestore:pressed
            {
                border-image:url(./restore.png) 0 27 0 54 ;
            }

            QPushButton#ButtonClose
            {
                border-image:url(./close.png) 0 81 0 0 ;
                border-top-right-radius:3 ;
            }

            QPushButton#ButtonClose:hover
            {
                border-image:url(./close.png) 0 54 0 27 ;
                border-top-right-radius:3 ;
            }

            QPushButton#ButtonClose:pressed
            {
                border-image:url(./close.png) 0 27 0 54 ;
                border-top-right-radius:3 ;
            }

        '''
        self.setStyleSheet(Qss)

        self.restorePos = None
        self.restoreSize = None
        self.startMovePos = None

    def saveRestoreInfo(self, point, size):
        self.restorePos = point
        self.restoreSize = size

    def getRestoreInfo(self):
        return self.restorePos, self.restoreSize


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowMinimizeButtonHint)
        self.resize(800, 600)
        AllWidget = QWidget()
        # AllWidget.setStyleSheet("background-color:red")
        Alllayout = QVBoxLayout()
        Alllayout.setSpacing(0)
        Alllayout.setContentsMargins(0, 0, 0, 0)
        AllWidget.setLayout(Alllayout)
        self.title = TitleWidget()
        self.title.setFixedWidth(self.width())
        self.title.setFixedHeight(TITLE_HEIGHT)
        self.title.ButtonMin.clicked.connect(self.ButtonMinSlot)
        self.title.ButtonMax.clicked.connect(self.ButtonMaxSlot)
        self.title.ButtonRestore.clicked.connect(self.ButtonRestoreSlot)
        self.title.ButtonClose.clicked.connect(self.ButtonCloseSlot)
        centerWidget = QWidget()
        # centerWidget 中可以随意添加自己想用的控件
        # centerWidget.setStyleSheet("background-color:red")
        Qss = '''
            QMainWindow{
                background:qlineargradient(spread:pad,x1:1,y1:0,x2:0,y2:1,stop:0 rgba(51,146,255,255),stop:1 rgba(255,255,255,255));  

            }
        '''

        Alllayout.addWidget(self.title)
        Alllayout.addWidget(centerWidget)
        self.setCentralWidget(AllWidget)
        self.setStyleSheet(Qss)

    def ButtonMinSlot(self):
        self.showMinimized()

    def ButtonMaxSlot(self):
        self.title.ButtonMax.setVisible(False)
        self.title.ButtonRestore.setVisible(True)
        self.title.saveRestoreInfo(self.pos(), QSize(self.width(), self.height()))
        desktopRect = QApplication.desktop().availableGeometry()
        FactRect = QRect(desktopRect.x() - 3, desktopRect.y() - 3, desktopRect.width() + 6, desktopRect.height() + 6)
        print(FactRect)
        self.setGeometry(FactRect)
        self.setFixedSize(desktopRect.width() + 6, desktopRect.height() + 6)

    def ButtonRestoreSlot(self):
        self.title.ButtonMax.setVisible(True)
        self.title.ButtonRestore.setVisible(False)
        windowPos, windowSize = self.title.getRestoreInfo()
        # print(windowPos,windowSize.width(),windowSize.height())
        self.setGeometry(windowPos.x(), windowPos.y(), windowSize.width(), windowSize.height())
        self.setFixedSize(windowSize.width(), windowSize.height())

    def ButtonCloseSlot(self):
        self.close()

    def paintEvent(self, event):
        self.title.setFixedWidth(self.width())


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mw = MainWindow()
    mw.show()
    sys.exit(app.exec_())
