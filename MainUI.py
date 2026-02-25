# coding:utf-8
import sys

from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtGui import QIcon, QDesktopServices
from PyQt5.QtWidgets import QApplication, QFrame, QHBoxLayout, QWidget

from IdentifyResults import IdentifyResults
from qfluentwidgets import (NavigationItemPosition, MessageBox, setTheme, Theme, MSFluentWindow,
                            NavigationAvatarWidget, qrouter, SubtitleLabel, setFont, FluentWindow, qconfig)
from qfluentwidgets import FluentIcon as FIF

from UI.centralwidget import Ui_centralwidget
from UI.labels_settings import LabelsSettings
from UI.setting_interface import SettingInterface
from UI.users_widget import UserWidget
from UI.view_interface import ViewInterface
from common.config import cfg, MY_URL
from common.signal_bus import signalBus
from mainwindow import Mainwindow
from common import resource


class Widget(QFrame):

    def __init__(self, text: str, parent=None):
        super().__init__(parent=parent)
        self.label = SubtitleLabel(text, self)
        self.hBoxLayout = QHBoxLayout(self)

        setFont(self.label, 24)
        self.label.setAlignment(Qt.AlignCenter)
        self.hBoxLayout.addWidget(self.label, 1, Qt.AlignCenter)
        self.setObjectName(text.replace(' ', '-'))


class MainWindow(MSFluentWindow):

    def __init__(self, controller=None):
        super().__init__()
        self.controller = controller

        self.centralwidget = Mainwindow(self)
        self.SettingInterface = SettingInterface()
        self.identifyResults = IdentifyResults()
        self.usersInterface = ViewInterface(self)
        self.labelsInterface = LabelsSettings(self)

        # 将controlWidget添加到verticalLayout_10中
        self.setMicaEffectEnabled(cfg.get(cfg.micaEnabled))
        signalBus.micaEnableChanged.connect(self.setMicaEffectEnabled)
        cfg.themeChanged.connect(self.centralwidget.setthemeicon)
        # create sub interface
        # self.appInterface = Widget('Application Interface', self)

        self.initNavigation()
        self.initWindow()
        self.showMaximized()

    def initNavigation(self):
        self.addSubInterface(self.centralwidget, FIF.HOME, '主页', FIF.HOME_FILL)
        self.addSubInterface(self.labelsInterface, FIF.TAG, '异常标签')
        self.addSubInterface(self.identifyResults, FIF.LIBRARY, '识别结果')
        self.addSubInterface(self.usersInterface, FIF.PEOPLE, '用户')

        self.addSubInterface(self.SettingInterface, FIF.SETTING, '设置', FIF.SETTING,
                             NavigationItemPosition.BOTTOM)
        self.navigationInterface.addItem(
            routeKey='Help',
            icon=FIF.HELP,
            text='帮助',
            onClick=lambda: QDesktopServices.openUrl(QUrl(MY_URL)),
            selectable=False,
            position=NavigationItemPosition.BOTTOM,
        )
        self.stackedWidget.currentChanged.connect(self.refreshresultItem)
        self.navigationInterface.setCurrentItem(self.centralwidget.objectName())

    def refreshresultItem(self, index):
        if index == 2:
            self.identifyResults.showExp()

    def initWindow(self):
        self.resize(900, 700)
        if qconfig.theme == Theme.LIGHT:
            self.setWindowIcon(QIcon('resources/UI/logo_new1.ico'))
        else:
            self.setWindowIcon(QIcon('resources/UI/logo_new2.ico'))
        self.setWindowTitle('e视平安——面向公共交通安全的人工智能守护平台')

        desktop = QApplication.desktop().availableGeometry()
        w, h = desktop.width(), desktop.height()
        self.move(w // 2 - self.width() // 2, h // 2 - self.height() // 2)


if __name__ == '__main__':
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)

    # setTheme(Theme.DARK)

    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    app.exec_()
