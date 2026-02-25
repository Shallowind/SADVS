# coding:utf-8
import sys
from enum import Enum

from PyQt5.QtCore import QLocale, QUrl
from qfluentwidgets import (qconfig, QConfig, ConfigItem, OptionsConfigItem, BoolValidator,
                            OptionsValidator, RangeConfigItem, RangeValidator,
                            FolderListValidator, Theme, FolderValidator, ConfigSerializer, __version__, EnumSerializer,
                            MessageBoxBase, SubtitleLabel, LineEdit)


class Language(Enum):
    """ Language enumeration """

    CHINESE_SIMPLIFIED = QLocale(QLocale.Chinese, QLocale.China)
    # CHINESE_TRADITIONAL = QLocale(QLocale.Chinese, QLocale.HongKong)
    # ENGLISH = QLocale(QLocale.English)
    AUTO = QLocale()


class Identify(Enum):
    YOLOV5 = 'YOLOv5'
    YOLOV8 = 'YOLOv8'
    YOLO_SLOWFAST = 'YOLO_SLOWFAST'


class Label(Enum):
    YOLOV5 = 'YOLOv5'
    YOLOV8 = 'YOLOv8'
    YOLO_SLOWFAST = 'YOLO_SLOWFAST'


class Num(Enum):
    NUM_20 = 20
    NUM_40 = 40
    NUM_80 = 80
    NUM_100 = 100
    NUM_200 = 200


class LanguageSerializer(ConfigSerializer):
    """ Language serializer """

    def serialize(self, language):
        return language.value.name() if language != Language.AUTO else "Auto"

    def deserialize(self, value: str):
        return Language(QLocale(value)) if value != "Auto" else Language.AUTO


def isWin11():
    return sys.platform == 'win32' and sys.getwindowsversion().build >= 22000


class Config(QConfig):
    """ Config of application """

    # folders
    musicFolders = ConfigItem(
        "Folders", "LocalMusic", [], FolderListValidator())
    downloadFolder = ConfigItem(
        "Folders", "Download", "app/download", FolderValidator())

    # main window
    micaEnabled = ConfigItem("SettingInterface", "MicaEnabled", isWin11(), BoolValidator())
    dpiScale = OptionsConfigItem(
        "MainWindow", "DpiScale", "Auto", OptionsValidator([1, 1.25, 1.5, 1.75, 2, "Auto"]), restart=True)
    language = OptionsConfigItem(
        "MainWindow", "Language", Language.AUTO, OptionsValidator(Language), LanguageSerializer(), restart=True)

    # Material
    blurRadius = RangeConfigItem("Material", "AcrylicBlurRadius", 15, RangeValidator(0, 40))

    # software update
    checkUpdateAtStartUp = ConfigItem("Update", "CheckUpdateAtStartUp", True, BoolValidator())

    identifyType = OptionsConfigItem(
        "QFluentWidgets", "identifyType", Identify.YOLOV5, OptionsValidator(Identify), EnumSerializer(Identify))

    weightsType = OptionsConfigItem(
        "QFluentWidgets", "weightsType", Identify.YOLOV5, OptionsValidator(Identify), EnumSerializer(Identify))

    labelType = OptionsConfigItem(
        "QFluentWidgets", "labelType", Identify.YOLOV5, OptionsValidator(Identify), EnumSerializer(Identify))

    numType = OptionsConfigItem(
        "QFluentWidgets", "numType", 20, OptionsValidator(Num), EnumSerializer(Num))


YEAR = 2023
AUTHOR = "e视平安"
VERSION = "2.0.1"
MY_URL = "http://47.93.57.125:789"

cfg = Config()
cfg.themeMode.value = Theme.AUTO
qconfig.load('config/config.json', cfg)


class MyQuestionMessageBox(MessageBoxBase):
    """ Custom message box """

    def __init__(self, title, text, parent=None):
        super().__init__(parent)
        self.titleLabel = SubtitleLabel(title, self)
        self.LineEdit = LineEdit(self)

        self.LineEdit.setPlaceholderText(text)
        self.LineEdit.setClearButtonEnabled(True)

        # add widget to view layout
        self.viewLayout.addWidget(self.titleLabel)
        self.viewLayout.addWidget(self.LineEdit)

        # change the text of button
        self.yesButton.setText('确定')
        self.cancelButton.setText('取消')

        self.widget.setMinimumWidth(350)
        self.yesButton.clicked.connect(self.accept)

