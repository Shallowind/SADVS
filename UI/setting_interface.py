# coding:utf-8
import json
import os
from functools import partial
from typing import Union

from PyQt5.QtCore import Qt, QUrl, pyqtSignal
from PyQt5.QtGui import QDesktopServices, QIcon
from PyQt5.QtWidgets import QWidget, QLabel, QFileDialog, QSizePolicy, QPushButton, QButtonGroup, \
    QVBoxLayout, QHeaderView, QHBoxLayout, QAbstractItemView, QVBoxLayout, QHeaderView, QHBoxLayout

from common.config import cfg, AUTHOR, VERSION, YEAR, isWin11, MY_URL, MyQuestionMessageBox
from common.signal_bus import signalBus
from common.style_sheet import StyleSheet

from qfluentwidgets import FluentIcon as FIF, DoubleSpinBox, SettingCard, SpinBox, FluentIconBase, LineEdit, \
    RadioButton, PushButton, TableWidget, PrimaryPushButton, InfoBarPosition, ColorConfigItem, MessageBox
from qfluentwidgets import InfoBar
from qfluentwidgets import (SettingCardGroup, SwitchSettingCard, OptionsSettingCard, HyperlinkCard,
                            PrimaryPushSettingCard, ScrollArea,
                            ComboBoxSettingCard, ExpandLayout, CustomColorSettingCard,
                            setTheme, setThemeColor, RangeSettingCard, ExpandSettingCard)

from common.config import cfg, AUTHOR, VERSION, YEAR, isWin11, MY_URL, MyQuestionMessageBox
from common.signal_bus import signalBus
from common.style_sheet import StyleSheet


class InformationLab(QWidget):
    change = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        # Create the layout
        self.config_dict = None
        self.layout = QVBoxLayout(self)

        self.table = TableWidget()
        self.layout.addWidget(self.table)
        self.setLayout(self.layout)

        self.table.verticalHeader().hide()
        self.table.setBorderRadius(8)
        self.table.setBorderVisible(True)

        self.table.setColumnCount(2)
        self.table.setRowCount(60)
        self.table.setHorizontalHeaderLabels(['预设名称', '操作'])

        self.fill_table()

    def fill_table(self):
        self.table.clearContents()
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSortingEnabled(True)

        try:
            with open("config.json", "r") as json_file:
                self.config_dict = json.load(json_file)
        except FileNotFoundError:
            self.config_dict = {}
        index = 0
        for defaultName in self.config_dict.keys():
            username_label = QLabel(defaultName)
            username_widget = QWidget()
            username_layout = QVBoxLayout()
            username_layout.addWidget(username_label)
            username_layout.setAlignment(Qt.AlignCenter)
            username_widget.setLayout(username_layout)
            self.table.setCellWidget(index, 0, username_widget)
            self.table.setVisible(True)

            # Action button
            change_button = PushButton("修改预设")
            change_button.clicked.connect(partial(self.chanceDefault, defaultName))
            detect_button = PushButton("删除预设")
            detect_button.clicked.connect(partial(self.deleteDefault, defaultName))
            action_button_widget = QWidget()
            action_button_layout = QHBoxLayout()
            action_button_layout.addWidget(change_button)
            action_button_layout.addWidget(detect_button)
            action_button_layout.setAlignment(Qt.AlignCenter)
            action_button_layout.setContentsMargins(0, 0, 0, 0)
            action_button_widget.setLayout(action_button_layout)
            self.table.setCellWidget(index, 1, action_button_widget)

            index = index + 1
        self.table.setFixedHeight(300)
        self.table.resizeColumnsToContents()

    def deleteDefault(self, name):
        index = 0
        for defaultName in self.config_dict.keys():
            if defaultName == name:
                del self.config_dict[defaultName]
                with open("config.json", "w") as json_file:
                    json.dump(self.config_dict, json_file)
                self.table.removeRow(index)
                return
            index = index + 1

    def chanceDefault(self, defaultName):
        self.change.emit(defaultName)


class MyCards(ExpandSettingCard):
    open = pyqtSignal()

    def __init__(self, icon: Union[str, QIcon, FIF], title: str, content: str = None, parent=None):
        super().__init__(icon, title, content, parent=parent)

    def addCard(self, card: QWidget):
        self.viewLayout.addWidget(card)

    def setExpand(self, isExpand: bool):
        super().setExpand(isExpand)
        self.adjustViewSize()
        self.open.emit()

    def adjustViewSize(self):
        self._adjustViewSize()


class MyCard(ExpandSettingCard):
    valueChanged = pyqtSignal(object)

    def __init__(self, icon: Union[str, QIcon, FIF], title: str, texts: [str], content: str = None, parent=None):
        super().__init__(icon, title, parent=parent)
        self.texts = texts or []
        self.choiceLabel = QLabel(content, self)
        self.buttonGroup = QButtonGroup(self)

        self.addWidget(self.choiceLabel)

        # create buttons
        self.viewLayout.setSpacing(19)
        self.viewLayout.setContentsMargins(48, 18, 0, 18)
        for text in texts:
            button = RadioButton(text, self.view)
            self.buttonGroup.addButton(button)
            self.viewLayout.addWidget(button)

        self._adjustViewSize()
        self.buttonGroup.buttonClicked.connect(self.__onButtonClicked)

    def __onButtonClicked(self, button: RadioButton):
        if self.choiceLabel.text() == button.text():
            return
        self.choiceLabel.setText(button.text())
        self.valueChanged.emit(button.text())


class LineCard(SettingCard):
    def __init__(self, icon: Union[str, QIcon, FIF], title: str, content: str, lindWidget: QWidget, parent=None):
        super().__init__(icon, title, content, parent)
        lindWidget.setSizePolicy(QSizePolicy.Policy(QSizePolicy.Expanding), QSizePolicy.Policy(QSizePolicy.Expanding))
        self.hBoxLayout.addWidget(lindWidget, Qt.AlignLeft)
        self.hBoxLayout.addSpacing(16)


class ButtonCard(SettingCard):
    clicked = pyqtSignal()
    close = pyqtSignal()

    def __init__(self, text, icon: Union[str, QIcon, FluentIconBase], title=None, parent=None):
        super().__init__(icon, title, parent=parent)
        self.iconLabel.setVisible(False)
        self.saveButton = PrimaryPushButton(FIF.UPDATE, text, self)
        self.closeButton = PrimaryPushButton(FIF.CLOSE, '关闭修改', self)
        self.saveButton.clicked.connect(self.clicked.emit)
        self.closeButton.clicked.connect(self.close.emit)
        self.hBoxLayout.addWidget(self.saveButton, Qt.AlignCenter)
        self.hBoxLayout.addSpacing(65)
        self.hBoxLayout.addWidget(self.closeButton)
        self.hBoxLayout.addSpacing(65)

        self.labelLine = LineEdit(self)
        self.hBoxLayout.addWidget(self.labelLine, Qt.AlignCenter)
        self.labelLine.setReadOnly(True)
        self.labelLine.setVisible(False)
        self.closeButton.setVisible(False)


class PathCard(SettingCard):

    def __init__(self, text, icon: Union[str, QIcon, FluentIconBase], title, content=None, parent=None):
        super().__init__(icon, title, parent=parent)
        self.button = QPushButton(text, self)
        self.pathLine = LineEdit(self)
        self.pathLine.setReadOnly(True)
        self.pathLine.setText(content)
        self.hBoxLayout.addWidget(self.pathLine, Qt.AlignLeft)
        self.hBoxLayout.addSpacing(16)
        self.hBoxLayout.addWidget(self.button, 0, Qt.AlignRight)
        self.hBoxLayout.addSpacing(16)
        self.button.clicked.connect(self.buttonClicked)

    def buttonClicked(self):
        folder = QFileDialog.getExistingDirectory(
            self, self.tr("选择文件夹"), "./")
        if not folder or cfg.get(cfg.downloadFolder) == folder:
            return
        cfg.set(cfg.downloadFolder, folder)
        self.pathLine.setText(folder)


class MyCustomColorSettingCard(CustomColorSettingCard):
    def __init__(self, texts: [str], configItem: ColorConfigItem, icon: Union[str, QIcon, FluentIconBase], title: str,
                 content=None, parent=None, enableAlpha=False):
        super().__init__(configItem, icon, title, content, parent, enableAlpha)
        self.defaultRadioButton.setText(texts[0])
        self.customRadioButton.setText(texts[1])
        self.choiceLabel.setText(texts[1])
        self.customLabel.setText(texts[1])
        self.chooseColorButton.setText(texts[2])


class SettingInterface(ScrollArea):
    """ Setting interface """

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.scrollWidget = QWidget()
        self.expandLayout = ExpandLayout(self.scrollWidget)

        # setting label
        self.settingLabel = QLabel(self.tr("设置"), self)
        self.identifyGroup = SettingCardGroup(
            self.tr('识别预设'), self.scrollWidget)

        self.informationCard = MyCards(
            FIF.MIX_VOLUMES,
            self.tr('预设方案'),
            content=self.tr('配置现有预设识别方案'),
            parent=self.identifyGroup
        )
        self.informationLab = InformationLab()
        self.informationCard.addCard(self.informationLab)
        self.informationLab.change.connect(self.informationLabChange)

        self.mainCards = MyCards(
            FIF.SETTING,
            self.tr('预设设置'),
            parent=self.identifyGroup
        )

        self.identifyCard = MyCard(
            FIF.ROBOT,
            self.tr('模型选择'),
            ['yolov5', 'yolov8_BRA_DCNv3', 'yolo_slowfast'],
            content='yolov5',
            parent=self.identifyGroup
        )
        self.identifyCard.expandAni.valueChanged.connect(self.mainCards.adjustViewSize)
        for i, button in enumerate(self.identifyCard.buttonGroup.buttons()):
            if button.text() == 'yolov5':
                button.setChecked(True)
                break

        self.labelCard = MyCard(
            FIF.LABEL,
            self.tr('标签集选择'),
            texts=['yolov5'],
            content='yolov5',
            parent=self.identifyGroup
        )
        self.labelCard.expandAni.valueChanged.connect(self.mainCards.adjustViewSize)
        self.setLabels('yolov5', False)
        # self.weightsCard = MyCard(
        #     FIF.BRUSH,
        #     self.tr('权重选择'),
        #     texts=[
        #         self.tr('YOLOv5'),
        #         self.tr('YOLOv8'),
        #         self.tr('YOLO_SLOWFAST')
        #     ],
        #     content='yolov5',
        #     parent=self.identifyGroup
        # )
        self.savePathCard = PathCard(
            self.tr('选择文件夹'),
            FIF.DOWNLOAD,
            self.tr("识别结果保存路径"),
            cfg.get(cfg.downloadFolder),
            self.identifyGroup
        )

        self.numCard = OptionsSettingCard(
            cfg.numType,
            FIF.PEOPLE,
            self.tr('最大识别数量'),
            texts=[
                self.tr('20'),
                self.tr('40'),
                self.tr('80'),
                self.tr('100'),
                self.tr('200'),
            ],
            parent=self.identifyGroup
        )
        self.numCard.expandAni.valueChanged.connect(self.mainCards.adjustViewSize)

        self.PDFSavePathCard = PathCard(
            self.tr('选择文件夹'),
            FIF.DOWNLOAD,
            self.tr("识别报告保存路径"),
            cfg.get(cfg.downloadFolder),
            self.identifyGroup
        )

        self.confidenceLine = DoubleSpinBox(self)
        self.confidenceLine.setRange(0, 1)
        self.confidenceLine.setSingleStep(0.01)
        self.confidenceCard = LineCard(FIF.MIX_VOLUMES, self.tr('置信度'),
                                       self.tr('置信度越高，识别精度越高，但可能会漏掉不明显的动作'), self.confidenceLine)
        self.confidenceLine.setValue(0.4)

        self.iuoLine = DoubleSpinBox(self)
        self.iuoLine.setRange(0, 1)
        self.iuoLine.setSingleStep(0.01)
        self.iuoCard = LineCard(FIF.MIX_VOLUMES, self.tr('IOU阈值'), self.tr('IOU值越高，两个物体被看做一个所需要重叠程度越大'), self.iuoLine)
        self.iuoLine.setValue(0.5)

        self.lineSizeLine = SpinBox(self)
        self.lineSizeLine.setRange(0, 10)
        self.lineSizeLine.setSingleStep(1)
        self.lineSizeCard = LineCard(FIF.FONT_SIZE, self.tr('识别框线条大小'), self.tr('控制最终显示在识别结果的标识框的粗细'),
                                     self.lineSizeLine)
        self.lineSizeLine.setValue(3)

        self.saveCard = ButtonCard(
            self.tr('使用当前设置新建预设方案'),
            FIF.SAVE,
            parent=self.identifyGroup
        )
        self.saveCard.clicked.connect(self.saveClicked)
        self.saveCard.close.connect(self.closeChange)

        self.mainCards.addCard(self.saveCard)
        self.mainCards.addCard(self.identifyCard)
        self.mainCards.addCard(self.labelCard)
        self.mainCards.addCard(self.savePathCard)
        self.mainCards.addCard(self.numCard)
        self.mainCards.addCard(self.PDFSavePathCard)
        self.mainCards.addCard(self.confidenceCard)
        self.mainCards.addCard(self.iuoCard)
        self.mainCards.addCard(self.lineSizeCard)
        # self.buLine = PushButton(self)
        # self.iuoCard = LineCard(FIF.DOWNLOAD, self.tr('IOU阈值'), self.buLine)
        # self.buttonCard = LineCard(
        #     FIF.SAVE,
        #     self.tr('保存'),
        #     self.identifyGroup
        # )
        # self.identifyGroup.addCard(self.buttonCard)

        # music folders
        # self.musicInThisPCGroup = SettingCardGroup(
        #     self.tr('此 PC 上的音乐'), self.scrollWidget)
        # self.musicFolderCard = FolderListSettingCard(
        #     cfg.musicFolders,
        #     self.tr("本地音乐库"),
        #     directory=QStandardPaths.writableLocation(
        #         QStandardPaths.MusicLocation),
        #     parent=self.musicInThisPCGroup
        # )
        # self.downloadFolderCard = PushSettingCard(
        #     self.tr('选择文件夹'),
        #     FIF.DOWNLOAD,
        #     self.tr("下载目录"),
        #     cfg.get(cfg.downloadFolder),
        #     self.musicInThisPCGroup
        # )

        # personalization
        self.personalGroup = SettingCardGroup(
            self.tr('个性化'), self.scrollWidget)
        self.micaCard = SwitchSettingCard(
            FIF.TRANSPARENT,
            self.tr('启用亚克力效果'),
            self.tr('亚克力效果的视觉体验更好，但可能导致窗口卡顿'),
            cfg.micaEnabled,
            self.personalGroup
        )
        self.themeCard = OptionsSettingCard(
            cfg.themeMode,
            FIF.BRUSH,
            self.tr('应用主题'),
            self.tr("调整你的应用外观（建议使用深色外观）"),
            texts=[
                self.tr('浅色'), self.tr('深色'),
                self.tr('跟随系统设置')
            ],
            parent=self.personalGroup
        )
        self.themeColorCard = MyCustomColorSettingCard(
            ['默认颜色', '自定义颜色', '选择颜色'],
            cfg.themeColor,
            FIF.PALETTE,
            self.tr('主题色'),
            self.tr('调整你的应用主题颜色'),
            self.personalGroup
        )
        self.themeColorCard.card.contentLabel.setText(self.tr('默认颜色'))
        # self.zoomCard = OptionsSettingCard(
        #     cfg.dpiScale,
        #     FIF.ZOOM,
        #     self.tr("界面缩放"),
        #     self.tr("调整组件和字体的大小"),
        #     texts=[
        #         "100%", "125%", "150%", "175%", "200%",
        #         self.tr("系统默认")
        #     ],
        #     parent=self.personalGroup
        # )
        self.languageCard = ComboBoxSettingCard(
            cfg.language,
            FIF.LANGUAGE,
            self.tr('语言'),
            self.tr('更改UI界面语言'),
            # texts=['简体中文', '繁體中文', 'English', self.tr('系统默认')],
            texts=['简体中文', self.tr('系统默认')],
            parent=self.personalGroup
        )

        # material
        self.materialGroup = SettingCardGroup(
            self.tr('材质'), self.scrollWidget)
        self.blurRadiusCard = RangeSettingCard(
            cfg.blurRadius,
            FIF.ALBUM,
            self.tr('亚克力磨砂半径'),
            self.tr('磨砂半径越大，图像越模糊'),
            self.materialGroup
        )

        # update software
        self.updateSoftwareGroup = SettingCardGroup(
            self.tr("软件更新"), self.scrollWidget)
        self.updateOnStartUpCard = SwitchSettingCard(
            FIF.UPDATE,
            self.tr('在应用程序启动时检查更新'),
            self.tr('新版本将更加稳定并拥有更多功能（建议启用此选项）'),
            configItem=cfg.checkUpdateAtStartUp,
            parent=self.updateSoftwareGroup
        )

        # application
        self.aboutGroup = SettingCardGroup(self.tr('关于'), self.scrollWidget)
        self.helpCard = HyperlinkCard(
            MY_URL,
            self.tr('打开帮助页面'),
            FIF.HELP,
            self.tr("帮助"),
            self.tr(
                '发现新功能并了解有关e视平安的使用技巧'),
            self.aboutGroup
        )
        self.feedbackCard = PrimaryPushSettingCard(
            self.tr('提供反馈'),
            FIF.FEEDBACK,
            self.tr('提供反馈'),
            self.tr('通过提供反馈帮助我们改进'),
            self.aboutGroup
        )
        self.aboutCard = PrimaryPushSettingCard(
            self.tr('检查更新'),
            FIF.INFO,
            self.tr('关于'),
            '© ' + self.tr('Copyright') + f" {YEAR}, {AUTHOR}. " +
            self.tr('当前版本') + " " + VERSION,
            self.aboutGroup
        )

        self.__initWidget()

    def informationLabChange(self, name):
        self.saveCard.saveButton.setText('保存修改')
        self.informationCard.toggleExpand()
        if not self.mainCards.isExpand:
            self.mainCards.toggleExpand()
        try:
            with open("config.json", "r") as json_file:
                config_dict = json.load(json_file)
        except FileNotFoundError:
            config_dict = {}
        config = config_dict[name]
        self.saveCard.labelLine.setText('正在修改: ' + name)
        self.saveCard.labelLine.setVisible(True)
        self.saveCard.closeButton.setVisible(True)
        self.setLabels(config['model_select'])
        # 设置当前被点击的按钮
        for i, button in enumerate(self.identifyCard.buttonGroup.buttons()):
            if button.text() == config['model_select']:
                button.setChecked(True)
                break
        self.identifyCard.choiceLabel.setText(config['model_select'])
        for i, button in enumerate(self.labelCard.buttonGroup.buttons()):
            if button.text() == config['labels']:
                button.setChecked(True)
                break
        self.labelCard.choiceLabel.setText(config['labels'])
        self.savePathCard.pathLine.setText(config['save_path'])
        for i, button in enumerate(self.numCard.buttonGroup.buttons()):
            if button.text() == str(config['max_det']):
                button.setChecked(True)
                break
        self.numCard.choiceLabel.setText(str(config['max_det']))
        self.PDFSavePathCard.pathLine.setText(config['pdf_save_path'])
        self.confidenceLine.setValue(config['conf'])
        self.iuoLine.setValue(config['iou'])
        self.lineSizeLine.setValue(config['line_thickness'])

    def __initWidget(self):
        self.resize(1000, 800)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setViewportMargins(0, 80, 0, 20)
        self.setWidget(self.scrollWidget)
        self.setWidgetResizable(True)
        self.setObjectName('settingInterface')

        # initialize style sheet
        self.scrollWidget.setObjectName('scrollWidget')
        self.settingLabel.setObjectName('settingLabel')
        StyleSheet.SETTING_INTERFACE.apply(self)

        self.micaCard.setEnabled(isWin11())

        # initialize layout
        self.__initLayout()
        self.__connectSignalToSlot()

    def __initLayout(self):
        self.settingLabel.move(36, 30)

        # add cards to group
        self.identifyGroup.addSettingCard(self.informationCard)
        self.identifyGroup.addSettingCard(self.mainCards)
        # self.identifyGroup.addSettingCard(self.identifyCard)
        # self.identifyGroup.addSettingCard(self.labelCard)
        # # self.identifyGroup.addSettingCard(self.weightsCard)
        # self.identifyGroup.addSettingCard(self.savePathCard)
        # self.identifyGroup.addSettingCard(self.highCards)
        # self.identifyGroup.addSettingCard(self.saveCard)

        # self.musicInThisPCGroup.addSettingCard(self.musicFolderCard)
        # self.musicInThisPCGroup.addSettingCard(self.downloadFolderCard)

        self.personalGroup.addSettingCard(self.micaCard)
        self.personalGroup.addSettingCard(self.themeCard)
        self.personalGroup.addSettingCard(self.themeColorCard)
        # self.personalGroup.addSettingCard(self.zoomCard)
        self.personalGroup.addSettingCard(self.languageCard)

        self.materialGroup.addSettingCard(self.blurRadiusCard)

        self.updateSoftwareGroup.addSettingCard(self.updateOnStartUpCard)

        self.aboutGroup.addSettingCard(self.helpCard)
        self.aboutGroup.addSettingCard(self.feedbackCard)
        self.aboutGroup.addSettingCard(self.aboutCard)

        # add setting card group to layout
        self.expandLayout.setSpacing(28)
        self.expandLayout.setContentsMargins(36, 10, 36, 0)
        self.expandLayout.addWidget(self.identifyGroup)
        # self.expandLayout.addWidget(self.musicInThisPCGroup)
        self.expandLayout.addWidget(self.personalGroup)
        self.expandLayout.addWidget(self.materialGroup)
        self.expandLayout.addWidget(self.updateSoftwareGroup)
        self.expandLayout.addWidget(self.aboutGroup)

    def __connectSignalToSlot(self):
        """ connect signal to slot """
        cfg.appRestartSig.connect(self.__showRestartTooltip)

        self.identifyCard.valueChanged.connect(lambda c: self.setLabels(c))
        # self.weightsCard.optionChanged.connect(lambda c: setWeights(cfg.get(c)))
        # self.labelCard.optionChanged.connect(lambda c: setLabel(cfg.get(c)))

        # music in the pc
        # self.downloadFolderCard.clicked.connect(self.__onDownloadFolderCardClicked)

        # personalization
        self.themeCard.optionChanged.connect(lambda ci: setTheme(cfg.get(ci)))
        self.themeColorCard.colorChanged.connect(lambda c: setThemeColor(c))
        self.micaCard.checkedChanged.connect(signalBus.micaEnableChanged)

        # about
        self.feedbackCard.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(MY_URL)))
        self.aboutCard.clicked.connect(
            lambda: InfoBar.success(
                title='成功！',
                content='当前已经是最新版本',
                orient=Qt.Horizontal,
                isClosable=True,
                position=InfoBarPosition.BOTTOM_RIGHT,
                # position='Custom',   # NOTE: use custom info bar manager
                duration=2000,
                parent=self.parent().parent().parent()
            )
        )
        # 匿名函数

    def __showRestartTooltip(self):
        """ show restart tooltip """
        InfoBar.success(
            self.tr(''),
            self.tr('重启后生效'),
            duration=1500,
            parent=self
        )

    def __onDownloadFolderCardClicked(self):
        """ download folder card clicked slot """
        folder = QFileDialog.getExistingDirectory(
            self, self.tr("选择文件夹"), "./")
        if not folder or cfg.get(cfg.downloadFolder) == folder:
            return

        cfg.set(cfg.downloadFolder, folder)
        self.downloadFolderCard.setContent(folder)

    def closeChange(self):
        self.saveCard.labelLine.setVisible(False)
        self.saveCard.labelLine.setText('')
        self.saveCard.closeButton.setVisible(False)
        self.saveCard.saveButton.setText('使用当前设置新建预设方案')

    def saveClicked(self):
        # Gather values from UI elements
        model_select = self.identifyCard.choiceLabel.text()
        labels = self.labelCard.choiceLabel.text()
        # pt_path = self.weightsCard.choiceLabel.text()
        save_path = self.savePathCard.pathLine.text()
        max_det = int(self.numCard.choiceLabel.text())  # Assuming it's an integer
        pdf_save_path = self.PDFSavePathCard.pathLine.text()
        conf = float(self.confidenceLine.text())
        iou = float(self.iuoLine.text())
        line_thickness = int(self.lineSizeLine.text())

        # Create a dictionary with the gathered values
        config_dict_new = {
            "model_select": model_select,
            "labels": labels,
            "pt_path": model_select,
            "save_path": save_path,
            "max_det": max_det,
            "pdf_save_path": pdf_save_path,
            "conf": conf,
            "iou": iou,
            "line_thickness": line_thickness
        }
        text = self.saveCard.labelLine.text()

        if text:
            config_id = text.split('正在修改: ')[-1]
        else:
            # 新建
            # timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            # config_id = f"config_{timestamp}"
            k = MyQuestionMessageBox('新建识别预设', '请输入新建预设的名称', self)
            if k.exec_() == MyQuestionMessageBox.Accepted:
                config_id = k.LineEdit.text()
            else:
                return

        self.saveCard.labelLine.setVisible(False)
        self.saveCard.closeButton.setVisible(False)
        self.saveCard.labelLine.setText('')

        # 加载或创建主配置字典
        try:
            with open("config.json", "r") as json_file:
                config_dict = json.load(json_file)
        except FileNotFoundError:
            config_dict = {}

        config_dict[config_id] = config_dict_new

        # Dump the dictionary to a JSON file
        with open("config.json", "w") as json_file:
            json.dump(config_dict, json_file, indent=2)
        self.informationLab.fill_table()

        InfoBar.success(
            title='成功！',
            content='成功新建预设模型',
            orient=Qt.Horizontal,
            isClosable=True,
            position=InfoBarPosition.BOTTOM_RIGHT,
            # position='Custom',   # NOTE: use custom info bar manager
            duration=2000,
            parent=self.parent().parent().parent()
        )

    def setLabels(self, identify, checkExpand=False):
        texts = []
        if identify == 'yolov5':
            path = 'labels/yolov5'
            content = 'yolov5'
        elif identify == 'yolov8_BRA_DCNv3':
            w = MessageBox(
                title='警告——实验性功能',
                content='使用yolov8_BRA_DCNv3模型为实验性功能，特训后的模型对密集人群卓有成效，但本应用暂未完整适配，仅可使用部分功能，敬请期待更新。',
                parent=self
            )
            w.show()
            path = 'labels/yolov8_BRA_DCNv3'
            content = 'yolov8_BRA_DCNv3'
        elif identify == 'yolo_slowfast':
            path = 'labels/yolo_slowfast'
            content = 'yolo_slowfast'
        else:
            return
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found")
        for file in os.listdir(path):
            if file.endswith('.pbtxt'):
                texts.append(file.split('.')[0])
        labelCard = MyCard(
            FIF.LABEL,
            self.tr('标签集选择'),
            texts=texts,
            content=content,
            parent=self.identifyGroup
        )
        labelCard.expandAni.valueChanged.connect(self.mainCards.adjustViewSize)
        for i, button in enumerate(labelCard.buttonGroup.buttons()):
            if button.text() == content:
                button.setChecked(True)
                break
        # labelCard替换原本的self.labelCard
        index = self.mainCards.viewLayout.indexOf(self.labelCard)
        self.labelCard.setParent(None)
        self.labelCard = labelCard
        self.mainCards.viewLayout.insertWidget(index, labelCard)
        if checkExpand and not self.labelCard.isExpand:
            self.labelCard.toggleExpand()