import datetime
import os

from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtWidgets import QWidget, QListWidgetItem, QSizePolicy
from qfluentwidgets import InfoBar, InfoBarPosition

from PDF import PDFReader
from UI.identifyResults_ui import Ui_IdentifyResults
from Users import UserManager
from utils.myutil import Globals


def getExpList():
    # 遍历文件夹中的所有图片文件
    base_path = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.join(base_path, "exception")
    # 创建
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    pathlist = []
    for filename in os.listdir(base_path):
        if filename.startswith('ep'):
            pathlist.append(filename)
    if len(pathlist) == 0:
        return
    pathlist.sort(key=lambda x: int(x.split('ep')[-1]))
    return pathlist


class IdentifyResults(Ui_IdentifyResults, QWidget):

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.manager = UserManager()
        self.expItemList.clicked.connect(self.expItemClicked)
        self.expList.clicked.connect(self.expClicked)
        self.expItemList.setIconSize(QSize(75, 75))
        self.showExp()
        self.pdfReader = None

        self.SegmentedWidget.addItem(
            routeKey="exp_widget",
            text="异常显示",
            onClick=lambda: self.stackedWidget.setCurrentWidget(self.expWidget),
        )
        self.SegmentedWidget.addItem(
            routeKey="pdf_widget",
            text="识别报告",
            onClick=lambda: (
                self.stackedWidget.setCurrentWidget(self.PDFWidget),
                self.showPDF()
            )
        )
        self.SegmentedWidget.setCurrentItem("exp_widget")
        self.stackedWidget.setCurrentWidget(self.expWidget)


    def showPDF(self):
        if self.pdfReader:
            self.pdfReader.setParent(None)
        item = self.expList.currentItem()
        if not item:
            return
        path = item.data(Qt.UserRole)
        base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'exception')
        folder_path = os.path.join(base_path, path)

        pdfName = ""
        for file in os.listdir(folder_path):
            if file.endswith(".pdf"):
                pdfName = file
                break
        if not pdfName:
            return
        self.pdfReader = PDFReader(os.path.join(folder_path, pdfName))
        self.gridLayout_5.addWidget(self.pdfReader)
        # 设置布局的尺寸策略
        self.pdfReader.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.PDFWidget.setLayout(self.gridLayout_5)

    def showExp(self):
        self.expList.clear()
        expPath = getExpList()
        base_path = os.path.dirname(os.path.abspath(__file__))
        base_path = os.path.join(base_path, "exception")
        if expPath:
            for path in expPath:
                path_new = os.path.join(base_path, path)
                creation_time = os.path.getctime(path_new)
                creation_date = datetime.datetime.fromtimestamp(creation_time).strftime('%Y-%m-%d %H:%M:%S')

                item = QListWidgetItem(creation_date)
                item.setData(Qt.UserRole, path)
                self.expList.addItem(item)
        # 模拟自动点击expList最后一个item
        if self.expList.count() > 0:
            self.expList.setCurrentRow(self.expList.count() - 1)
            self.expClicked()
        user = Globals.user
        if user is not None and not user.is_admin:
            self.expList.setEnabled(False)
            InfoBar.warning(
                title='警告！',
                content="非管理员用户无权查看过往异常记录",
                orient=Qt.Horizontal,
                isClosable=True,
                position=InfoBarPosition.BOTTOM_RIGHT,
                # position='Custom',   # NOTE: use custom info bar manager
                duration=2000,
                parent=self.parent()
            )

    def expItemClicked(self):
        item = self.expItemList.currentItem()
        if item:
            path = item.data(Qt.UserRole)
            if path:
                try:
                    # 显示异常图片
                    pixmap = QPixmap(path.split('.')[0] + 'all.jpg')
                    self.expplayer_2.setAlignment(Qt.AlignCenter)
                    self.expplayer_2.setPixmap(pixmap.scaled(self.expplayer_2.size(), aspectRatioMode=True))
                    # 打开.txt文件并读取内容
                    self.exp_inf_2.clear()
                    self.exp_type_2.clear()
                    with open(path.split('.')[0] + '.txt', 'r') as file:
                        text = file.read()
                    # 将文件内容写入textEdit_2中
                    self.exp_inf_2.setText(text)
                    path1 = os.path.join(os.path.dirname(path), os.path.basename(os.path.dirname(path)) + '.txt')
                    # print(os.path.dirname(path) + '.txt')
                    with open(path1, 'r') as file:
                        text = file.read()
                    self.exp_type_2.setText(text)
                    self.exp_inf_2.setReadOnly(True)
                    self.exp_type_2.setReadOnly(True)
                except Exception as e:
                    # 处理错误，例如无效的图像路径
                    print(f"设置图像时发生错误：{e}")

    def expClicked(self):
        if self.stackedWidget.currentWidget() == self.PDFWidget:
            self.showPDF()
        else:
            item = self.expList.currentItem()
            if item is None:
                return
            # 清空列表以便重新加载
            self.expItemList.clear()
            path = item.data(Qt.UserRole)
            base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'exception')
            folder_path = os.path.join(base_path, path)

            for filename in os.listdir(folder_path):
                if filename.endswith(('.jpg', '.png', '.jpeg', '.bmp')):  # 仅处理图片文件
                    if filename.endswith('all.jpg'):
                        continue
                    # 创建 QListWidgetItem
                    item = QListWidgetItem()
                    path = os.path.join(folder_path, filename)
                    item.setData(Qt.UserRole, path.split('all')[0])
                    # 创建 QPixmap 以便在列表中显示缩略图
                    pixmap = QPixmap(path)
                    icon = QIcon(pixmap)
                    item.setIcon(icon)
                    with open(path.split('.')[0].split('all')[0] + '.txt', 'r') as file:
                        files = file.read()
                    text = files.split('动作 :')[-1].split('\n')[0]
                    if text == '无' or text == '':
                        text = files.split('类别 :')[-1].split('\n')[0]
                    item.setTextAlignment(Qt.AlignRight)
                    item.setText(text)
                    # 将 QListWidgetItem 添加到 QListWidget
                    self.expItemList.addItem(item)
