import os
import pickle
import sys

from PyQt5.QtCore import Qt, QUrl, QTimer
from PyQt5.QtGui import QPixmap, QIcon, QImage
from PyQt5.QtWidgets import QApplication, QMainWindow, QInputDialog, QLineEdit, QWidget
from qfluentwidgets import setTheme, Theme, MessageBoxBase, SubtitleLabel, LineEdit, qconfig
from qframelesswindow import FramelessWindow, StandardTitleBar

from common.style_sheet import StyleSheet
from utils.myutil import Globals
from Users import UserManager
# from UI.user_management_ui import Ui_user_management
from .testui import Ui_Form


class CustomMessageBox(MessageBoxBase):
    """ Custom message box """

    def __init__(self, str1, str2, parent=None):
        super().__init__(parent)
        self.titleLabel = SubtitleLabel(str1, self)
        self.urlLineEdit = LineEdit(self)

        self.urlLineEdit.setPlaceholderText(str2)
        self.urlLineEdit.setClearButtonEnabled(True)

        # add widget to view layout
        self.viewLayout.addWidget(self.titleLabel)
        self.viewLayout.addWidget(self.urlLineEdit)

        # change the text of button
        self.yesButton.setText('打开')
        self.cancelButton.setText('取消')

        self.widget.setMinimumWidth(100)
        self.yesButton.setDisabled(True)
        self.urlLineEdit.textChanged.connect(self._validateUrl)

        # self.hideYesButton()

    def _validateUrl(self, text):
        self.yesButton.setEnabled(QUrl(text).isValid())


class User_management(Ui_Form, FramelessWindow):
    def __init__(self, controller):
        super(User_management, self).__init__()

        # 加载UI文件
        self.dragPos = None
        self.controller = controller
        self.setupUi(self)
        self.manager = UserManager()
        self.saveuser = {}
        self.db_path = 'saved'
        self.saved = False
        # setTheme(Theme.DARK)
        self.setTitleBar(StandardTitleBar(self))

        StyleSheet.Demo.apply(self)

        self.ComboBox.addItems(['外伤性疾病：脊柱骨折：如腰椎压缩性骨折、胸椎粉碎性骨折等。脊柱脱位：尤其是颈椎脱位。',
                                '先天性和发育性疾病：先天性脊柱畸形：如半椎体畸形、脊柱侧弯、脊柱裂等。发育性异常：包括各种影响脊柱正常发育的因素。',
                                '脊柱的退行性病变：颈椎病：常见于中老年人，因颈椎间盘和椎体结构随年龄变化而退化。腰椎间盘突出：椎间盘的退变导致压迫神经根。椎管狭窄：可发生在颈椎或腰椎，导致神经受压。',
                                '炎症性疾病：化脓性脊柱炎、化脓性间盘炎。特殊感染和脊柱结核。少见的其他感染。',
                                '肿瘤性疾病：原发性脊柱肿瘤，如脊柱的成骨肉瘤较少见。转移癌，较原发性肿瘤更为常见。多发性骨髓瘤等血液系统肿瘤对脊柱的影响。',
                                '代谢性与免疫性疾病：骨质疏松症，影响脊柱的强度和稳定性。强直性脊柱炎，一种慢性炎症性疾病，影响脊柱和骶髂关节。类风湿性关节炎，可能累及脊柱的小关节。'])
        self.ComboBox_3.addItems(['按解剖区域分类', '按功能特性分类'])
        self.ComboBox_3.setCurrentIndex(0)
        self.ComboBox_2.addItems(['颈椎模型:C1、C2、C3、C4、C5、C6、C7 可具体到哪几个节段',
                                  '胸椎模型:同样分为T1,T2…T12',
                                  '腰椎模型:L1.L2.L3,L4,L5',
                                  '骶尾骨模型:单独或结合骨盆展示骶骨和尾骨的结构。',
                                  '全脊柱模型:从颈椎至骶尾骨的完整脊柱模型，可含骨盆。'])
        self.ComboBox_3.currentIndexChanged.connect(self.show_text)
        self.image_timer = QTimer()
        self.image_timer.timeout.connect(self.showimage)
        self.image_timer.start(1000)
        self.PushButton.clicked.connect(self.showimage)

    def showimage(self):
        # 加载图片
        showImage = QImage("C:/Users/sodetensonkid/Pictures/Saved Pictures/2.png")
        scale_factor = min(self.player_2.width() / showImage.width(),
                           self.player_2.height() / showImage.height())
        # 计算新的宽度和高度
        print(self.player_2.width(), self.player_2.height())
        new_width = int(showImage.width() * scale_factor)
        new_height = int(showImage.height() * scale_factor)
        print(new_width, new_height)
        # 设置新的最大大小
        self.camera_2.setMaximumSize(new_width, new_height)
        self.camera_2.setPixmap(QPixmap(showImage))
        self.camera_2.setScaledContents(True)

    def show_text(self):
        if self.ComboBox_3.currentIndex() == 0:
            self.ComboBox_2.clear()
            self.ComboBox_2.addItems(['颈椎模型:C1、C2、C3、C4、C5、C6、C7 可具体到哪几个节段',
                                      '胸椎模型:同样分为T1,T2…T12',
                                      '腰椎模型:L1.L2.L3,L4,L5',
                                      '骶尾骨模型:单独或结合骨盆展示骶骨和尾骨的结构。',
                                      '全脊柱模型:从颈椎至骶尾骨的完整脊柱模型，可含骨盆。'])
        elif self.ComboBox_3.currentIndex() == 1:
            self.ComboBox_2.clear()
            self.ComboBox_2.addItems(['基础教学模型:展示脊柱的基本解剖结构，适合初学者学习。',
                                      '高级病理模型:定制特定脊柱疾病，如椎间盘突出、脊柱侧弯、管狭窄等。',
                                      '活动或可动模型:关节可动，便于演示脊柱的运动范围和特定病理状态。',
                                      '手术模拟模型:专为手术技能培训设计，一比一仿生，可拟手术操作过程。'])



if __name__ == '__main__':
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
    app = QApplication(sys.argv)
    window = User_management()
    window.show()
    sys.exit(app.exec_())
