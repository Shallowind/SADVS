import os
import pickle
import sys

from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtWidgets import QApplication, QMainWindow, QInputDialog, QLineEdit, QWidget
from qfluentwidgets import setTheme, Theme, MessageBoxBase, SubtitleLabel, LineEdit, qconfig
from qframelesswindow import FramelessWindow, StandardTitleBar

from common.style_sheet import StyleSheet
from utils.myutil import Globals
from Users import UserManager
# from UI.user_management_ui import Ui_user_management
from .user_management_ui import Ui_user_management


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


class User_management(Ui_user_management, FramelessWindow):
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
        self.comboBox.addItem('管理员')
        self.comboBox.addItem('普通用户')
        self.setTitleBar(StandardTitleBar(self))

        self.label_3.setPixmap(QPixmap("resources/UI/logo_new3.png"))
        StyleSheet.Demo.apply(self)
        self.resize(350, 600)
        self.setResizeEnabled(False)
        if os.path.exists(self.db_path):
            with open(self.db_path, 'rb') as f:
                self.saved = True
                users = pickle.load(f)
                self.username.setText(list(users.keys())[0])
                self.password.setText('*' * list(users.values())[0])
                self.checkBox.setChecked(True)
        else:
            self.users = {}

        self.level.setVisible(False)
        self.comboBox.setVisible(False)
        self.passwordlab_2.setVisible(False)
        self.password_2.setVisible(False)
        self.signupbutton.setVisible(False)
        self.signin.setVisible(False)
        self.resize(300, 500)
        if qconfig.theme == Theme.LIGHT:
            self.setWindowIcon(QIcon('resources/UI/logo_new1.ico'))
        else:
            self.setWindowIcon(QIcon('resources/UI/logo_new2.ico'))
        self.setWindowTitle('e视平安')
        # 设置窗口标志，去掉窗口边框
        # self.setWindowFlags(Qt.FramelessWindowHint)
        self.password.textChanged.connect(self.on_password_changed)
        # 登录
        self.signin.clicked.connect(self.Signin)
        self.signinbutton.clicked.connect(self.Login)
        # 注册
        self.signup.clicked.connect(self.Signup)
        self.signupbutton.clicked.connect(self.Logup)

        self.forget.clicked.connect(self.forgetpassword)
        # self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())

        # 设置窗口背景颜色为白色
        # self.centralwidget.setStyleSheet("background-color: #19232d;")

    def on_password_changed(self):
        if self.saved:
            self.saved = False
            self.password.clear()
            self.warning.clear()
            if os.path.exists(self.db_path):
                os.remove(self.db_path)

    def Logup(self):
        username = self.username.text()
        password = self.password.text()
        password_2 = self.password_2.text()
        level = self.comboBox.currentText()
        if level == '管理员':
            is_admin = True
        else:
            is_admin = False
        try:
            if username == '' or password == '' or password_2 == '':
                self.warning.setStyleSheet("color: red;")
                self.warning.setText('用户名或密码不能为空')
            elif password != password_2:
                self.warning.setStyleSheet("color: red;")
                self.warning.setText('两次密码不一致')
            else:
                if is_admin:
                    w = CustomMessageBox('注册管理员账户', '超级管理员密码:', self)
                    if w.exec():
                        if self.manager.check_password('superadmin', w.urlLineEdit.text()):
                            self.manager.add_user(username, password, is_admin)
                            self.warning.setStyleSheet("color: green;")
                            self.warning.setText('注册成功')
                        else:
                            self.warning.setStyleSheet("color: red;")
                            self.warning.setText('密码错误')
                else:
                    self.manager.add_user(username, password, is_admin)
                    self.warning.setStyleSheet("color: green;")
                    self.warning.setText('注册成功')
        except Exception as e:
            self.warning.setStyleSheet("color: red;")
            self.warning.setText(str(e))

    def Login(self):
        username = self.username.text()
        password = self.password.text()
        try:
            if username == '' or password == '':
                self.warning.setStyleSheet("color: red;")
                self.warning.setText('用户名或密码不能为空')
            elif self.saved:
                self.warning.setStyleSheet("color: green;")
                self.warning.setText('登录成功')
                self.close()
                Globals.user = self.manager.users[username]
                self.controller.show_mainui()

            elif self.manager.check_password(username, password) is True:
                self.warning.setStyleSheet("color: green;")
                self.warning.setText('登录成功')
                if self.checkBox.isChecked():
                    self.saveuser = {username: len(password)}
                    with open(self.db_path, 'wb') as f:
                        pickle.dump(self.saveuser, f)
                else:
                    if os.path.exists(self.db_path):
                        os.remove(self.db_path)
                Globals.user = self.manager.users[username]
                self.controller.show_mainui()
            else:
                self.warning.setStyleSheet("color: red;")
                self.warning.setText('用户名或密码错误')
        except Exception as e:
            self.warning.setStyleSheet("color: red;")
            self.warning.setText(str(e))
        # else:
        #     self.warning.setStyleSheet("color: red;")
        #     self.warning.setText(self.manager.check_password(username, password))

    def Signup(self):
        self.signup.setVisible(False)
        self.signin.setVisible(True)
        self.signinbutton.setVisible(False)
        self.signupbutton.setVisible(True)
        self.level.setVisible(True)
        self.comboBox.setVisible(True)
        self.passwordlab_2.setVisible(True)
        self.password_2.setVisible(True)
        self.checkBox.setVisible(False)
        self.username.clear()
        self.password.clear()

    def Signin(self):
        self.signin.setVisible(False)
        self.signup.setVisible(True)
        self.signupbutton.setVisible(False)
        self.signinbutton.setVisible(True)
        self.level.setVisible(False)
        self.comboBox.setVisible(False)
        self.passwordlab_2.setVisible(False)
        self.password_2.setVisible(False)
        self.checkBox.setVisible(True)

    def close(self):
        super(User_management, self).close()

    def mouseMoveEvent(self, event):
        # 实现窗口拖动
        if hasattr(self, 'dragPos'):
            if self.dragPos is not None:
                newPos = self.mapToGlobal(event.pos() - self.dragPos)
            else:
                # 处理 self.dragPos 为 None 的情况
                return

            self.move(self.mapToGlobal(newPos))
            self.dragPos = event.globalPos()
            event.accept()

    def forgetpassword(self):
        username = self.username.text()
        if username == '':
            self.warning.setStyleSheet("color: red;")
            self.warning.setText('用户名不能为空')
        # 弹出输入超级管理员密码，正确即可修改密码
        w = CustomMessageBox('忘记密码', '超级管理员密码:', self)
        if w.exec():
            if self.manager.check_password('superadmin', w.urlLineEdit.text()):
                # 弹出输入新密码对话框
                w = CustomMessageBox('忘记密码', '输入新密码:', self)
                if w.exec():
                    self.manager.change_password(self.username.text(), w.urlLineEdit.text())
                    self.warning.setStyleSheet("color: green;")
                    self.warning.setText('密码修改成功')
            else:
                self.warning.setStyleSheet("color: red;")
                self.warning.setText('密码错误')


if __name__ == '__main__':
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
    app = QApplication(sys.argv)
    window = User_management()
    window.show()
    sys.exit(app.exec_())
