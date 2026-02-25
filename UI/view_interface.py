# coding:utf-8
from functools import partial

from PyQt5.QtCore import Qt, QUrl, QSize
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtWidgets import (QListWidgetItem, QFrame, QTreeWidgetItem, QHBoxLayout,
                             QTreeWidgetItemIterator, QTableWidgetItem, QHeaderView, QWidget, QVBoxLayout, QLabel,
                             QPushButton, QSpacerItem, QSizePolicy)
from qfluentwidgets import TreeWidget, TableWidget, ListWidget, HorizontalFlipView, PushButton, SwitchButton, InfoBar, \
    InfoBarPosition, MessageBoxBase, SubtitleLabel, LineEdit, AvatarWidget, FluentIcon, ToolButton, TitleLabel, \
    TransparentPushButton, setCustomStyleSheet

from Users import UserManager
from utils.myutil import Globals
from .gallery_interface import GalleryInterface
from common.translator import Translator
from common.style_sheet import StyleSheet


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
        self.yesButton.setText('确定')
        self.cancelButton.setText('取消')

        self.widget.setMinimumWidth(350)
        self.yesButton.setDisabled(True)
        self.urlLineEdit.textChanged.connect(self._validateUrl)

        # self.hideYesButton()

    def _validateUrl(self, text):
        self.yesButton.setEnabled(QUrl(text).isValid())


class ViewInterface(GalleryInterface):
    """ View interface """

    def __init__(self, parent=None):
        self.controller = parent.controller
        t = Translator()
        super().__init__(
            title="用户",
            subtitle="",
            parent=parent
        )
        self.setObjectName('viewInterface')

        # list view
        self.addExampleCard(
            title=self.tr('当前用户'),
            widget=NowUserFrame(self),
            sourcePath=''
        )

        self.ManageUserFrame = ManageUserFrame(self)
        txt = self.tr('用户管理')
        user = Globals.user
        if user is not None and not user.is_admin:
            self.ManageUserFrame.setEnabled(False)
            txt = f'{txt} (需要管理员权限)'
        # table view
        self.addExampleCard(
            title=txt,
            widget=self.ManageUserFrame,
            sourcePath=''
        )


class Frame(QFrame):

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.hBoxLayout = QHBoxLayout(self)
        self.hBoxLayout.setContentsMargins(0, 8, 0, 0)

        self.setObjectName('frame')
        self.setStyleSheet("""#frame {
    border: 1px solid rgba(255, 255, 255, 13);
    border-radius: 5px;
    background-color: transparent;
}""")

    def addWidget(self, widget):
        self.hBoxLayout.addWidget(widget)


class NowUserFrame(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QHBoxLayout(self)
        self.layout.setAlignment(Qt.AlignCenter)

        # User avatar
        self.avatar = ToolButton(FluentIcon.PEOPLE)
        self.avatar.setIconSize(QSize(100, 100))
        StyleSheet.GALLERY_INTERFACE.apply(self.avatar)
        # self.avatar.setStyleSheet("background: transparent; border: none;")
        self.avatar.setFixedWidth(150)
        self.layout.addWidget(self.avatar)

        # Username
        self.label_layout = QVBoxLayout()
        self.label = TitleLabel()
        self.label.setFixedWidth(500)
        self.label.setObjectName('label')
        StyleSheet.GALLERY_INTERFACE.apply(self.label)
        # self.label.setStyleSheet("QLabel {font: 24px 'Segoe UI', 'Microsoft YaHei', 'PingFang SC';}")
        self.label_admin = QLabel()
        self.label_layout.addWidget(self.label)
        self.label_isadmin = TransparentPushButton()

        self.hbox_layout = QHBoxLayout()
        self.hbox_layout.addWidget(self.label_isadmin)
        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.hbox_layout.addItem(self.horizontalSpacer)
        self.label_layout.addLayout(self.hbox_layout)

        self.layout.addLayout(self.label_layout)

        # Buttons
        self.button_layout = QVBoxLayout()
        self.change_password_button = PushButton("修改密码")
        self.logout_button = PushButton("退出登录")
        self.button_layout.addWidget(self.logout_button)
        self.button_layout.addWidget(self.change_password_button)
        self.layout.addLayout(self.button_layout)

        self.setLayout(self.layout)
        self.controller = parent.controller
        self.manager = UserManager()
        self.update_user()

        # Connect buttons to their respective slots
        self.logout_button.clicked.connect(self.logout)
        self.change_password_button.clicked.connect(self.change_password)

    def update_user(self):
        user = Globals.user
        if user is None:
            self.label.setText('当前无用户登录')
        else:
            self.label.setText(f'用户：{user.username}')
            if self.manager.check_admin(user.username):
                self.label_isadmin.setText('管理员')
                self.label_isadmin.setIcon(FluentIcon.VPN)
            else:
                self.label_isadmin.setText('普通用户')
                self.label_isadmin.setIcon(FluentIcon.TAG)

    def logout(self):
        self.controller.show_user_management()
        self.controller.close_mainui()

    def change_password(self):
        try:
            w = CustomMessageBox('修改密码', '请输入旧密码:', self.parent().parent().parent().parent().parent())
            if w.exec():
                if not self.manager.check_password(Globals.user.username, w.urlLineEdit.text()):
                    content = "旧密码错误"
                    InfoBar.error(
                        title='Error!',
                        content=content,
                        orient=Qt.Horizontal,
                        isClosable=True,
                        position=InfoBarPosition.BOTTOM_RIGHT,
                        # position='Custom',   # NOTE: use custom info bar manager
                        duration=2000,
                        parent=self.parent().parent().parent().parent().parent()
                    )
                else:
                    w = CustomMessageBox('修改密码', '请输入新密码:', self.parent().parent().parent().parent().parent())
                    if w.exec():
                        new_password = w.urlLineEdit.text()
                        w = CustomMessageBox('修改密码', '请再次输入新密码:',
                                             self.parent().parent().parent().parent().parent())
                        if w.exec():
                            if new_password != w.urlLineEdit.text():
                                raise Exception('两次密码不一致')
                            self.manager.change_password(Globals.user.username, new_password)
                            content = f"用户 {Globals.user.username} 的密码已被修改。"
                            InfoBar.success(
                                title='成功！',
                                content=content,
                                orient=Qt.Horizontal,
                                isClosable=True,
                                position=InfoBarPosition.BOTTOM_RIGHT,
                                # position='Custom',   # NOTE: use custom info bar manager
                                duration=2000,
                                parent=self.parent().parent().parent().parent().parent()
                            )
        except Exception as e:
            content = f"修改密码时发生错误：{str(e)}"
            InfoBar.error(
                title='Error!',
                content=content,
                orient=Qt.Horizontal,
                isClosable=True,
                position=InfoBarPosition.BOTTOM_RIGHT,
                # position='Custom',   # NOTE: use custom info bar manager
                duration=2000,
                parent=self.parent().parent().parent().parent().parent()
            )


class ManageUserFrame(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        # Create the layout
        self.layout = QVBoxLayout(self)
        self.button = PushButton("添加用户")
        self.layout.addWidget(self.button)
        self.button.clicked.connect(self.add_user)
        self.table = TableWidget()
        self.layout.addWidget(self.table)
        self.setLayout(self.layout)

        self.table.verticalHeader().hide()
        self.table.setBorderRadius(8)
        self.table.setBorderVisible(True)
        self.manager = UserManager()

        self.table.setColumnCount(3)
        self.table.setRowCount(60)
        self.table.setHorizontalHeaderLabels(['用户', '管理员权限', '操作'])

        self.fill_table()

    def fill_table(self):
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSortingEnabled(True)
        self.table.setRowCount(0)
        users = self.manager.get_users()  # Assuming get_users() returns a dictionary of users
        self.table.setRowCount(len(users))

        for i, user in enumerate(users.values()):
            # Username
            username_label = QLabel(user.username)
            username_widget = QWidget()
            username_layout = QVBoxLayout()
            username_layout.addWidget(username_label)
            username_layout.setAlignment(Qt.AlignCenter)
            username_widget.setLayout(username_layout)
            self.table.setCellWidget(i, 0, username_widget)

            # Switch
            switch = SwitchButton()
            switch.setChecked(user.is_admin)
            switch.checkedChanged.connect(partial(self.set_admin, user.username, switch))
            switch_widget = QWidget()
            switch_layout = QVBoxLayout()
            switch_layout.addWidget(switch)
            switch_layout.setAlignment(Qt.AlignCenter)
            switch_widget.setLayout(switch_layout)
            self.table.setCellWidget(i, 1, switch_widget)

            # Action button
            delete_button = PushButton("删除用户")
            delete_button.clicked.connect(partial(self.delete_user, user.username))
            edit_button = PushButton("修改密码")
            edit_button.clicked.connect(partial(self.edit_user, user.username))
            action_button_widget = QWidget()
            action_button_layout = QHBoxLayout()
            action_button_layout.addWidget(delete_button)
            action_button_layout.addWidget(edit_button)
            action_button_layout.setAlignment(Qt.AlignCenter)
            action_button_layout.setContentsMargins(0, 0, 0, 0)
            action_button_widget.setLayout(action_button_layout)
            self.table.setCellWidget(i, 2, action_button_widget)

        self.table.setFixedSize(700, 500)
        self.table.resizeColumnsToContents()

    def set_admin(self, username, state):
        try:
            w = CustomMessageBox('修改权限', '请输入超级管理员密码:', self.parent().parent().parent().parent().parent())
            if w.exec():
                if not self.manager.check_password('superadmin', w.urlLineEdit.text()):
                    content = "超级管理员密码错误"
                    InfoBar.error(
                        title='Error!',
                        content=content,
                        orient=Qt.Horizontal,
                        isClosable=True,
                        position=InfoBarPosition.BOTTOM_RIGHT,
                        # position='Custom',   # NOTE: use custom info bar manager
                        duration=2000,
                        parent=self.parent().parent().parent().parent()
                    )
                    # Reset the switch
                    state.blockSignals(True)
                    state.setChecked(not state.isChecked())
                    state.blockSignals(False)
                else:
                    self.manager.set_admin(username, state.isChecked())
                    content = f"用户 {username} 现在是 {'管理员' if state.isChecked() else '普通用户'}。"
                    InfoBar.success(
                        title='成功！',
                        content=content,
                        orient=Qt.Horizontal,
                        isClosable=True,
                        position=InfoBarPosition.BOTTOM_RIGHT,
                        # position='Custom',   # NOTE: use custom info bar manager
                        duration=2000,
                        parent=self.parent().parent().parent().parent()
                    )
            else:
                # Reset the switch
                state.blockSignals(True)
                state.setChecked(not state.isChecked())
                state.blockSignals(False)
        except Exception as e:
            content = f"修改用户 {username} 的权限时发生错误：{str(e)}"
            InfoBar.error(
                title='Error!',
                content=content,
                orient=Qt.Horizontal,
                isClosable=True,
                position=InfoBarPosition.BOTTOM_RIGHT,
                # position='Custom',   # NOTE: use custom info bar manager
                duration=2000,
                parent=self.parent().parent().parent().parent()
            )

    def delete_user(self, username):
        try:
            self.manager.delete_user(username)
            content = f"用户 {username} 已被删除。"
            InfoBar.success(
                title='成功！',
                content=content,
                orient=Qt.Horizontal,
                isClosable=True,
                position=InfoBarPosition.BOTTOM_RIGHT,
                # position='Custom',   # NOTE: use custom info bar manager
                duration=2000,
                parent=self.parent().parent().parent().parent()
            )
            self.fill_table()
        except Exception as e:
            content = f"删除用户 {username} 时发生错误：{str(e)}"
            InfoBar.error(
                title='Error!',
                content=content,
                orient=Qt.Horizontal,
                isClosable=True,
                position=InfoBarPosition.BOTTOM_RIGHT,
                # position='Custom',   # NOTE: use custom info bar manager
                duration=2000,
                parent=self.parent().parent().parent().parent()
            )

    def edit_user(self, username):
        try:
            if self.manager.check_admin(username):
                w = CustomMessageBox('修改密码', '请输入超级管理员密码:',
                                     self.parent().parent().parent().parent().parent())
                if w.exec():
                    if self.manager.check_password('superadmin', w.urlLineEdit.text()):
                        # 弹出输入新密码对话框
                        w = CustomMessageBox('修改密码', '请输入新密码:',
                                             self.parent().parent().parent().parent().parent())
                        if w.exec():
                            self.manager.change_password(username, w.urlLineEdit.text())
                            content = f"用户 {username} 的密码已被修改。"
                            InfoBar.success(
                                title='成功！',
                                content=content,
                                orient=Qt.Horizontal,
                                isClosable=True,
                                position=InfoBarPosition.BOTTOM_RIGHT,
                                # position='Custom',   # NOTE: use custom info bar manager
                                duration=2000,
                                parent=self.parent().parent().parent().parent()
                            )
                    else:
                        content = "超级管理员密码错误"
                        InfoBar.error(
                            title='Error!',
                            content=content,
                            orient=Qt.Horizontal,
                            isClosable=True,
                            position=InfoBarPosition.BOTTOM_RIGHT,
                            # position='Custom',   # NOTE: use custom info bar manager
                            duration=2000,
                            parent=self.parent().parent().parent().parent()
                        )
            else:
                w = CustomMessageBox('修改密码', '请输入新密码:', self.parent().parent().parent().parent().parent())
                if w.exec():
                    self.manager.change_password(username, w.urlLineEdit.text())
                    content = f"用户 {username} 的密码已被修改。"
                    InfoBar.success(
                        title='成功！',
                        content=content,
                        orient=Qt.Horizontal,
                        isClosable=True,
                        position=InfoBarPosition.BOTTOM_RIGHT,
                        # position='Custom',   # NOTE: use custom info bar manager
                        duration=2000,
                        parent=self.parent().parent().parent().parent()
                    )
        except Exception as e:
            content = f"修改用户 {username} 的密码时发生错误：{str(e)}"
            InfoBar.error(
                title='Error!',
                content=content,
                orient=Qt.Horizontal,
                isClosable=True,
                position=InfoBarPosition.BOTTOM_RIGHT,
                # position='Custom',   # NOTE: use custom info bar manager
                duration=2000,
                parent=self.parent().parent().parent().parent()
            )

    def add_user(self):
        try:
            w = CustomMessageBox('添加用户', '请输入用户名:', self.parent().parent().parent().parent().parent())
            if w.exec():
                username = w.urlLineEdit.text()
                w = CustomMessageBox('添加用户', '请输入密码:', self.parent().parent().parent().parent().parent())
                if w.exec():
                    password = w.urlLineEdit.text()
                    w = CustomMessageBox('添加用户', '请再次输入密码:',
                                         self.parent().parent().parent().parent().parent())
                    if w.exec():
                        if password != w.urlLineEdit.text():
                            raise Exception('两次密码不一致')
                        is_admin = False
                        if Globals.user is not None and self.manager.check_admin(Globals.user.username):
                            w = CustomMessageBox('添加为管理员', '输入超级管理员密码(非管理员输入“否”)',
                                                 self.parent().parent().parent().parent().parent())
                            if w.exec():
                                if self.manager.check_password('superadmin', w.urlLineEdit.text()):
                                    is_admin = True
                        self.manager.add_user(username, password, is_admin)
                        content = f"用户 {username} 已被添加。"
                        InfoBar.success(
                            title='成功！',
                            content=content,
                            orient=Qt.Horizontal,
                            isClosable=True,
                            position=InfoBarPosition.BOTTOM_RIGHT,
                            # position='Custom',   # NOTE: use custom info bar manager
                            duration=2000,
                            parent=self.parent().parent().parent().parent()
                        )
                        self.fill_table()
        except Exception as e:
            content = f"添加用户时发生错误：{str(e)}"
            InfoBar.error(
                title='Error!',
                content=content,
                orient=Qt.Horizontal,
                isClosable=True,
                position=InfoBarPosition.BOTTOM_RIGHT,
                # position='Custom',   # NOTE: use custom info bar manager
                duration=2000,
                parent=self.parent().parent().parent().parent()
            )
