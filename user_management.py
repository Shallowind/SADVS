import sqlite3
import sys

import qdarkstyle
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QPainter, QBitmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox
from PyQt5.uic import loadUi


def dict_factory(cursor, row):
    """
    自定义字典工厂函数，将数据库查询结果转换为字典形式
    """
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d


class User_management(QMainWindow):
    def __init__(self):
        super(User_management, self).__init__()

        # 加载UI文件
        self.dragPos = None
        loadUi('user_management.ui', self)

        # 设置窗口标志，去掉窗口边框
        self.setWindowFlags(Qt.FramelessWindowHint)

        # 登录
        self.signin.clicked.connect(self.Login)
        self.signup.clicked.connect(self.Signup)
        # 退出
        # self.quit.clicked.connect(self.close)
        self.exit.clicked.connect(self.close)
        self.widget_2.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())

        # 设置窗口背景颜色为白色
        self.setStyleSheet("background-color: #19232d;")

        # 设置窗口大小
        # self.setGeometry(100, 100, 400, 200)
        self.conn = sqlite3.connect("Users.db")
        self.conn.row_factory = dict_factory
        try:
            cursor = self.conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='user'")
            result = cursor.fetchone()
            if result is not None:
                print("Table 'user' already exists.")
            else:
                # 创建表的代码
                self.conn.execute('''
                CREATE TABLE user(
                    user_level int,
                    user_name text,
                    password text
                )
                ''')
        except Exception as e:
            print("Error:", e)

    def mousePressEvent(self, event):
        # 实现窗口拖动
        if event.button() == Qt.LeftButton:
            self.dragPos = event.globalPos()
            event.accept()

    def Login(self):
        if self.username.text() and self.password.text():
            try:
                cursor = self.conn.cursor()
                cursor.execute("SELECT password FROM user WHERE user_name = ?", (self.username.text(),))
                result = cursor.fetchone()
                if result['password'] == self.password.text():
                    self.warning.setStyleSheet("color: green;")
                    self.warning.setText("登录成功")
                else:
                    self.warning.setStyleSheet("color: red;")
                    self.warning.setText("密码错误")
            except Exception as e:
                self.warning.setStyleSheet("color: red;")
                self.warning.setText("用户名或密码错误")
        else:
            self.warning.setStyleSheet("color: red;")
            self.warning.setText("用户名或密码不能为空")

    def Signup(self):
        if self.username.text() and self.password.text():
            try:
                # 创建游标对象
                cursor = self.conn.cursor()

                # 查询数据库中是否存在相同的 user_name
                cursor.execute("SELECT * FROM user WHERE user_name=?", (self.username.text(),))
                result = cursor.fetchone()

                if result:
                    # 用户名已存在，给出提示
                    self.warning.setStyleSheet("color: red;")
                    self.warning.setText("用户名已存在，请选择其他用户名")
                else:
                    # 用户名不存在，插入数据
                    cursor.execute("INSERT INTO user (user_level, user_name, password) VALUES(0, ?, ?)",
                                   (self.username.text(), self.password.text()))
                    self.conn.commit()
                    self.warning.setStyleSheet("color: green;")
                    self.warning.setText("注册成功")

                # cursor = self.conn.execute("SELECT * FROM user")
                # for row in cursor.fetchall():
                #     print(row)
            except Exception as e:
                self.warning.setStyleSheet("color: red;")
                self.warning.setText("用户名已存在")
        else:
            self.warning.setStyleSheet("color: red;")
            self.warning.setText("用户名或密码不能为空")

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


if __name__ == '__main__':
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
    app = QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    window = User_management()
    window.show()
    sys.exit(app.exec_())

