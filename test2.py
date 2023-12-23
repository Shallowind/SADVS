import os
import ctypes
import os
import sys
import threading
# import ffmpeg
from datetime import datetime
from time import sleep
from tkinter import Tk, simpledialog
from json import loads
import cv2
import numpy as np
import pygame.camera
import qdarkstyle
from PyQt5 import uic
from PyQt5.QtCore import QUrl, Qt, QTimer, QFileInfo, pyqtSignal
from PyQt5.QtGui import QIcon, QImage, QPixmap, QColor
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtWidgets import QFileDialog, QApplication, QSplitter, QTreeWidgetItem, QListView, QTreeWidget, QMainWindow, \
    QMenu, QAction, QMessageBox, QSizePolicy
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from moviepy.video.io.VideoFileClip import VideoFileClip

import detect
import detect_yolov5
from labels_settings import LabelsSettings
from model_settings import ModelSettings
from user_management import User_management
from utils.myutil import Globals, ConsoleRedirector
from mainwindow_ui import Ui_MainWindow

ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("myappid")
from test import Video


class MainWindow(Ui_MainWindow, QMainWindow):
    signal = pyqtSignal()

    def __init__(self):
        # 加载designer设计的ui程序
        super().__init__()
        self.setupUi(self)
        self.resize(1000, 600)
        self.showMaximized()
        self.setWindowTitle("MMX")
        self.icon = QIcon()
        self.icon.addPixmap(QPixmap("./resources/UI/logo.ico"), QIcon.Normal, QIcon.Off)
        self.setWindowIcon(self.icon)

        self.path = ''

        self.tabWidget.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        self.signal.connect(self.cut_completed)
        # 播放器
        self.vdplayer_1 = Video()
        self.vdplayer = self.vdplayer_1
        self.vdplayer_1.setVideoOutput([self.player_show_1, self.player_widget_1], self.video_slider, self.video_time,
                                       self.cut_time)
        self.vdplayer_2 = Video()
        self.vdplayer_2.setVideoOutput([self.player_show_2, self.player_widget_2], self.video_slider_2, self.video_time,
                                       self.cut_time)
        self.vdplayer_3 = Video()
        self.vdplayer_3.setVideoOutput([self.player_show_3, self.player_widget_3], self.video_slider_3, self.video_time,
                                       self.cut_time)
        self.vdplayer_4 = Video()
        self.vdplayer_4.setVideoOutput([self.player_show_4, self.player_widget_4], self.video_slider_5, self.video_time,
                                       self.cut_time)
        self.change_video_widget_enabled = False
        self.widget_1.doubleClicked.connect(lambda: self.change_video_widget(1))
        self.widget_1.mousePressed.connect(lambda: self.change_select_video_widget(1))
        self.widget_2.doubleClicked.connect(lambda: self.change_video_widget(2))
        self.widget_2.mousePressed.connect(lambda: self.change_select_video_widget(2))
        self.widget_3.doubleClicked.connect(lambda: self.change_video_widget(3))
        self.widget_3.mousePressed.connect(lambda: self.change_select_video_widget(3))
        self.widget_4.doubleClicked.connect(lambda: self.change_video_widget(4))
        self.widget_4.mousePressed.connect(lambda: self.change_select_video_widget(4))

        # 选择文件夹
        self.video_select.triggered.connect(self.openVideoFolder)
        self.new_file.clicked.connect(self.openVideoFolder)
        # 设置默认视频目录
        self.video_path.triggered.connect(self.changePath)
        # 默认保存路径
        self.save_path.triggered.connect(self.changePath)
        # 退出
        self.quit.triggered.connect(self.Quit)
        # 模型查看
        self.model_view.triggered.connect(self.ModelView)
        # 标签集设置
        self.action_sets.triggered.connect(self.labelSetsSettings)
        # 使用信息说明
        self.Use_Information.triggered.connect(self.display_Use_Information)
        # 版本信息说明
        self.Version_Information.triggered.connect(self.display_Version_Information)
        # 加入工作区
        self.add_workspace.clicked.connect(self.addWorkspace)
        # 暂停
        self.play_pause.clicked.connect(self.playPause)
        # 倍速功能
        self.speed.currentIndexChanged.connect(self.speed_change)
        # 上一个/下一个视频
        self.prev.clicked.connect(self.prev_video)
        self.next.clicked.connect(self.next_video)
        # 工作区/原始列表
        self.listcon.clicked.connect(self.listcontrol)
        self.workcon.clicked.connect(self.workcontrol)
        self.terminalcon.clicked.connect(self.terminalcontrol)
        self.original.activity = True
        self.works.activity = True
        self.controlWidget.activity = True
        # 双击播放
        self.video_tree.setSortingEnabled(False)
        self.video_tree.itemDoubleClicked.connect(self.CameraVideo)
        self.work_list.itemDoubleClicked.connect(self.WorkListPreview)
        # 文件树展开
        self.video_tree.itemExpanded.connect(self.loadSubtree)
        # 右键菜单
        self.video_tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.video_tree.customContextMenuRequested.connect(self.showContextMenu)
        # 工作区右键菜单
        self.work_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.work_list.customContextMenuRequested.connect(self.worklist_show_context_menu)
        # 进度条
        # self.vdplayer.durationChanged.connect(self.getDuration)
        # self.vdplayer.positionChanged.connect(self.getPosition)
        # self.idplayer.durationChanged.connect(self.getDuration)
        # self.idplayer.positionChanged.connect(self.getPosition)
        # self.video_slider.sliderMoved.connect(self.updatePosition)
        # self.video_slider_2.sliderMoved.connect(self.updatePosition)
        self.cut_slider.sliderMoved.connect(self.echo)
        self.cut_slider.setVisible(False)
        self.cut_time.setVisible(False)
        self.label_2.setVisible(False)
        self.cut_time_all.setVisible(False)
        self.hipo = 0
        self.lopo = 0
        # 保存标签
        self.save_label.clicked.connect(self.saveLabel)
        # 动作列表
        self.frame_dict = {}
        # 搜索动作
        self.item_search_button.clicked.connect(self.search_action)
        # 获取摄像头id列表
        self.camera_id_list = None
        camera_thread = threading.Thread(target=self.initialize_camera)
        camera_thread.daemon = True  # 主界面关闭时自动退出此线程
        camera_thread.start()
        self.capture = None
        # 视频/摄像头
        self.v_d_comboBox.currentIndexChanged.connect(self.select_V_D)
        self.v_d_comboBox.setView(QListView())
        # 启动识别
        self.start_identify.clicked.connect(self.startIdentifyClicked)
        self.settings_window = None
        # 停止识别
        self.stop_identify.clicked.connect(self.stopIdentify)
        # 使用样式表来设置项的高度
        self.v_d_comboBox.setStyleSheet('QComboBox QAbstractItemView::item { height: 20px; }')
        self.v_d_comboBox.setMaxVisibleItems(50)
        # tab切换
        self.tabWidget.currentChanged.connect(self.tabChanged)
        # 定时器
        self.timer_cv = QTimer()
        self.timer_cv.timeout.connect(self.updateFrame)
        self.timer_cv.setInterval(30)  # 1000毫秒 = 1秒
        # 剪辑模式
        self.cut_mode.setEnabled(False)
        self.save_cut.setVisible(False)
        self.exit_mode.setVisible(False)
        self.cut_mode.clicked.connect(self.cutMode)
        self.exit_mode.clicked.connect(self.exitMode)
        self.save_cut.clicked.connect(self.saveCut)
        self.cut_path = None
        # 创建图形
        plt.rcParams['font.family'] = 'Microsoft YaHei'
        self.figure, self.ax = plt.subplots()
        self.ax.set_facecolor('#19232d')
        self.ax.set_xlabel('时间', color='white')
        self.ax.set_ylabel('数目', color='white')
        self.ax.tick_params(axis='both', colors='white')
        self.ax.spines['bottom'].set_color('white')
        self.ax.spines['top'].set_color('white')
        self.ax.spines['right'].set_color('white')
        self.ax.spines['left'].set_color('white')
        self.ax.set_title('每秒动作数目趋势', color='white')
        self.figure.set_facecolor('#19232d')
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        # 添加到布局中
        self.verticalLayout_6.addWidget(self.toolbar)
        self.verticalLayout_6.addWidget(self.canvas)
        # 创建图形
        self.figure2, self.ax2 = plt.subplots()
        labels = ('unkown',)
        sizes = [100]
        self.ax2.set_title('动作分布', color='white')
        self.ax2.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, textprops={'color': 'white'})
        self.figure2.set_facecolor('#19232d')
        self.canvas2 = FigureCanvas(self.figure2)
        self.toolbar2 = NavigationToolbar(self.canvas2, self)
        # 添加到布局中
        self.verticalLayout_7.addWidget(self.toolbar2)
        self.verticalLayout_7.addWidget(self.canvas2)
        # 启用多选
        self.video_tree.setSelectionMode(QTreeWidget.ExtendedSelection)
        self.video_tree.setStyleSheet("""
        QTreeWidget::branch:has-siblings:!adjoins-item{ \
            border-image:None 0;\
        }\
        QTreeWidget::branch:has-siblings:adjoins-item{\
            border-image:None 0;\
        }\
        QTreeWidget::branch:!has-children:!has-siblings:adjoins-item{\
            border-image:None 0;\
        }\
        QTreeWidget::branch:has-siblings:adjoins-item{\
            border-image:None 0;\
        }\
        QLineEdit{\
            padding: 0;\
            margin: 0;\
        }\
        """)
        self.work_list.setStyleSheet("""
        QLineEdit{\
            padding: 0;\
            margin: 0;\
        }\
        """)
        self.work_list.setSelectionMode(QTreeWidget.ExtendedSelection)
        self.terminal.setStyleSheet("background-color: 000000")
        # sys.stdout = ConsoleRedirector(self, self.terminal)
        # sys.stderr = ConsoleRedirector(self, self.terminal, QColor(255, 0, 0))
        print()

        self.videotree = []
        self.selected_folder = ""
        self.selected_path = ""
        self.selected_item = None
        self.labsettings_window = None

        # 分割器
        splitter_list = QSplitter(Qt.Vertical)
        splitter_list.addWidget(self.original)
        splitter_list.addWidget(self.works)
        self.verticalLayout_13.addWidget(splitter_list)
        self.verticalLayout_13.setStretch(0, 1)  # 第一个部件的伸缩因子为1
        self.verticalLayout_13.setStretch(1, 40)  # 第二个部件的伸缩因子为2
        self.verticalLayout_13.setStretch(2, 40)  # 第三个部件的伸缩因子为3
        splitter_list.setStyleSheet("""
            QSplitter {
                background-color: 19232d;
            }
            QSplitter::handle {
                background-color: 19232d;
            }
        """)

        splitter_tab = QSplitter(Qt.Horizontal)
        splitter_tab.addWidget(self.widget_list)
        splitter_tab.addWidget(self.tabWidget)
        splitter_tab.setStretchFactor(0, 8)
        splitter_tab.setStretchFactor(1, 10)
        splitter_tab.setStyleSheet("""
            QSplitter {
                background-color: 19232d;
            }
            QSplitter::handle {
                background-color: 19232d;
            }
        """)
        self.widget_list.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        self.centralwidget.setStyleSheet("""
            QSplitter {
                background-color: 19232d;
            }
            QSplitter::handle {
                background-color: 19232d;
            }
        """)

        self.horizontalLayout_14.addWidget(splitter_tab)

        splitter_video = QSplitter(Qt.Horizontal)
        splitter_video.addWidget(self.video_widget)
        splitter_video.addWidget(self.video_label_widget)
        splitter_video.setStretchFactor(0, 5)
        splitter_video.setStretchFactor(1, 2)
        self.horizontalLayout_5.addWidget(splitter_video)
        splitter_video.setStyleSheet("""
            QSplitter {
                background-color: 19232d;
            }
            QSplitter::handle {
                background-color: 19232d;
            }
        """)

        splitter_video = QSplitter(Qt.Horizontal)
        splitter_video.addWidget(self.video_widget_2)
        splitter_video.addWidget(self.video_label_widget_2)
        splitter_video.setStretchFactor(0, 5)
        splitter_video.setStretchFactor(1, 1)
        splitter_video.setStyleSheet("""
            QSplitter {
                background-color: 19232d;
            }
            QSplitter::handle {
                background-color: 19232d;
            }
        """)
        self.horizontalLayout_7.addWidget(splitter_video)

        splitter_control = QSplitter(Qt.Vertical)
        splitter_control.addWidget(self.useSpace)
        splitter_control.addWidget(self.controlWidget)
        splitter_control.setStretchFactor(0, 10)
        splitter_control.setStretchFactor(1, 2)
        splitter_control.setStyleSheet("""
            QSplitter {
                background-color: 19232d;
            }
            QSplitter::handle {
                background-color: 19232d;
            }
        """)
        self.verticalLayout_10.addWidget(splitter_control)
        # 图标
        self.file_icon = QIcon("resources/file_ico.png")
        self.folder_icon = QIcon("resources/folder_ico.ico")
        self.image_icon = QIcon("resources/img_ico.ico")
        self.text_icon = QIcon("resources/text_ico.png")
        self.video_icon = QIcon("resources/video_ico.png")
        self.camera_icon = QIcon("resources/cam_ico.png")
        self.play_ico = QIcon("resources/play_ico.png")
        self.pause_ico = QIcon("resources/pause_ico.png")
        self.menubar.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        self.statusbar.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        self.centralwidget.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())

        # 获得默认地址
        try:
            with open('Default settings.txt', 'r', encoding='utf-8') as f:
                content = f.read()
            seted = loads(content)
            try:
                video_path = seted['video_path']
                print(video_path)
                if os.path.isdir(video_path):
                    self.openVideoFolder(video_path)
            except KeyError:
                print("The key 'video_path' does not exist in the dictionary.")
        except FileNotFoundError:
            print("文件不存在")

        path = os.path.join(os.getcwd(), 'labels')
        with open(os.path.join(path, "yolo_slowfast", "字典.txt"), 'r', encoding='utf-8') as f:
            content = f.read()
        Globals.yolo_slowfast_dict = loads(content)

        with open(os.path.join(path, "yolov5", "字典.txt"), 'r', encoding='utf-8') as f:
            content = f.read()
        Globals.yolov5_dict = loads(content)

    def change_select_video_widget(self, num):
        if num == 1:
            self.vdplayer = self.vdplayer_1
        elif num == 2:
            self.vdplayer = self.vdplayer_2
        elif num == 3:
            self.vdplayer = self.vdplayer_3
        elif num == 4:
            self.vdplayer = self.vdplayer_4

    def change_video_widget(self, num):
        if self.change_video_widget_enabled:
            self.change_video_widget_enabled = False
            self.widget_1.setVisible(True)
            self.widget_2.setVisible(True)
            self.widget_3.setVisible(True)
            self.widget_4.setVisible(True)
        else:
            self.change_video_widget_enabled = True
            if num == 1:
                self.widget_2.setVisible(False)
                self.widget_4.setVisible(False)
                self.widget_3.setVisible(False)
            elif num == 2:
                self.widget_1.setVisible(False)
                self.widget_3.setVisible(False)
                self.widget_4.setVisible(False)
            elif num == 3:
                self.widget_1.setVisible(False)
                self.widget_2.setVisible(False)
                self.widget_4.setVisible(False)
            elif num == 4:
                self.widget_1.setVisible(False)
                self.widget_2.setVisible(False)
                self.widget_3.setVisible(False)

    # 保存行为识别报告
    def stopIdentify(self):
        QMessageBox.information(self, "提示", "识别报告已保存到文件夹\nD:/VScode/motion-monitor-x/result")
        Globals.detection_run = False
        self.start_identify.setEnabled(True)
        self.stop_identify.setEnabled(False)

    # 显示/关闭 终端
    def terminalcontrol(self):
        if self.controlWidget.activity:
            self.controlWidget.activity = False
        else:
            self.controlWidget.activity = True
        self.refresh_widgetlist()

    # 显示/关闭 视频列表
    def listcontrol(self):
        if self.original.activity:
            self.original.activity = False
        else:
            self.original.activity = True
        self.refresh_widgetlist()

    # 显示/关闭 工作区
    def workcontrol(self):
        if self.works.activity:
            self.works.activity = False
        else:
            self.works.activity = True
        self.refresh_widgetlist()

    # 刷新视频列表/工作区/终端
    def refresh_widgetlist(self):
        if self.original.activity:
            self.original.setVisible(True)
            self.listcon.setStyleSheet("""
                QPushButton {
                    background-color: #54687a;
                    border: 1px solid #259ae9;
                }

                QPushButton:hover {
                    background-color: #54687a;
                }

                QPushButton:pressed {
                    background-color: #455364;
                }
            """)
        else:
            self.original.setVisible(False)
            self.listcon.setStyleSheet("""
                    QPushButton {
                        background-color: #2A3A4C;
                        border: 1px solid #666666;
                    }

                    QPushButton:hover {
                        background-color: #2A3A4C;
                    }

                    QPushButton:pressed {
                        background-color: #455364;
                    }
                """)
        if self.works.activity:
            self.works.setVisible(True)
            self.workcon.setStyleSheet("""
                QPushButton {
                    background-color: #54687a;
                    border: 1px solid #259ae9;
                }

                QPushButton:hover {
                    background-color: #54687a;
                }

                QPushButton:pressed {
                    background-color: #455364;
                }
            """)
        else:
            self.works.setVisible(False)
            self.workcon.setStyleSheet("""
                    QPushButton {
                        background-color: #2A3A4C;
                        border: 1px solid #666666;
                    }

                    QPushButton:hover {
                        background-color: #2A3A4C;
                    }
                """)
        if self.controlWidget.activity:
            self.controlWidget.setVisible(True)
            self.terminalcon.setStyleSheet("""
                QPushButton {
                    background-color: #54687a;
                    border: 1px solid #259ae9;
                }

                QPushButton:hover {
                    background-color: #54687a;
                }

                QPushButton:pressed {
                    background-color: #455364;
                }
            """)
        else:
            self.controlWidget.setVisible(False)
            self.terminalcon.setStyleSheet("""
                    QPushButton {
                        background-color: #2A3A4C;
                        border: 1px solid #666666;
                    }

                    QPushButton:hover {
                        background-color: #2A3A4C;
                    }
                """)

    # 显示软件使用信息
    def display_Use_Information(self):
        message_box = QMessageBox()
        # 设置对话框的标题
        message_box.setWindowTitle("使用信息")
        # 设置对话框的文本内容
        message = "yolo_slowfast是一种快速的目标检测模型，可以用于实时检测图像或视频中的人体动作。\n"
        message_box.setText(message)
        # 添加 OK 按钮
        message_box.addButton(QMessageBox.Ok)
        # 显示对话框
        message_box.exec_()
        return

    # 显示软件版本信息
    def display_Version_Information(self):
        message_box = QMessageBox()
        # 设置对话框的标题
        message_box.setWindowTitle("版本信息")
        # 设置对话框的文本内容
        message = "yolo_slowfast是一种快速的目标检测模型，可以用于实时检测图像或视频中的人体动作。\n"
        message_box.setText(message)
        # 添加 OK 按钮
        message_box.addButton(QMessageBox.Ok)
        # 显示对话框
        message_box.exec_()
        return

    # 显示模型简介
    def ModelView(self):
        message_box = QMessageBox()
        # 设置对话框的标题
        message_box.setWindowTitle("模型查看")
        # 设置对话框的文本内容
        message = "yolo_slowfast是一种快速的目标检测模型，可以用于实时检测图像或视频中的人体动作。\n"
        message += "yolo5则是一种目标检测模型，可以用于检测图像或视频中的物体。\n"
        message_box.setText(message)
        # 添加 OK 按钮
        message_box.addButton(QMessageBox.Ok)
        # 显示对话框
        message_box.exec_()
        return

    # 改变默认视频文件、保存路径
    def changePath(self):
        try:
            with open('Default settings.txt', 'r', encoding='utf-8') as f:
                content = f.read()
            seted = loads(content)
            video_path = seted['video_path']
            save_path = seted['save_path']

        except FileNotFoundError:
            print("文件不存在")

    # 退出功能
    def Quit(self):
        del pyqt5.ui

    # 启动自定义标签设置窗口
    def labelSetsSettings(self):
        self.labsettings_window = LabelsSettings(self)
        # self.settings_window.setWindowZOrder(Qt.TopMost)
        self.labsettings_window.show()

    def saveCut(self):
        cut_thread = threading.Thread(target=self.cut_thread)
        cut_thread.daemon = True  # 主界面关闭时自动退出此线程
        cut_thread.start()

    # 剪辑完成后的视频加入工作区
    def cut_completed(self):
        print("剪辑已完成！")
        result = QMessageBox.warning(self, "已完成", "剪辑已完成！\n是否加入工作区？", QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.Yes)
        if result == QMessageBox.Yes:
            item = QTreeWidgetItem(self.work_list)
            item.isCamera = False
            item.path = self.cut_path
            file_name = QFileInfo(self.cut_path).baseName()  # 获取文件名（不包括后缀）
            file_extension = QFileInfo(self.cut_path).completeSuffix()  # 获取后缀名
            # 根据文件后缀选择不同图标
            if file_extension == "txt":
                item.setIcon(0, self.text_icon)
            elif file_extension == "jpg" or file_extension == "png":
                item.setIcon(0, self.image_icon)
            elif file_extension == "avi" or file_extension == "mp4":
                item.setIcon(0, self.video_icon)
            else:
                item.setIcon(0, self.file_icon)
            item.setText(0, file_name)
            self.work_list.addTopLevelItem(item)

    # 剪辑视频线程
    def cut_thread(self):
        # folder_path = QFileDialog.getExistingDirectory()
        default_name = QFileInfo(self.selected_path).baseName()  # 获取文件名（不包括后缀）
        default_extension = QFileInfo(self.selected_path).completeSuffix()  # 获取后缀名

        target, fileType = QFileDialog.getSaveFileName(self, "保存文件", default_name, f"*.{default_extension}")
        source = self.selected_path.strip()
        target = target.strip()
        # 获得视频时长
        all_seconds = self.vdplayer.player.get(cv2.CAP_PROP_FRAME_COUNT) / self.vdplayer.player.get(
            cv2.CAP_PROP_FPS)

        start_time_sec = self.lopo / 100 * all_seconds  # 获取开始剪切时间（毫秒）
        stop_time_sec = self.hipo / 100 * all_seconds  # 获取剪切的结束时间（毫秒）

        print(f"开始剪切时间：{start_time_sec}")
        print(f"结束剪切时间：{stop_time_sec}")
        try:
            print("剪辑进行中，请耐心等待...")
            video = VideoFileClip(source)  # 视频文件加载
            video = video.subclip(start_time_sec, stop_time_sec)  # 执行剪切操作，参数为秒
            video.to_videofile(target, remove_temp=True)  # 输出文件
            self.cut_path = target
            self.signal.emit()
        except Exception as e:
            print(f"出现错误： {e}")

    # 退出剪辑模式
    def exitMode(self):
        self.cut_slider.setVisible(False)
        self.cut_time.setVisible(False)
        self.label_2.setVisible(False)
        self.cut_time_all.setVisible(False)
        self.cut_mode.setVisible(True)
        self.save_cut.setVisible(False)
        self.exit_mode.setVisible(False)

    # 开始剪辑模式
    def cutMode(self):
        self.vdplayer.pause()
        self.play_pause.setIcon(self.play_ico)
        self.cut_slider.setVisible(True)
        self.cut_time.setVisible(True)
        self.label_2.setVisible(True)
        self.cut_time_all.setVisible(True)
        self.cut_mode.setVisible(False)
        self.save_cut.setVisible(True)
        self.exit_mode.setVisible(True)

    # 获得视频文件路径列表
    def get_video_files(self):
        path = os.path.dirname(self.selected_path)
        # 从当前文件夹中获取所有视频文件
        try:
            video_files = [f for f in os.listdir(path) if f.lower().endswith(('.mp4', '.avi', '.mov', '.jpg', '.png'))]
        except FileNotFoundError:
            return None
        # 构建完整的视频文件路径列表
        video_paths = [os.path.join(path, video_file) for video_file in video_files]
        print(video_paths)
        return video_paths

    # 获取文件夹上一个视频
    def prev_video(self):
        video_paths = self.get_video_files()
        if video_paths is None:
            return
        index = video_paths.index(self.selected_path)
        index -= 1
        # 如果当前索引小于0，循环到视频列表的末尾
        if index < 0:
            index = len(video_paths) - 1
        # 停止当前视频的播放
        self.vdplayer.pause()
        # 播放上一个视频
        self.playSelectedVideo(None, None, video_paths[index])

    # 获取文件夹下一个视频
    def next_video(self):
        video_paths = self.get_video_files()
        if video_paths is None:
            return
        index = video_paths.index(self.selected_path)
        index += 1
        # 如果当前索引小于0，循环到视频列表的末尾
        if index == len(video_paths):
            index = 0
        # 停止当前视频的播放
        self.vdplayer.pause()
        # 播放上一个视频
        self.playSelectedVideo(None, None, video_paths[index])

    # 倍速数值改变触发
    def speed_change(self):
        if self.tabWidget.currentIndex() == 0:
            speed = self.speed.currentText()
            speed = float(speed.split("x")[0])
            self.vdplayer.speed_change(speed)

    def echo(self, low_value, high_value):
        # print(low_value, high_value)
        if self.tabWidget.currentIndex() == 0:
            if low_value != self.lopo:
                self.lopo = low_value
                self.vdplayer.setPosition(low_value)
                # self.vdplayer.play()
                # self.speed_play(self.vdplayer)
                # self.vdplayer.pause()
                self.play_pause.setIcon(self.play_ico)
            elif high_value != self.hipo:
                self.hipo = high_value
                self.vdplayer.setPosition(high_value)
                # self.vdplayer.play()
                # self.speed_play(self.vdplayer)
                # self.vdplayer.pause()
                self.play_pause.setIcon(self.play_ico)

    # 工作区右键菜单显示
    def worklist_show_context_menu(self, position):
        item = self.work_list.itemAt(position)
        if item is None:
            return
        menu = QMenu(self.work_list)
        # 添加新建文件夹的菜单项
        new_folder_action = QAction("重命名", self.work_list)
        new_folder_action.triggered.connect(lambda: self.worklist_rename(item))
        menu.addAction(new_folder_action)

        new_folder_action = QAction("删除", self.work_list)
        new_folder_action.triggered.connect(lambda: self.worklist_delete(item))
        menu.addAction(new_folder_action)
        # 显示菜单
        menu.exec_(self.work_list.mapToGlobal(position))

    # 工作区文件重命名
    def worklist_rename(self, item):
        # 获取当前项的文本
        old_name = self.getFullPath(item)
        if item:
            item.setFlags(item.flags() | Qt.ItemIsEditable)
            self.work_list.editItem(item)  # 启动编辑模式
        new_name = item.text(0)
        if new_name is None:
            return

        item.setText(0, new_name)

    # 工作区文件删除
    def worklist_delete(self, item):
        # 删除项
        if item.parent():
            # 如果有父节点，从父节点中移除
            parent = item.parent()
            index = parent.indexOfChild(item)
            parent.takeChild(index)
        else:
            # 如果没有父节点，从顶级项中移除
            index = self.work_list.indexOfTopLevelItem(item)
            self.work_list.takeTopLevelItem(index)

    # 显示文件夹和文件不同菜单
    def showContextMenu(self, position):
        item = self.video_tree.itemAt(position)
        if item is not None:
            item_path = self.getFullPath(item)
            # 根据文件类型选择不同菜单
            if os.path.isdir(item_path):
                context_menu = self.createFolderMenu(item)
            else:
                context_menu = self.createFileMenu(item)
            # 在给定位置显示上下文菜单
            context_menu.exec_(self.video_tree.mapToGlobal(position))

    # 创建文件右键菜单
    def createFileMenu(self, item):
        context_menu = QMenu(self.video_tree)
        context_menu.setStyleSheet("background-color: white")

        action = QAction("重命名", context_menu)
        context_menu.addAction(action)
        action.triggered.connect(lambda: self.rename_file(item))

        action = QAction("删除", context_menu)
        context_menu.addAction(action)
        action.triggered.connect(lambda: self.delete_file(item))

        action = QAction("加入工作区", context_menu)
        context_menu.addAction(action)
        action.triggered.connect(lambda: self.addWorkspace())

        return context_menu

    # 创建文件夹右键菜单
    def createFolderMenu(self, item):
        context_menu = QMenu(self.video_tree)
        context_menu.setStyleSheet("background-color: white")

        action = QAction("新建文件夹", context_menu)
        context_menu.addAction(action)
        action.triggered.connect(lambda: self.create_folder(item))

        action = QAction("新建", context_menu)
        context_menu.addAction(action)
        action.triggered.connect(lambda: self.create_file(item))

        action = QAction("删除", context_menu)
        context_menu.addAction(action)
        action.triggered.connect(lambda: self.delete_folder(item))

        action = QAction("重命名", context_menu)
        context_menu.addAction(action)
        action.triggered.connect(lambda: self.rename_file(item))

        return context_menu

    # 文件重命名
    def rename_file(self, item):
        old_name = self.getFullPath(item)
        current_folder_path = os.path.dirname(old_name)  # 获取选中视频的当前文件夹路径
        if item:
            item.setFlags(item.flags() | Qt.ItemIsEditable)
            self.video_tree.editItem(item)  # 启动编辑模式
        new_name = item.text(0)
        if new_name is None:
            return

        # 在编辑模式退出后调用重命名
        self.video_tree.itemChanged.connect(lambda item: self.on_item_changed(item, old_name, current_folder_path))
        item.setFlags(item.flags() & ~Qt.ItemIsEditable)

    def on_item_changed(self, item, old_name, current_folder_path):
        new_name = item.text(0)
        if new_name is None:
            return
        combined_path = os.path.join(current_folder_path, os.path.basename(new_name))  # 将当前文件夹路径和视频文件名结合成新的路径
        try:
            os.rename(old_name, combined_path)
            print(f"文件已成功重命名为： {new_name}")
        except FileNotFoundError:
            print("找不到指定的文件。请确认文件路径和名称是否正确。")
        except Exception as e:
            print(f"重命名文件时出现错误： {e}")

    # 文件删除
    def delete_file(self, item):
        file_path = self.getFullPath(item)
        print(file_path)
        index = item.parent().indexOfChild(item)
        if index != -1:
            item.parent().takeChild(index)
        del item
        # self.video_tree.remove(move_item)
        os.remove(file_path)

    # 新建文件
    def create_file(self, item):
        filename = "new_file.txt"
        current_folder_path = self.getFullPath(item)
        if filename == None:
            return
        combined_path = os.path.join(current_folder_path,
                                     os.path.basename(filename))  # 将当前文件夹路径和视频文件名结合成新的路径
        print(combined_path)
        # 在文件树中生成文件节点
        file_item_tree = QTreeWidgetItem([filename])
        file_item_tree.setData(0, Qt.UserRole, True)
        item.addChild(file_item_tree)

        # 在文件中生成文件节点
        with open(combined_path, 'w') as file:
            # 写入文件内容
            file.write('This is a file.')

        print("File created:", filename)

    # 新建文件夹
    def create_folder(self, item):
        folder_name = 'wenjianjia'
        current_folder_path = self.getFullPath(item)
        new_folder_path = os.path.join(current_folder_path, folder_name)  # 将当前文件夹路径和新文件夹名称结合成新的路径

        if not os.path.exists(new_folder_path):
            os.makedirs(new_folder_path)

            # 在文件树中生成文件夹节点
            folder_item_tree = QTreeWidgetItem([folder_name])
            folder_item_tree.setData(0, Qt.UserRole, False)  # 标记为文件夹节点
            item.addChild(folder_item_tree)
        else:
            print("Folder already exists")

    # 加入工作区
    def addWorkspace(self):
        selected_items = self.video_tree.selectedItems()
        for item in selected_items:
            # 复制选中的项到目标 TreeWidget
            if item.childCount() <= 0:
                cloned_item = item.clone()
                cloned_item.isCamera = item.isCamera
                if not item.isCamera:
                    cloned_item.path = self.getFullPath(item)
                else:
                    cloned_item.device = int(item.text(0))
                self.work_list.addTopLevelItem(cloned_item)

    # 启动识别按钮点击
    def startIdentifyClicked(self):
        if self.settings_window is None:
            Globals.settings['saved'] = False
            self.settings_window = ModelSettings(self)
            self.settings_window.show()
            self.setEnabled(False)  # 暂停主窗口活动

    # 识别完成回复函数
    def startIdentifyThread(self):
        if Globals.settings['saved']:
            self.timer_cv.stop()
            Globals.camera_running = True
            identify_thread = threading.Thread(target=self.startIdentify)
            identify_thread.daemon = True  # 主界面关闭时自动退出此线程
            identify_thread.start()

    # 启动模型开始识别
    def startIdentify(self):
        # 开始识别
        Globals.detection_run = True
        self.start_identify.setEnabled(False)
        self.stop_identify.setEnabled(True)

        if self.selected_item.isCamera:
            if Globals.settings['model_select'] == 'yolov5':
                # 使用yolov5模型进行相机检测
                detect_yolov5.run(source=self.selected_item.device, weights=Globals.settings['pt_path'],
                                  show_label=self.camera_2, project=Globals.settings['save_path'],
                                  save_img=False, use_camera=True, show_window=self, classes=Globals.select_labels,
                                  max_det=int(Globals.settings['max_det']), conf_thres=Globals.settings['conf'],
                                  iou_thres=Globals.settings['iou'], line_thickness=Globals.settings['line_thickness'])
            elif Globals.settings['model_select'] == 'yolo_slowfast':
                # 使用yolo_slowfast模型进行相机检测
                detect.run(source=self.selected_item.device, weights=Globals.settings['pt_path'],
                           show_label=self.camera_2, project=Globals.settings['save_path'],
                           save_img=False, use_camera=True, show_window=self, select_labels=Globals.select_labels,
                           max_det=int(Globals.settings['max_det']), conf_thres=Globals.settings['conf'],
                           iou_thres=Globals.settings['iou'], line_thickness=Globals.settings['line_thickness'])
        else:
            if Globals.settings['model_select'] == 'yolov5':
                # 使用yolov5模型进行图像检测
                detect_yolov5.run(source=self.selected_item.path, weights=Globals.settings['pt_path'],
                                  show_label=self.camera_2, project=Globals.settings['save_path'],
                                  save_img=True, show_window=self, classes=Globals.select_labels,
                                  max_det=int(Globals.settings['max_det']), conf_thres=Globals.settings['conf'],
                                  iou_thres=Globals.settings['iou'], line_thickness=Globals.settings['line_thickness'])
            elif Globals.settings['model_select'] == 'yolo_slowfast':
                # 使用yolo_slowfast模型进行图像检测
                detect.run(source=self.selected_item.path, weights=Globals.settings['pt_path'],
                           show_label=self.camera_2, project=Globals.settings['save_path'],
                           save_img=True, show_window=self, select_labels=Globals.select_labels,
                           max_det=int(Globals.settings['max_det']), conf_thres=Globals.settings['conf'],
                           iou_thres=Globals.settings['iou'], line_thickness=Globals.settings['line_thickness'])
        # detect.run(source=self.selected_path, weights=model_path, show_label=self.camera_2,
        # save_img=True, show_labellist=self.action_list)

        # 视频/设备切换时触发

    def select_V_D(self):
        selected_option = self.v_d_comboBox.currentText()
        if selected_option == '视频列表':
            # 释放资源
            if self.capture is not None:
                self.capture.release()
            self.timer_cv.stop()
            self.video_tree.clear()

            # 添加视频列表
            if self.selected_folder != "":
                self.addFolderToTree(self.video_tree, self.selected_folder)
        elif selected_option == '设备列表':
            self.video_tree.clear()

            # 添加设备列表
            if self.camera_id_list is not None:
                for camera_id in self.camera_id_list:
                    item = QTreeWidgetItem(self.video_tree)
                    item.isCamera = True
                    item.setText(0, str(camera_id))
                    item.setIcon(0, self.camera_icon)

        # 初始化检测摄像头线程

    def initialize_camera(self):
        # 初始化pygame摄像头模块
        pygame.camera.init()
        # 获取摄像头列表
        self.camera_id_list = pygame.camera.list_cameras()
        print("检测到设备：" + str(self.camera_id_list))
        selected_option = self.v_d_comboBox.currentText()
        if selected_option == '设备列表':
            # 清空视频树的显示内容
            self.video_tree.clear()
            if self.camera_id_list is not None:
                # 遍历摄像头列表
                for camera_id in self.camera_id_list:
                    # 创建新的树项
                    item = QTreeWidgetItem(self.video_tree)
                    # 设置该项为摄像头标志
                    item.isCamera = True
                    # 设置该项的文本为摄像头，ID图标为摄像头图标
                    item.setText(0, str(camera_id))
                    item.setIcon(0, self.camera_icon)

    # 文件树展开时调用
    def loadSubtree(self, item):
        # 调整视频树的第一列大小为自适应
        self.video_tree.resizeColumnToContents(0)
        selected_video_path = self.getFullPath(item)
        # 为选中的文件夹项设置自定义角色数据为True
        item.setData(0, Qt.UserRole, True)
        self._addFilesToTree(item, selected_video_path, 0)

    # 打开视频文件夹
    def openVideoFolder(self, path=""):
        # 选择文件夹
        print(path)
        if os.path.isdir(path):
            folder_path = path
        else:
            folder_path = QFileDialog.getExistingDirectory()

        if folder_path:
            self.selected_folder = folder_path
            self.video_tree.clear()
            self.addFolderToTree(self.video_tree, folder_path)
        self.path = folder_path

    # 文件数增加结点
    def addFolderToTree(self, tree_widget, folder_path):
        # 创建根节点
        root_item = QTreeWidgetItem(tree_widget)
        root_item.setData(0, Qt.UserRole, False)
        root_item.setText(0, os.path.basename(folder_path))
        icon = QIcon("resources/folder_ico.ico")
        root_item.setIcon(0, icon)
        self._addFilesToTree(root_item, folder_path, 0)

    # 递归文件夹和文件，加入到文件树里
    def _addFilesToTree(self, parent_item, folder_path, deep):
        for root, dirs, files in os.walk(folder_path):
            # 计算遍历深度
            depth = root.count(os.sep) - folder_path.count(os.sep)
            if depth == 0 and deep <= 1:
                child_items = parent_item.takeChildren()
                if not parent_item.data(0, Qt.UserRole):
                    for child_item in child_items:
                        del child_item
                # 只遍历一层，否则会把所有子文件放到同一栏
                for dir in dirs:
                    # 文件夹
                    dir_path = os.path.join(root, dir)
                    child_item = QTreeWidgetItem(parent_item)
                    child_item.isCamera = False
                    child_item.setData(0, Qt.UserRole, False)
                    child_item.setText(0, dir)

                    # 获取文件夹图标（可选）
                    child_item.setIcon(0, self.folder_icon)

                    self._addFilesToTree(child_item, dir_path, deep + 1)
                for file in files:
                    # 文件
                    item = QTreeWidgetItem(parent_item)
                    item.setData(0, Qt.UserRole, False)
                    item.isCamera = False
                    item.setText(0, file)

                    # 获取文件后缀
                    file_extension = os.path.splitext(file)[1].lower()

                    # 根据文件后缀选择不同图标
                    if file_extension == ".txt":
                        item.setIcon(0, self.text_icon)
                    elif file_extension == ".jpg" or file_extension == ".png":
                        item.setIcon(0, self.image_icon)
                    elif file_extension == ".avi" or file_extension == ".mp4":
                        item.setIcon(0, self.video_icon)
                    else:
                        item.setIcon(0, self.file_icon)
            else:
                # 如果当前深度大于0，停止继续遍历更深的目录
                del dirs[:]

    def load_frame_dict(self, frame_path=None):
        # 如果frame_path为空，则返回预设字典
        if not frame_path:
            return {
                2: [("talk", 0.9), ("stand", 0.8)],
                4: [("talk", 0.7), ("smoke", 0.9)]
            }
        else:
            frame_dict = {}
            # 遍历frame_path的键值对，将键转换为浮点数后添加到frame_dict中
            for k in frame_path:
                frame_dict[float(k)] = frame_path[k]
            return frame_dict

    # 视频流数据判断
    def CameraVideo(self, item):
        selected_option = self.v_d_comboBox.currentText()
        if selected_option == '视频列表':
            self.playSelectedVideo(item, False)
        elif selected_option == '设备列表':
            self.capture = cv2.VideoCapture(int(item.text(0)))
            self.timer_cv.start()

    # 预览工作区列表项
    def WorkListPreview(self, item):
        self.selected_item = item
        self.start_identify.setEnabled(True)
        self.start_identify.setEnabled(True)
        if item.isCamera:
            # 如果项目是摄像头
            self.capture = cv2.VideoCapture(int(item.text(0)))
            # 初始化摄像头捕捉对象
            self.timer_cv.start()
            # 启动定时器
        else:
            # 播放选定的视频，带声音
            self.playSelectedVideo(item, True)

    # 从OpenCV捕获摄像头获取一帧图像
    def updateFrame(self):
        flag, image = self.capture.read()
        show = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 将图像转换为QImage对象

        showImage = QImage(show.data, show.shape[1], show.shape[0], show.shape[1] * 3, QImage.Format_RGB888)

        label = self.player_show_1
        widget = self.player_2
        # 根据当前选中的选项卡索引调整标签
        if self.tabWidget.currentIndex() == 0:
            label = self.player_show_1
            widget = self.player_widget_1
        elif self.tabWidget.currentIndex() == 1:
            label = self.camera_2
            widget = self.player_2
        label_size = label.size()
        # 缩小尺寸并保持宽高比
        # label_size.setWidth(label_size.width() - 10)
        # label_size.setHeight(label_size.height() - 10)
        # scaled_image = showImage.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        # pixmap = QPixmap.fromImage(scaled_image)
        # label.setPixmap(pixmap)
        # label.setAlignment(Qt.AlignCenter)

        scale_factor = min(widget.width() / showImage.width(),
                           widget.height() / showImage.height())

        # 计算新的宽度和高度
        new_width = int(showImage.width() * scale_factor)
        new_height = int(showImage.height() * scale_factor)

        # 设置新的最大大小
        label.setMaximumSize(new_width, new_height)

        label.setPixmap(QPixmap(showImage))
        label.setScaledContents(True)

    def setMedia(self, video_path):
        self.vdplayer = cv2.VideoCapture(video_path)
        self.fps = self.vdplayer.get(cv2.CAP_PROP_FPS)

    # 播放选中视频
    def playSelectedVideo(self, item, isworklist, path=''):
        if self.capture is not None:
            self.capture.release()
        self.timer_cv.stop()
        # 重置标签
        self.video_label_edit.setText("")
        # 解禁设置视频标签
        self.video_label_edit.setEnabled(True)
        self.save_label.setEnabled(True)
        # 获取所选项的完整路径，包括文件夹结构
        if path:
            selected_video_path = path
        else:
            if isworklist:
                selected_video_path = item.path
            else:
                selected_video_path = self.getFullPath(item)

        self.selected_path = selected_video_path
        file_extension = os.path.splitext(selected_video_path)[1]

        # 播放或禁用播放
        player = self.vdplayer
        play_pause = self.play_pause
        if self.tabWidget.currentIndex() == 0:
            if os.path.exists(selected_video_path):
                # 获取文件的基本名称（不包含路径）
                file_name = os.path.basename(selected_video_path)
                # 获取文件的目录路径
                directory = os.path.dirname(selected_video_path)
                # 将文件名中的后缀名更改为.ini
                new_file_name = os.path.splitext(file_name)[0] + ".ini"
                # 构建新的文件路径
                new_path = os.path.join(directory, new_file_name)
                if os.path.exists(new_path):
                    # 以只读模式打开文件并读取内容
                    with open(new_path, "r") as file:
                        file_content = file.read()
                    self.video_label_edit.setText(file_content)
                # 加载动作列表
                # new_file_name = os.path.splitext(file_name)[0] + ".json"
                # new_path = os.path.join(directory, new_file_name)
                # if os.path.exists(new_path):
                #     self.frame_dict = self.load_frame_dict(new_path)
            else:
                print("err")
            player = self.vdplayer
            play_pause = self.play_pause
            if os.path.isfile(selected_video_path) and file_extension:
                player.setMedia(selected_video_path)
                self.getVideoinfo(selected_video_path)
                # player.setMedia(QMediaContent(QUrl.fromLocalFile(selected_video_path)))
                # 根据文件后缀选择不同处理方式
                if file_extension == ".avi" or file_extension == ".mp4":
                    play_pause.setEnabled(True)
                    self.cut_mode.setEnabled(True)
                    self.play_pause.setIcon(self.pause_ico)
                    # self.speed_play(player)
                    player.play()
                elif file_extension == ".jpg" or file_extension == ".png":
                    play_pause.setEnabled(False)
                    self.cut_mode.setEnabled(False)
                    # self.speed_play(player)
                    player.play()
                else:
                    self.cut_mode.setEnabled(False)
                    play_pause.setEnabled(False)

        elif self.tabWidget.currentIndex() == 1:
            video_path = selected_video_path
            self.idplayer.setMedia(video_path)
            if self.idplayer.player.isOpened():
                # 读取第一帧
                flag, image = self.idplayer.player.read()
                if flag:
                    show = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    showImage = QImage(show.data, show.shape[1], show.shape[0], show.shape[1] * 3, QImage.Format_RGB888)
                    # label_size = self.camera_2.size()
                    # label_size.setWidth(label_size.width() - 10)
                    # label_size.setHeight(label_size.height() - 10)
                    # scaled_image = showImage.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    # pixmap = QPixmap.fromImage(scaled_image)
                    # self.camera_2.setPixmap(pixmap)
                    # self.camera_2.setAlignment(Qt.AlignCenter)
                    # 计算缩放比例
                    scale_factor = min(self.player_2.width() / showImage.width(),
                                       self.player_2.height() / showImage.height())

                    # 计算新的宽度和高度
                    new_width = int(showImage.width() * scale_factor)
                    new_height = int(showImage.height() * scale_factor)

                    # 设置新的最大大小
                    self.camera_2.setMaximumSize(new_width, new_height)

                    self.camera_2.setPixmap(QPixmap(showImage))
                    self.camera_2.setScaledContents(True)
            else:
                self.idplayer.play()

    def getFullPath(self, item):
        # 从所选项递归构建完整路径
        path_components = [item.text(0)]
        if item.parent() is None:
            return self.selected_folder
        while item.parent() is not None and item.parent().parent() is not None:
            item = item.parent()
            path_components.insert(0, item.text(0))
        # 将路径组件连接起来
        full_path = os.path.join(self.selected_folder, *path_components)
        return full_path

    # 视频详细数据获取
    def getVideoinfo(self, selected_video_path):
        try:
            # 获取视频文件的基本信息
            video_info = os.stat(selected_video_path)
            file_size = video_info.st_size  # 文件大小（字节）

            modified_time = os.path.getmtime(selected_video_path)  # 文件修改日期时间
            formatted_date = datetime.fromtimestamp(modified_time).strftime("%Y-%m-%d %H:%M:%S")

            # 打开视频文件
            cap = cv2.VideoCapture(selected_video_path)

            # 获取视频的总帧数
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # 获取视频的帧率
            frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

            # 获取视频的时长（以秒为单位）
            duration_seconds = total_frames / frame_rate
            # 改成时分秒格式
            hours = int(duration_seconds / 3600)
            minutes = int((duration_seconds % 3600) / 60)
            seconds = int(duration_seconds % 60)

            # 获取视频的分辨率
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # 关闭视频文件
            cap.release()
            file_size = self.convert_bytes_to_readable(file_size)
            # 将信息设置到UI元素中
            info_text = f"总帧数: {total_frames}\n\n帧率: {frame_rate}\n\n时长: {hours}:{minutes}:{seconds}\n\n"
            self.video_time_all.setText(f"{minutes}:{seconds}")
            self.cut_time_all.setText(f"{minutes}:{seconds}")
            info_text += f"修改日期:{formatted_date} \n\n文件大小: {file_size}\n\n分辨率: {width}x{height}"
            self.video_info.setPlainText(info_text)

        except Exception as e:
            # 处理异常情况
            if Video.is_image_file(selected_video_path):
                # 获取图像信息
                img = cv2.imread(selected_video_path)
                height, width, channels = img.shape

                # 添加分辨率信息
                resolution_info = f"分辨率: {width} x {height}"
                info_text = f"{resolution_info}  \n通道: {channels}"

                self.video_info.setPlainText(info_text)
            else:
                self.video_info.setPlainText(f"获取视频信息时发生错误")

        # 单位转换

    def convert_bytes_to_readable(self, size_in_bytes):
        units = ["B", "KB", "MB", "GB", "TB"]
        unit_index = 0
        # 初始化待转换的字节数
        size = size_in_bytes
        # 循环条件为 size大于等于1024且单位索引小于单位列表长度减1
        while size >= 1024 and unit_index < len(units) - 1:
            size /= 1024
            unit_index += 1
        formatted_size = "{:.2f}".format(size)
        # 返回格式化后的结果，包括单位
        return f"{formatted_size} {units[unit_index]}"

    # 显示剩余时间
    def displayTime(self, ms):
        # print(ms)
        minutes = int(ms / 60000)
        seconds = int((ms % 60000) / 1000)
        milliseconds = int(ms % 1000)
        if self.tabWidget.currentIndex() == 0:
            self.video_time.setText('{}:{}'.format(minutes, seconds))
            self.cut_time.setText('{}:{}'.format(minutes, seconds))
        elif self.tabWidget.currentIndex() == 1:
            self.video_time_2.setText('{}:{}'.format(minutes, seconds))
            self.action_list.clear()

    # 暂停/播放
    def playPause(self):
        if self.tabWidget.currentIndex() == 0:
            if self.vdplayer.state() == 1:
                self.play_pause.setIcon(self.play_ico)
                self.vdplayer.pause()
            else:
                self.play_pause.setIcon(self.pause_ico)
                self.vdplayer.play()
        elif self.tabWidget.currentIndex() == 1:
            if self.idplayer.state() == 1:
                self.play_pause_2.setIcon(self.play_ico)
                self.idplayer.pause()
            else:
                self.play_pause_2.setIcon(self.pause_ico)
                self.speed_play(self.vdplayer)

    # 保存标签
    def saveLabel(self):
        select_path = self.selected_path
        if os.path.exists(select_path):
            # 获取文件的基本名称（不包含路径）
            file_name = os.path.basename(select_path)
            # 获取文件的目录路径
            directory = os.path.dirname(select_path)
            # 将文件名中的后缀名更改为.ini
            new_file_name = os.path.splitext(file_name)[0] + ".ini"
            # 构建新的文件路径
            new_path = os.path.join(directory, new_file_name)
            content_to_write = self.video_label_edit.toPlainText()
            if os.path.exists(new_path):
                with open(new_path, "w") as file:
                    file.write(content_to_write)
            else:
                # 如果文件不存在，就创建一个新文件并写入内容
                with open(new_path, "w") as file:
                    file.write(content_to_write)
        else:
            print("保存失败")

    # 搜索动作标签
    def search_action(self):
        if not Globals.dict_text:
            return
        self.frame_dict = self.load_frame_dict(Globals.dict_text)
        if not self.frame_dict:
            return
        item = self.item_search.text()
        self.search_result.clear()
        if item:
            for t, action_list in self.frame_dict.items():
                for order in action_list:
                    if item in action_list[order]:
                        # print(action_list[action])
                        # item = QListWidgetItem(video_file)
                        self.search_result.addItem(f"时间：{t} 动作：{order}-{action_list[order]}")
            # self.item_search.setText("")

    # 绘制折线图
    def drawLineChart(self):
        # 如果全局变量 Globals.dict_text 为空，则返回
        if not Globals.dict_text:
            return

        # 从 Globals.dict_text 加载数据到 self.frame_dict
        self.frame_dict = self.load_frame_dict(Globals.dict_text)

        # 如果 self.frame_dict 为空，则返回
        if not self.frame_dict:
            return

        # 清除之前的绘图
        self.ax.clear()

        # 创建一个字典来存储每个唯一动作的计数
        action_counts = {}

        # 遍历 frame_dict 并计算每个唯一动作的发生次数
        for t, action_dict in self.frame_dict.items():
            # 为每个时间戳初始化计数字典
            counts = {}

            for order, action in action_dict.items():
                if action not in counts:
                    counts[action] = 0

                counts[action] += 1

                if action not in action_counts:
                    action_counts[action] = {'t': [t], 'count': [1]}
                else:
                    # 检查 't' 是否发生变化
                    if action_counts[action]['t'][-1] != t:
                        action_counts[action]['t'].append(t)
                        action_counts[action]['count'].append(1)
                    else:
                        # 更新最后一个 'count' 的值
                        action_counts[action]['count'][-1] = counts[action]

        # 初始化一个包含所有时间点的集合
        all_t = set()

        # 遍历 action_counts 获取所有时间点
        for action, data in action_counts.items():
            all_t.update(data['t'])

        # 遍历 action_counts 更新数据结构
        for action, data in action_counts.items():
            # 创建补全后的数据结构
            filled_data = {'t': list(all_t), 'count': [0] * len(all_t)}

            # 将已有数据填入相应位置
            for i, t in enumerate(data['t']):
                # 只在时间点在原始数据之后的位置进行填充
                if t in filled_data['t']:
                    filled_data['count'][filled_data['t'].index(t)] = data['count'][i]

            # 剔除前面的零
            first_non_zero_index = next((i for i, count in enumerate(filled_data['count']) if count != 0), None)
            if first_non_zero_index is not None:
                filled_data['t'] = filled_data['t'][first_non_zero_index:]
                filled_data['count'] = filled_data['count'][first_non_zero_index:]

            # 替换原始数据
            action_counts[action] = filled_data
            last_10_data = {'t': filled_data['t'][-10:], 'count': filled_data['count'][-10:]}

            # 进行绘图等操作
            self.ax.plot(last_10_data['t'], last_10_data['count'], marker='o', label=action)

        # 设置标签和标题
        self.ax.set_xlabel('时间', color='white')
        self.ax.set_ylabel('数目', color='white')
        self.ax.set_title('每秒动作数目趋势', color='white')

        # 添加图例
        self.ax.legend()

        # 重新绘制画布
        self.canvas.draw()

    # 绘制饼图
    def drawPieChart(self):
        # 如果全局变量 Globals.dict_text 为空，则返回
        if not Globals.dict_text:
            return

        # 从 Globals.dict_text 加载数据到 self.frame_dict
        self.frame_dict = self.load_frame_dict(Globals.dict_text)

        # 如果 self.frame_dict 为空，则返回
        if not self.frame_dict:
            return

        # 清除之前的绘图
        self.ax2.clear()

        # 创建一个字典来存储每个唯一动作的计数
        action_counts = {}

        # 遍历 frame_dict 并计算每个唯一动作的发生次数
        for t, action_dict in self.frame_dict.items():
            # 为每个时间戳初始化计数字典
            counts = {}

            for order, action in action_dict.items():
                if action not in counts:
                    counts[action] = 0

                counts[action] += 1

            # 累积每个动作在所有时间戳上的计数
            for action, count in counts.items():
                if action not in action_counts:
                    action_counts[action] = count
                else:
                    action_counts[action] += count

        # 为累积计数绘制饼图
        labels = list(action_counts.keys())
        counts = list(action_counts.values())

        # 绘制饼图
        self.ax2.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90, textprops={'color': 'white'})

        # 设置标题
        self.ax2.set_title('动作分布', color='white')

        # 重新绘制画布
        self.canvas2.draw()

    def tabChanged(self):
        # 检查是否需要暂停播放器
        self.vdplayer.pause()
        # 停止计时器
        self.timer_cv.stop()
        # 检查是否已经捕获了视频流
        if self.capture is not None:
            # 释放视频流捕获
            self.capture.release()


if __name__ == "__main__":
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
    app = QApplication([])
    # app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    user = User_management()
    user.show()
    app.exec()

    pyqt5 = MainWindow()
    pyqt5.show()
    app.exec()
