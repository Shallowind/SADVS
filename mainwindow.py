import ctypes
import os
import sys
import threading
import time
# import ffmpeg
from datetime import datetime
from json import loads

import matplotx
import GPUtil
import cv2
import psutil
from PyQt5.QtCore import Qt, QTimer, QFileInfo, pyqtSignal, QSize, QTime, QObject, QThread
from PyQt5.QtGui import QIcon, QImage, QPixmap, QColor
from PyQt5.QtWidgets import QFileDialog, QApplication, QSplitter, QTreeWidgetItem, QListView, QTreeWidget, QMainWindow, \
    QMenu, QAction, QMessageBox, QListWidgetItem, QGroupBox, QLabel, QTabBar, QInputDialog, QLineEdit, QFileSystemModel, \
    QWidget
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from moviepy.video.io.VideoFileClip import VideoFileClip
from numpy import sqrt
from qfluentwidgets import setTheme, Theme, Action, setCustomStyleSheet, RoundMenu, InfoBar, InfoBarIcon, \
    InfoBarPosition, PushButton, qconfig, FluentIcon, MessageBox, StrongBodyLabel

import detect
import detect_yolov5
# import detect_yolov8
from IdentifyResults import getExpList
from PDF import PDFReader, PDF
from UI.VdWidget import VdWidget
from UI.centralwidget import Ui_centralwidget
from UI.labels_settings import LabelsSettings
from UI.model_settings import ModelSettings
from UI.user_management import User_management
from UI.video_widget import Ui_video_widget
from Users import UserManager
from easyowov2 import easydemo
# from mainwindow_ui import Ui_MainWindow
from utils.myutil import Globals, get_video_info, ConsoleRedirector

ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("myappid")
from UI.Video import Video


def create_and_setup_splitter(direction, widgets, stretch_factors, layout):
    splitter = QSplitter(direction)
    for widget in widgets:
        splitter.addWidget(widget)
    for i, factor in enumerate(stretch_factors):
        splitter.setStretchFactor(i, factor)
    layout.addWidget(splitter)
    splitter.setStyleSheet("""
        QSplitter {
            background-color: 19232d;
            spacing: 50px;
        }
        QSplitter::handle {
            background-color: 19232d;
            spacing: 50px;
        }
    """)
    return splitter


class IdentifyThreads(QObject):
    # 定义一个信号
    pdf_added = pyqtSignal(str)
    memory = pyqtSignal(str)

    def __init__(self, parent=None):
        super(IdentifyThreads, self).__init__(parent)
        self.pdf_path = None

    def start_identify(main_window, self):
        pdf = PDF()
        now_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_save_path = os.path.join(Globals.settings['pdf_save_path'],
                                     now_time + '.pdf')
        # 开始识别
        Globals.detection_run = True
        self.start_identify.setEnabled(False)
        self.stop_identify.setEnabled(True)

        if self.selected_item.isCamera:
            source = self.selected_item.device
            UseCam = True
        else:
            source = self.selected_item.path
            UseCam = False
        try:
            if Globals.settings['model_select'] == 'yolov5':
                # 使用yolov5模型进行检测
                detect_yolov5.run(source=source, weights=Globals.settings['pt_path'], pdf=pdf,
                                  show_label=self.camera_2, project=Globals.settings['save_path'],
                                  save_img=True, use_camera=UseCam, show_window=self, classes=Globals.select_labels,
                                  max_det=int(Globals.settings['max_det']), conf_thres=Globals.settings['conf'],
                                  iou_thres=Globals.settings['iou'], line_thickness=Globals.settings['line_thickness'])
            elif Globals.settings['model_select'] == 'yolo_slowfast':
                # 使用yolo_slowfast模型进行检测
                detect.run(source=source, weights=Globals.settings['pt_path'], pdf=pdf, show_label=self.camera_2,
                           select_objects=Globals.select_objects, project=Globals.settings['save_path'],
                           save_img=True, use_camera=UseCam, show_window=self, select_labels=Globals.select_labels,
                           max_det=int(Globals.settings['max_det']), conf_thres=Globals.settings['conf'],
                           iou_thres=Globals.settings['iou'], line_thickness=Globals.settings['line_thickness'])
            elif Globals.settings['model_select'] == 'yolov8_BRA_DCNv3':
                pass
                # 使用yolo_slowfast模型进行检测
                # detect_yolov8.run(source=source, weights=Globals.settings['pt_path'], pdf=pdf, show_label=self.camera_2,
                #                   select_objects=Globals.select_objects, project=Globals.settings['save_path'],
                #                   save_img=True, use_camera=UseCam, show_window=self,
                #                   select_labels=Globals.select_labels,
                #                   max_det=int(Globals.settings['max_det']), conf_thres=Globals.settings['conf'],
                #                   iou_thres=Globals.settings['iou'], line_thickness=Globals.settings['line_thickness'])
            elif Globals.settings['model_select'] == 'yowo':
                # 使用yowo模型进行检测
                thread = threading.Thread(target=easydemo.run_easydemo(show_window=self, video=source))
                thread.daemon = True  # 主界面关闭时自动退出此线程
                thread.start()
            if Globals.settings['model_select'] == 'yolo_slowfast':
                pdf_data = pdf.pdf_data
                pdf.build_pdf(pdf_save_path, source, save_path=Globals.settings['save_path'],
                              model_select=Globals.settings['model_select'],
                              select_labels=Globals.select_labels, UseCam=UseCam)
                expPath = getExpList()
                base_path = os.path.dirname(os.path.abspath(__file__))
                base_path = os.path.join(base_path, "exception")
                if expPath:
                    path = expPath[-1]
                    path_new = os.path.join(base_path, path, now_time + '.pdf')
                    pdf.pdf_data = pdf_data
                    pdf.build_pdf(path_new, source, save_path=Globals.settings['save_path'],
                                  model_select=Globals.settings['model_select'],
                                  select_labels=Globals.select_labels, UseCam=UseCam)
                main_window.pdf_added.emit(pdf_save_path)
        # 内存超限
        except Exception as e:
            main_window.memory.emit(str(e))


class UsageThread(QThread):
    usage_signal = pyqtSignal(float, float, float, float)

    def run(self):
        while True:
            current_time = QTime.currentTime().toString('hh:mm:ss')
            cpu_usage = psutil.cpu_percent(interval=1)
            GPUs = GPUtil.getGPUs()
            # 获取最后一个GPU的使用率
            gpu_usage = GPUs[0].load * 100
            # 显存
            gpu_memory = GPUs[0].memoryUtil * 100

            self.usage_signal.emit(cpu_usage, gpu_usage, psutil.virtual_memory().percent, gpu_memory)


class Mainwindow(Ui_centralwidget, QWidget):
    signal = pyqtSignal()
    warnSignal = pyqtSignal(str)

    def __init__(self, parent=None):
        # 加载designer设计的ui程序
        super().__init__()
        self.warnSignal.connect(self.show_warn)
        self.father = parent
        self.setupUi(self)
        # setTheme(Theme.DARK)
        # self.controller = controller
        self.icon = QIcon()
        self.icon.addPixmap(QPixmap(":/gallery/UI/logo.ico"), QIcon.Normal, QIcon.Off)
        self.setWindowIcon(self.icon)
        # with open(f'resources/qss/dark/demo.qss', encoding='utf-8') as f:
        #     str1 = f.read()
        #     self.setStyleSheet(str1)
        #     setCustomStyleSheet(self.video_widget, str1, str1)
        #     setCustomStyleSheet(self.widget_list, str1, str1)
        # str1 = ""
        # setCustomStyleSheet(self.video_widget, str1, str1)
        self.path = ''
        self.worker = IdentifyThreads()
        self.worker.pdf_added.connect(self.add_pdf_tab)
        self.worker.memory.connect(self.memory_out)
        self.conwidget.setVisible(False)
        self.mod_widget.setVisible(False)
        self.cutwidget.setVisible(False)
        # self.video_tree.setBorderVisible(True)
        # self.video_tree.setBorderRadius(8)

        # self.stackedWidget.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        self.signal.connect(self.initcomplete)
        # 播放器
        self.vdplayer_1 = Video()
        self.vdplayer = self.vdplayer_1
        self.video_time_all = self.video_time_all_1
        self.label_inf_widget_2.setVisible(False)
        self.video_slider_2.setVisible(False)
        self.label_inf_widget_3.setVisible(False)
        self.video_slider_3.setVisible(False)
        self.label_inf_widget_4.setVisible(False)
        self.video_slider_4.setVisible(False)

        self.VdWidgets = []
        self.Vdplayers = []
        self.VdUIs = []

        for i in range(64):
            obj = VdWidget()
            obj.id = i
            obj.doubleClicked.connect(self.create_connect(obj))
            obj.mousePressed.connect(self.create_mouse_press(obj))
            VdUI = Ui_video_widget()
            VdUI.setupUi(obj)
            self.VdWidgets.append(obj)
            self.VdUIs.append(VdUI)

            Vdplayer = Video()
            Vdplayer.setVideoOutput([VdUI.player_show, VdUI.player_widget])
            self.Vdplayers.append(Vdplayer)

        self.thisvdwidget = self.VdWidgets[0]
        self.maximized_widget = None

        for i in range(1, 9):
            self.set_video_to_n(i)
        self.video_to_n.setCurrentItem("9")
        self.videoWidgetChanged(3)

        # self.VdUIs[1].player_widget.setStyleSheet("background-color: white")

        # self.vdplayer_1.setVideoOutput([VdUI1.player_show, VdUI1.player_widget], self.video_slider_1,
        #                                self.video_time_1,
        #                                self.cut_time)
        # self.vdplayer_2 = Video()
        # self.vdplayer_2.setVideoOutput([self.player_show_2, self.player_widget_2], self.video_slider_2,
        #                                self.video_time_2,
        #                                self.cut_time)
        # self.vdplayer_3 = Video()
        # self.vdplayer_3.setVideoOutput([self.player_show_3, self.player_widget_3], self.video_slider_3,
        #                                self.video_time_3,
        #                                self.cut_time)
        # self.vdplayer_4 = Video()
        # self.vdplayer_4.setVideoOutput([self.player_show_4, self.player_widget_4], self.video_slider_4,
        #                                self.video_time_4,
        #                                self.cut_time)
        self.change_video_widget_enabled = False
        # self.widget_1.doubleClicked.connect(lambda: self.videoWidgetChanged(1))
        # self.widget_1.mousePressed.connect(lambda: self.selectVideoWidgetChanged(1))
        # self.widget_2.doubleClicked.connect(lambda: self.videoWidgetChanged(2))
        # self.widget_2.mousePressed.connect(lambda: self.selectVideoWidgetChanged(2))
        # self.widget_3.doubleClicked.connect(lambda: self.videoWidgetChanged(3))
        # self.widget_3.mousePressed.connect(lambda: self.selectVideoWidgetChanged(3))
        # self.widget_4.doubleClicked.connect(lambda: self.videoWidgetChanged(4))
        # self.widget_4.mousePressed.connect(lambda: self.selectVideoWidgetChanged(4))
        self.idplayer = Video()
        self.idplayer.setVideoOutput([self.camera_2, self.camera_2], None, None,
                                     None)
        # 选择文件夹
        # self.video_select.triggered.connect(self.openVideoFolder)
        # # 设置默认视频目录
        # self.video_path.triggered.connect(self.changePath)
        # # 默认保存路径
        # self.save_path.triggered.connect(self.changePath)
        # # 退出
        # self.quit.triggered.connect(self.Quit)
        # # 模型查看
        # self.model_view.triggered.connect(self.modelView)
        # # 标签集设置
        # self.action_sets.triggered.connect(self.labelSetsSettings)
        # # 使用信息说明
        # self.Use_Information.triggered.connect(self.displayUseInformation)
        # # 用户管理
        # self.action_control.triggered.connect(self.controller.show_user_widget)
        # # 登出
        # self.action_signout.triggered.connect(self.signout)
        # # 修改密码
        # self.action_edit.triggered.connect(self.changePassword)
        # # 版本信息说明
        # self.Version_Information.triggered.connect(self.displayVersionInformation)
        # 暂停
        self.play_pause.clicked.connect(self.playPause)
        # 倍速功能
        self.speed.currentIndexChanged.connect(self.speedChange)
        # 上一个/下一个视频
        self.prev.clicked.connect(self.prevVideo)
        self.next.clicked.connect(self.nextVideo)
        # 工作区/原始列表
        self.listcon.clicked.connect(self.refreshWidgetlist)
        self.workcon.clicked.connect(self.refreshWidgetlist)
        self.terminalcon.clicked.connect(self.refreshWidgetlist)
        self.original.activity = True
        self.works.activity = True
        self.controlWidget.activity = True
        # 双击播放
        self.video_tree.setSortingEnabled(False)
        self.video_tree.setBorderRadius(10)
        self.video_tree.itemDoubleClicked.connect(self.CameraVideo)
        self.work_list.itemDoubleClicked.connect(self.WorkListPreview)
        # 文件树展开
        self.video_tree.itemExpanded.connect(self.loadSubtree)
        # 右键菜单
        self.video_tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.video_tree.customContextMenuRequested.connect(self.showContextMenu)
        # 工作区右键菜单
        self.work_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.work_list.customContextMenuRequested.connect(self.worklistShowContextMenu)
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
        self.item_search_button.clicked.connect(self.searchAction)
        # 获取摄像头id列表
        self.camera_id_list = []
        camera_thread = threading.Thread(target=self.initialize_camera)
        camera_thread.daemon = True  # 主界面关闭时自动退出此线程
        camera_thread.start()

        self.capture = None
        # 启动识别
        self.start_identify.clicked.connect(self.startIdentifyClicked)
        self.settings_window = None
        # 停止识别
        self.stop_identify.clicked.connect(self.stopIdentify)
        # tab切换
        self.stackedWidget.currentChanged.connect(self.tabChanged)
        # 定时器
        self.timer_cv = QTimer()
        self.timer_cv.timeout.connect(self.updateFrame)
        self.timer_cv.setInterval(30)  # 1000毫秒 = 1秒
        # 剪辑模式
        # 创建图形
        self.carflow = 0
        self.menflow = 0
        if qconfig.theme == Theme.LIGHT:
            style = 'light'
        else:
            style = 'dark'
        with plt.style.context(matplotx.styles.pitaya_smoothie[style]):
            plt.rcParams['font.family'] = 'Microsoft YaHei'

            self.figure, self.ax = plt.subplots()
            self.ax.set_xlabel('时间')
            self.ax.set_ylabel('数目')
            self.ax.tick_params(axis='both')
            self.ax.set_title('每秒动作数目趋势')
            self.canvas = FigureCanvas(self.figure)
            self.toolbar = NavigationToolbar(self.canvas, self)
            # 添加到布局中
            self.verticalLayout_6.addWidget(self.toolbar)
            self.verticalLayout_6.addWidget(self.canvas)

            self.figure3, self.ax3 = plt.subplots()
            # self.ax3.set_xlabel('时间')
            # self.ax3.set_ylabel('数目')
            self.ax3.tick_params(axis='both')
            self.ax3.set_title('流量图')
            self.canvas3 = FigureCanvas(self.figure3)
            self.toolbar3 = StrongBodyLabel(f"车流量：{self.carflow} 辆/秒     人流量：{self.menflow} 人/秒")
            # 添加到布局中
            self.verticalLayout_9.addWidget(self.toolbar3)
            self.verticalLayout_9.addWidget(self.canvas3)

            # 创建图形
            self.figure2, self.ax2 = plt.subplots()
            labels = ('unkown1',)
            sizes = [100]
            self.ax2.set_title('动作分布')
            self.ax2.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, textprops={})
            # self.figure2.set_facecolor('#19232d')
            self.canvas2 = FigureCanvas(self.figure2)
            self.toolbar2 = NavigationToolbar(self.canvas2, self)
            # 添加到布局中
            self.verticalLayout_7.addWidget(self.toolbar2)
            self.verticalLayout_7.addWidget(self.canvas2)

        # 启用多选
        self.video_tree.setSelectionMode(QTreeWidget.ExtendedSelection)
        # self.video_tree.setStyleSheet("""
        # QTreeWidget::branch:has-siblings:!adjoins-item{ \
        #     border-image:None 0;\
        # }\
        # QTreeWidget::branch:has-siblings:adjoins-item{\
        #     border-image:None 0;\
        # }\
        # QTreeWidget::branch:!has-children:!has-siblings:adjoins-item{\
        #     border-image:None 0;\
        # }\
        # QTreeWidget::branch:has-siblings:adjoins-item{\
        #     border-image:None 0;\
        # }\
        # QLineEdit{\
        #     padding: 0;\
        #     margin: 0;\
        # }\
        # """)
        self.work_list.setSelectionMode(QTreeWidget.ExtendedSelection)
        sys.stdout = ConsoleRedirector(self, self.terminal_2)
        print("e视平安——面向公共交通安全的人工智能守护平台 - 2.0.1")
        # sys.stderr = ConsoleRedirector(self, self.terminal_2, QColor(255, 0, 0))
        print()

        self.videotree = []
        self.selected_folder = ""
        self.selected_path = ""
        self.selected_item = None
        self.labsettings_window = None

        # 分割器
        create_and_setup_splitter(Qt.Vertical, [self.original, self.works], [2, 1],
                                  self.verticalLayout_13)
        create_and_setup_splitter(Qt.Horizontal, [self.widget_list, self.vstackedWidget], [8, 10],
                                  self.horizontalLayout_14)
        create_and_setup_splitter(Qt.Horizontal, [self.video_widget, self.video_label_widget], [5, 2],
                                  self.horizontalLayout_5)
        create_and_setup_splitter(Qt.Horizontal, [self.video_widget_2, self.video_label_widget_2], [5, 1],
                                  self.horizontalLayout_7)
        create_and_setup_splitter(Qt.Vertical, [self.useSpace, self.controlWidget], [10, 1],
                                  self.verticalLayout_10)
        create_and_setup_splitter(Qt.Horizontal, [self.exp_widget, self.splitter], [10, 4],
                                  self.horizontalLayout_9)

        # 图标
        self.file_icon = QIcon(":/gallery/UI/file_ico.png")
        self.folder_icon = QIcon(":/gallery/UI/folder_ico.ico")
        self.image_icon = QIcon(":/gallery/UI/img_ico.ico")
        self.text_icon = QIcon(":/gallery/UI/text_ico.png")
        self.video_icon = QIcon(":/gallery/UI/video_ico.png")
        self.camera_icon = QIcon(":/gallery/UI/cam_ico.png")
        self.play_ico = QIcon(":/gallery/UI/play_ico.png")
        self.pause_ico = QIcon(":/gallery/UI/pause_ico.png")
        # self.menubar.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())

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

        self.SegmentedWidget.addItem(
            routeKey="ori_video",
            text="原始视频",
            onClick=lambda: self.stackedWidget.setCurrentWidget(self.ori_video),
        )
        self.SegmentedWidget.addItem(
            routeKey="act_identify",
            text="行为检测",
            onClick=lambda: self.stackedWidget.setCurrentWidget(self.act_identify),
        )
        self.SegmentedWidget.addItem(
            routeKey="tab",
            text="异常检测",
            onClick=lambda: (
                self.stackedWidget.setCurrentWidget(self.tab),
                self.tabClicked()
            )
        )
        self.SegmentedWidget.setCurrentItem("ori_video")
        self.result_display.addItem(
            routeKey="search",
            text="搜索",
            onClick=lambda: self.stackedWidget_2.setCurrentWidget(self.search_tab),
        )
        self.result_display.addItem(
            routeKey="plot",
            text="折线图",
            onClick=lambda: self.stackedWidget_2.setCurrentWidget(self.plot_tab),
        )
        self.result_display.addItem(
            routeKey="pie",
            text="饼图",
            onClick=lambda: self.stackedWidget_2.setCurrentWidget(self.tab_3),
        )
        self.result_display.addItem(
            routeKey="flow",
            text="流量图",
            onClick=lambda: self.stackedWidget_2.setCurrentWidget(self.flow),
        )
        self.result_display.setCurrentItem("plot")
        self.stackedWidget_2.setCurrentWidget(self.plot_tab)

        self.exp_list.clicked.connect(self.listClicked)

        self.v_d_select.addItem(
            routeKey="video",
            text="本地视频",
            onClick=lambda: self.select_V_D(1),
        )
        self.v_d_select.addItem(
            routeKey="camera",
            text="摄像头",
            onClick=lambda: self.select_V_D(2),
        )
        self.v_d_select.setCurrentItem("video")
        # 原始列表
        self.CommandBar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        action = Action(FluentIcon.FOLDER_ADD, "选择", self)
        action.triggered.connect(self.openVideoFolder)
        self.CommandBar.addAction(action)
        # self.CommandBar.addSeparator()
        action = Action(FluentIcon.ADD, "添加工作区", self)
        action.triggered.connect(self.addWorkspace)
        self.CommandBar.addAction(action)
        # self.CommandBar.addSeparator()
        action = Action(FluentIcon.EDIT, "重命名", self)
        action.triggered.connect(self.rename_file)
        self.CommandBar.addAction(action)
        # self.CommandBar.addSeparator()
        action = Action(QIcon(":/gallery/op_icon/delete-file.png"), "删除", self)
        action.triggered.connect(self.delete_file)
        self.CommandBar.addAction(action)

        self.CommandBar_2.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        action = Action(FluentIcon.BROOM, "清空", self)
        action.triggered.connect(lambda: self.work_list.clear())
        self.CommandBar_2.addAction(action)
        # self.CommandBar_2.addSeparator()
        action = Action(FluentIcon.EDIT, "重命名", self)
        action.triggered.connect(self.worklist_rename)
        self.CommandBar_2.addAction(action)
        # self.CommandBar_2.addSeparator()
        action = Action(FluentIcon.REMOVE, "移除", self)
        action.triggered.connect(self.worklist_delete)
        self.CommandBar_2.addAction(action)
        self.setthemeicon()
        # 添加状态栏
        # self.sta_time_label = QLabel()
        # self.statusbar.addPermanentWidget(self.sta_time_label)
        self.usage_thread = UsageThread()

        self.have_tip = False
        self.usage_thread.usage_signal.connect(self.update_usage)
        self.usage_thread.start()

        # 添加关闭选项卡按钮
        # self.stackedWidget.setTabsClosable(True)
        # self.stackedWidget.tabCloseRequested.connect(self.on_tab_close_requested)
        # 基本标签页不可移除
        # for i in range(3):
        # self.stackedWidget.tabBar().setTabButton(i, QTabBar.RightSide, None)

    def memory_out(self, e):
        w = MessageBox(
            title='识别发生错误！',
            content='由于系统GPU硬件不支持/机器Cuda版本过低/内存、显存不足等问题导致识别终止，建议依照报错信息更新设备环境或者重启设备。'
                    '错误信息：' + e,
            parent=self
        )
        w.show()
        self.stop_identify.setEnabled(True)
        self.start_identify.setEnabled(True)

    def show_warn(self, warn_text):
        InfoBar.warning(
            title='警告！',
            content=warn_text,
            orient=Qt.Horizontal,
            isClosable=True,
            position=InfoBarPosition.BOTTOM_RIGHT,
            # position='Custom',   # NOTE: use custom info bar manager
            duration=3000,
            parent=self
        )

    def initcomplete(self):
        InfoBar.success(
            title='检测到设备：' + str(self.camera_id_list),
            content='摄像头初始化成功！现在可以进行摄像头视频的预览了！',
            orient=Qt.Horizontal,
            isClosable=True,
            position=InfoBarPosition.BOTTOM_RIGHT,
            # position='Custom',   # NOTE: use custom info bar manager
            duration=3000,
            parent=self.parent().parent().parent()
        )

    def setthemeicon(self):
        str1 = ''
        theme = qconfig.theme
        if theme == Theme.LIGHT:
            str1 = '_1'
        self.listcon.setIcon(FluentIcon.FOLDER)
        self.workcon.setIcon(FluentIcon.LABEL)
        self.terminalcon.setIcon(FluentIcon.COMMAND_PROMPT)
        self.save_label.setIcon(f":/gallery/op_icon/save{str1}.png")
        self.item_search_button.setIcon(f":/gallery/op_icon/search{str1}.png")
        if qconfig.theme == Theme.LIGHT:
            style = 'light'
        else:
            style = 'dark'
        with plt.style.context(matplotx.styles.pitaya_smoothie[style]):
            plt.rcParams['font.family'] = 'Microsoft YaHei'
            # 假设你的布局对象是self.verticalLayout_6和self.verticalLayout_7
            # 移除canvas和toolbar
            self.verticalLayout_6.removeWidget(self.canvas)
            self.verticalLayout_6.removeWidget(self.toolbar)
            self.verticalLayout_7.removeWidget(self.canvas2)
            self.verticalLayout_7.removeWidget(self.toolbar2)
            self.verticalLayout_9.removeWidget(self.canvas3)
            self.verticalLayout_9.removeWidget(self.toolbar3)

            # 删除canvas和toolbar
            self.canvas.deleteLater()
            self.toolbar.deleteLater()
            self.canvas2.deleteLater()
            self.toolbar2.deleteLater()
            self.canvas3.deleteLater()
            self.toolbar3.deleteLater()

            # 将figure，ax，canvas和toolbar设置为None
            self.figure = None
            self.ax = None
            self.canvas = None
            self.toolbar = None
            self.figure2 = None
            self.ax2 = None
            self.canvas2 = None
            self.toolbar2 = None
            self.figure3 = None
            self.ax3 = None
            self.canvas3 = None
            self.toolbar3 = None

            self.figure, self.ax = plt.subplots()
            self.ax.set_xlabel('时间')
            self.ax.set_ylabel('数目')
            self.ax.tick_params(axis='both')
            # self.ax.spines['bottom'].set_color()
            # self.ax.spines['top'].set_color()
            # self.ax.spines['right'].set_color()
            # self.ax.spines['left'].set_color()
            self.ax.set_title('每秒动作数目趋势')
            self.canvas = FigureCanvas(self.figure)
            self.toolbar = NavigationToolbar(self.canvas, self)
            # 添加到布局中
            self.verticalLayout_6.addWidget(self.toolbar)
            self.verticalLayout_6.addWidget(self.canvas)

            self.figure3, self.ax3 = plt.subplots()
            # self.ax3.set_xlabel('时间')
            # self.ax3.set_ylabel('数目')
            self.ax3.tick_params(axis='both')
            self.ax3.set_title('流量图')
            self.canvas3 = FigureCanvas(self.figure3)
            self.toolbar3 = StrongBodyLabel("车流量：0 辆/秒     人流量：0 人/秒")
            # 添加到布局中
            self.verticalLayout_9.addWidget(self.toolbar3)
            self.verticalLayout_9.addWidget(self.canvas3)

            # 创建图形
            self.figure2, self.ax2 = plt.subplots()
            labels = ('unkown1',)
            sizes = [100]
            self.ax2.set_title('动作分布')
            self.ax2.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, textprops={})
            # self.figure2.set_facecolor('#19232d')
            self.canvas2 = FigureCanvas(self.figure2)
            self.toolbar2 = NavigationToolbar(self.canvas2, self)
            # 添加到布局中
            self.verticalLayout_7.addWidget(self.toolbar2)
            self.verticalLayout_7.addWidget(self.canvas2)

            self.drawLineChart()
            self.drawPieChart()

    def set_video_to_n(self, i):
        self.video_to_n.addItem(
            routeKey=str(i ** 2),
            text=str(i ** 2),
            onClick=lambda: self.videoWidgetChanged(i),
        )

    def create_connect(self, obj):
        return lambda: self.vdwidget_doubele_clicked(obj)

    def create_mouse_press(self, obj):
        return lambda: self.vdwidget_clicked(obj)

    def vdwidget_clicked(self, obj):
        self.thisvdwidget = obj
        pass

    def vdwidget_doubele_clicked(self, obj):
        self.thisvdwidget = obj
        if self.maximized_widget is None:
            # 如果当前没有VdWidget被放大显示，那么放大显示被双击的VdWidget
            for widget in self.VdWidgets:
                if widget != obj:
                    widget.setVisible(False)  # 隐藏其他的VdWidget
            self.maximized_widget = obj
        else:
            self.videoWidgetChanged(int(sqrt(int(self.video_to_n.currentItem().text()))))
            self.maximized_widget = None
        pass

    def showMaximized(self):
        admin = ""
        if Globals.user.is_admin:
            admin = " (管理员)"
        self.action_xiaohuangren.setText(Globals.user.get_username() + admin)
        super().showMaximized()

    def signout(self):
        self.controller.show_user_management()
        self.close()

    def changePassword(self):
        username = Globals.user.get_username()
        manager = UserManager()
        # 弹出输入超级管理员密码，正确即可修改密码
        uperadmin_password, ok = QInputDialog.getText(self, '修改密码', '请你输入超级管理员密码:', QLineEdit.Password)
        if ok:
            if manager.check_password('superadmin', uperadmin_password):
                # 弹出输入新密码对话框
                new_password, ok = QInputDialog.getText(self, '修改密码', '请输入新密码:', QLineEdit.Password)
                if ok:
                    manager.change_password(username, new_password)
                    QMessageBox.information(self, '修改密码', '修改密码成功')
            else:
                QMessageBox.warning(self, '修改密码', '密码错误')

    def add_pdf_tab(self, pdf_path):
        w = InfoBar.success(
            title='识别完成',
            content='识别报告已生成，是否需要查看？',
            orient=Qt.Vertical,  # vertical layout
            isClosable=True,
            position=InfoBarPosition.TOP,
            duration=8000,
            parent=self
        )
        button = PushButton('查看')
        button.clicked.connect(lambda: self.on_add_pdf_tab(pdf_path))
        w.addWidget(button)
        w.show()

    def on_add_pdf_tab(self, pdf_path):
        # 弹出 PDFReader 对话框
        pdf_reader = PDFReader(pdf_path)
        self.stackedWidget.addWidget(pdf_reader)
        self.SegmentedWidget.addItem(
            routeKey="pdfreader",
            text="识别报告",
            onClick=lambda: self.stackedWidget.setCurrentWidget(pdf_reader),
        )
        # 点击新增的选项卡
        self.stackedWidget.setCurrentIndex(self.stackedWidget.count() - 1)
        self.SegmentedWidget.setCurrentItem("pdfreader")

    def on_tab_close_requested(self, index):
        # 删除当前选项卡
        self.stackedWidget.removeTab(index)

    def update_usage(self, cpu, gpu, mem, gpu_mem):
        self.ProgressRing.setValue(int(cpu))
        self.ProgressRing_2.setValue(int(gpu))
        self.ProgressRing_3.setValue(int(mem))
        self.ProgressRing_4.setValue(int(gpu_mem))
        if not self.have_tip and (cpu > 90 or mem > 90 or gpu_mem > 90):
            self.have_tip = True
            w = MessageBox(
                title='性能警告',
                content='识别所需资源占用不足，可能会出现卡顿或其它不可预知的问题！',
                parent=self
            )
            w.show()

    def videoWidgetChanged(self, index):
        # 清空gridLayout_17里的所有
        while self.gridLayout_17.count():
            child = self.gridLayout_17.takeAt(0)
            if child.widget():
                self.gridLayout_17.removeWidget(child.widget())
                child.widget().setVisible(False)

        # 根据选中的按钮来填充gridLayout_17
        rows = cols = index  # 计算正方形的行数和列数
        for i in range(rows):
            for j in range(cols):
                self.gridLayout_17.addWidget(self.VdWidgets[i * cols + j], i, j)
                self.VdWidgets[i * cols + j].setVisible(True)

    def listClicked(self):
        item = self.exp_list.currentItem()
        if item and self.stackedWidget.currentIndex() == 2:
            path = item.data(Qt.UserRole)
            if path:
                try:
                    # 显示异常图片
                    pixmap = QPixmap(path.split('.')[0] + 'all.jpg')
                    self.expplayer.setAlignment(Qt.AlignCenter)
                    self.expplayer.setPixmap(pixmap.scaled(self.expplayer.size(), aspectRatioMode=True))
                    # 打开.txt文件并读取内容
                    self.exp_inf.clear()
                    self.exp_type.clear()
                    with open(path.split('.')[0] + '.txt', 'r') as file:
                        text = file.read()
                    # 将文件内容写入textEdit_2中
                    self.exp_inf.setText(text)
                    path1 = os.path.join(os.path.dirname(path), os.path.basename(os.path.dirname(path)) + '.txt')
                    # print(os.path.dirname(path) + '.txt')
                    with open(path1, 'r') as file:
                        text = file.read()
                    self.exp_type.setText(text)
                    self.exp_inf.setReadOnly(True)
                    self.exp_type.setReadOnly(True)
                except Exception as e:
                    # 处理错误，例如无效的图像路径
                    print(f"设置图像时发生错误：{e}")

    def tabClicked(self):
        self.exp_list.setIconSize(QSize(75, 75))

        # 清空列表以便重新加载
        self.exp_list.clear()

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
        folder_path = os.path.join(base_path, pathlist[-1])
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
                self.exp_list.addItem(item)

    # 保存行为识别报告
    def stopIdentify(self):
        # QMessageBox.information(self, "提示", "识别报告已保存到文件夹\nD:/VScode/motion-monitor-x/result")
        Globals.detection_run = False
        self.start_identify.setEnabled(True)
        self.stop_identify.setEnabled(True)

    # 刷新视频列表/工作区/终端
    def refreshWidgetlist(self):
        if not self.listcon.isChecked() and not self.workcon.isChecked():
            self.widget_list.setVisible(False)
        else:
            self.widget_list.setVisible(True)

        if self.listcon.isChecked():
            self.original.setVisible(True)
        else:
            self.original.setVisible(False)
        if self.workcon.isChecked():
            self.works.setVisible(True)
        else:
            self.works.setVisible(False)
        if self.terminalcon.isChecked():
            self.controlWidget.setVisible(True)
        else:
            self.controlWidget.setVisible(False)

    # 退出功能
    def Quit(self):
        super().close()

    # 启动自定义标签设置窗口
    def labelSetsSettings(self):
        self.controller.show_label_settings()

    def saveCut(self):
        cut_thread = threading.Thread(target=self.cutThread)
        cut_thread.daemon = True  # 主界面关闭时自动退出此线程
        cut_thread.start()

    # 剪辑完成后的视频加入工作区
    def cutCompleted(self):
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
    def cutThread(self):
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
    def getVideoFiles(self):
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
    def prevVideo(self):
        video_paths = self.getVideoFiles()
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
    def nextVideo(self):
        video_paths = self.getVideoFiles()
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
    def speedChange(self):
        if self.stackedWidget.currentIndex() == 0:
            speed = self.speed.currentText()
            speed = float(speed.split("x")[0])
            self.vdplayer.speed_change(speed)

    def echo(self, low_value, high_value):
        # print(low_value, high_value)
        if self.stackedWidget.currentIndex() == 0:
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
    def worklistShowContextMenu(self, position):
        item = self.work_list.itemAt(position)
        if item is None:
            return
        menu = RoundMenu(self.work_list)
        # 添加新建文件夹的菜单项
        new_folder_action = QAction("重命名", self.work_list)
        new_folder_action.triggered.connect(self.worklist_rename)
        menu.addAction(new_folder_action)

        new_folder_action = QAction("删除", self.work_list)
        new_folder_action.triggered.connect(lambda: self.worklist_delete(item))
        menu.addAction(new_folder_action)
        # 显示菜单
        menu.exec_(self.work_list.mapToGlobal(position))

    # 工作区文件重命名
    def worklist_rename(self):
        item = self.work_list.currentItem()
        if item is None:
            return
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
        if item is False or item is None:
            item = self.work_list.currentItem()
        if item is None:
            return
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
        context_menu = RoundMenu(self.video_tree)

        action = QAction("重命名", context_menu)
        context_menu.addAction(action)
        action.triggered.connect(self.rename_file)

        action = QAction("删除", context_menu)
        context_menu.addAction(action)
        action.triggered.connect(self.delete_file)

        action = QAction("加入工作区", context_menu)
        context_menu.addAction(action)
        action.triggered.connect(self.addWorkspace)

        return context_menu

    # 创建文件夹右键菜单
    def createFolderMenu(self, item):
        context_menu = RoundMenu(self.video_tree)

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
        action.triggered.connect(lambda: self.rename_file)

        return context_menu

    # 文件重命名
    def rename_file(self):
        item = self.work_list.currentItem()
        if item is None:
            return
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

    # 文件删除
    def delete_file(self):
        w = MessageBox(
            title='删除文件',
            content='是否删除文件？此操作不可恢复',
            parent=self
        )
        w.yesButton.setText('确认删除')
        w.cancelButton.setText('取消')
        if w.exec_():
            item = self.video_tree.selectedItems()
            if item is None:
                return
            for i in item:
                file_path = self.getFullPath(i)
                print(file_path)
                index = i.parent().indexOfChild(i)
                if index != -1:
                    i.parent().takeChild(index)
                del i
                # self.video_tree.remove(move_item)
                os.remove(file_path)
        InfoBar.success(
            title='成功！',
            content='文件已删除',
            orient=Qt.Horizontal,
            isClosable=True,
            position=InfoBarPosition.BOTTOM_RIGHT,
            # position='Custom',   # NOTE: use custom info bar manager
            duration=2000,
            parent=self.parent().parent()
        )

    # 新建文件
    def create_file(self, item):
        filename = "new_file.txt"
        current_folder_path = self.getFullPath(item)
        if filename is None:
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
                self.selected_item = cloned_item

    # 启动识别按钮点击
    def startIdentifyClicked(self):
        if self.selected_item is None:
            InfoBar.warning(
                title='警告！',
                content='请先选择一个视频源',
                orient=Qt.Horizontal,
                isClosable=True,
                position=InfoBarPosition.BOTTOM_RIGHT,
                # position='Custom',   # NOTE: use custom info bar manager
                duration=2000,
                parent=self.parent().parent()
            )
            return
        if self.settings_window is None:
            Globals.settings['saved'] = False
            self.settings_window = ModelSettings(self)

            self.settings_window.show()
            # self.setEnabled(False)  # 暂停主窗口活动

    # 识别完成回复函数
    def startIdentifyThread(self):
        if Globals.settings['saved']:
            self.timer_cv.stop()
            Globals.camera_running = True
            identify_thread = threading.Thread(target=self.worker.start_identify, args=(self,))
            identify_thread.daemon = True  # 主界面关闭时自动退出此线程
            identify_thread.start()

    def select_V_D(self, index):
        if index == 1:
            self.v_d_select.setCurrentItem("video")
            # 释放资源
            # if self.capture is not None:
            #     self.capture.release()
            # self.timer_cv.stop()
            self.video_tree.clear()

            # 添加视频列表
            if self.selected_folder != "":
                self.addFolderToTree(self.video_tree, self.selected_folder)
            for action in self.CommandBar.actions():
                self.CommandBar.removeAction(action)

            action = Action(FluentIcon.FOLDER_ADD, "选择", self)
            action.triggered.connect(self.openVideoFolder)
            self.CommandBar.addAction(action)
            # self.CommandBar.addSeparator()
            action = Action(FluentIcon.ADD, "添加工作区", self)
            action.triggered.connect(self.addWorkspace)
            self.CommandBar.addAction(action)
            # self.CommandBar.addSeparator()
            action = Action(FluentIcon.EDIT, "重命名", self)
            action.triggered.connect(self.rename_file)
            self.CommandBar.addAction(action)
            # self.CommandBar.addSeparator()
            action = Action(QIcon(":/gallery/op_icon/delete-file.png"), "删除", self)
            action.triggered.connect(self.delete_file)
            self.CommandBar.addAction(action)
        elif index == 2:
            self.v_d_select.setCurrentItem("camera")
            self.video_tree.clear()

            # 添加设备列表
            if self.camera_id_list is not None:
                for camera_id in self.camera_id_list:
                    item = QTreeWidgetItem(self.video_tree)
                    item.isCamera = True
                    item.setText(0, str(camera_id))
                    item.setIcon(0, self.camera_icon)
            for action in self.CommandBar.actions():
                self.CommandBar.removeAction(action)

            action = Action(FluentIcon.ADD, "添加工作区", self)
            action.triggered.connect(self.addWorkspace)
            self.CommandBar.addAction(action)

        # 初始化检测摄像头线程

    def initialize_camera(self):
        index = 0
        camera_id_list = []
        failed = 0
        while failed < 3:
            cam = cv2.VideoCapture(index)
            if not cam.read()[0]:
                failed += 1
            else:
                camera_id_list.append(index)
            cam.release()
            index += 1
        self.camera_id_list = camera_id_list
        print("检测到设备：" + str(self.camera_id_list))

        selected_option = self.v_d_select.currentItem().text()
        if selected_option == '摄像头':
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
        self.signal.emit()

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

    @staticmethod
    def loadFrameDict(frame_path=None):
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
        thread = threading.Thread(target=self.CameraVideo_thread(item))
        thread.start()

    def CameraVideo_thread(self, item):
        # 显示pdf文件
        path = self.getFullPath(item)
        if path.endswith('.pdf'):
            self.on_add_pdf_tab(path)
            return

        selected_option = self.v_d_select.currentItem().text()
        if selected_option == '本地视频':
            if path.endswith('.avi') or path.endswith('.mp4') or path.endswith('.jpg') or path.endswith('.png'):
                self.addWorkspace()
                self.playSelectedVideo(item, False)
        elif selected_option == '摄像头':
            self.addWorkspace()
            if self.stackedWidget.currentIndex() == 1:
                self.capture = cv2.VideoCapture(int(item.text(0)))
                # 初始化摄像头捕捉对象
                self.timer_cv.start()
                # 启动定时器
            elif self.stackedWidget.currentIndex() == 0:
                if self.capture is not None:
                    # 释放视频流捕获
                    self.capture.release()
                    self.timer_cv.stop()
                player = self.Vdplayers[self.thisvdwidget.id]
                player.setMedia(item.text(0))
                player.play()

    # 预览工作区列表项
    def WorkListPreview(self, item):
        thread = threading.Thread(target=self.WorkListPreview_thread(item))
        thread.start()

    def WorkListPreview_thread(self, item):
        self.selected_item = item
        self.start_identify.setEnabled(True)
        self.start_identify.setEnabled(True)
        if item.isCamera:
            # 如果项目是摄像头
            if self.stackedWidget.currentIndex() == 1:
                self.capture = cv2.VideoCapture(int(item.text(0)))
                # 初始化摄像头捕捉对象
                self.timer_cv.start()
                # 启动定时器
            elif self.stackedWidget.currentIndex() == 0:
                player = self.Vdplayers[self.thisvdwidget.id]
                player.setMedia(item.text(0))
                player.play()
        else:
            if self.capture is not None:
                # 释放视频流捕获
                self.capture.release()
                self.timer_cv.stop()
            # 播放选定的视频，带声音
            self.playSelectedVideo(item, True)

    # 从OpenCV捕获摄像头获取一帧图像
    def updateFrame(self):
        flag, image = self.capture.read()
        if flag:
            show = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # 将图像转换为QImage对象

            showImage = QImage(show.data, show.shape[1], show.shape[0], show.shape[1] * 3, QImage.Format_RGB888)

            # 根据当前选中的选项卡索引调整标签
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
        # if self.capture is not None:
        #     self.capture.release()
        # self.timer_cv.stop()
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

        if self.stackedWidget.currentIndex() == 0:
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
            player = self.Vdplayers[self.thisvdwidget.id]
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

        elif self.stackedWidget.currentIndex() == 1:
            video_path = selected_video_path
            if file_extension != '':
                self.idplayer.setMedia(video_path)
                if self.idplayer.player.isOpened():
                    # 读取第一帧
                    flag, image = self.idplayer.player.read()
                    if flag:
                        show = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        showImage = QImage(show.data, show.shape[1], show.shape[0], show.shape[1] * 3,
                                           QImage.Format_RGB888)
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
            total_frames, frame_rate, hours, minutes, seconds, file_size, formatted_date, width, height = get_video_info(
                selected_video_path)
            # 将信息设置到UI元素中
            info_text = f"总帧数: {total_frames}\n\n帧率: {frame_rate}\n\n时长: {hours}:{minutes}:{seconds}\n\n"
            self.video_time_all.setText(f"{minutes}:{seconds}")
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

    # 暂停/播放
    def playPause(self):
        if self.stackedWidget.currentIndex() == 0:
            if self.vdplayer.state() == 1:
                self.play_pause.setIcon(self.play_ico)
                self.vdplayer.pause()
            else:
                self.play_pause.setIcon(self.pause_ico)
                self.vdplayer.play()
        elif self.stackedWidget.currentIndex() == 1:
            if self.idplayer.state() == 1:
                self.play_pause_2.setIcon(self.play_ico)
                self.idplayer.pause()
            else:
                self.play_pause_2.setIcon(self.pause_ico)
                self.speed_play(self.vdplayer)

    # 保存标签
    def saveLabel(self):
        # self.gridLayout_25.setParent(None)
        # self.gridLayout_25.showFullScreen()
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
    def searchAction(self):
        if not Globals.dict_text:
            return
        self.frame_dict = self.loadFrameDict(Globals.dict_text)
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

    def saveChart(self):
        # 保存
        self.ax.figure.savefig('result/LineChart.png')
        self.ax2.figure.savefig('result/PieChart.png')

    # 绘制折线图
    # 绘制折线图
    def drawLineChart(self):
        # Globals.dict_text = {
        #     1.0: {1: '汽车 ', 2: '人 ', 3: '人 ', 4: '汽车 ', 5: '人 ', 7: '人 '},
        #     2.0: {1: '汽车 ', 2: '人 携带/握住物体', 4: '汽车 ', 7: '人 走路', 9: '人 ', 10: '摩托车 '},
        #     3.0: {1: '汽车 ', 2: '人 携带/握住物体', 4: '汽车 ', 7: '人 骑车（例如，自行车、汽车、马）', 11: '摩托车 '},
        #     4.0: {1: '汽车 ', 2: '人 ', 3: '人 ', 4: '汽车 ', 5: '人 ', 7: '人 '},
        #     5.0: {1: '汽车 ', 2: '人 携带/握住物体', 4: '汽车 ', 7: '人 走路', 9: '人 ', 10: '摩托车 '},
        #     6.0: {1: '汽车 ', 2: '人 携带/握住物体', 4: '汽车 ', 7: '人 骑车（例如，自行车、汽车、马）', 11: '摩托车 '},
        #     7.0: {1: '汽车 ', 2: '人 ', 3: '人 ', 4: '汽车 ', 5: '人 ', 7: '人 '},
        #     8.0: {1: '汽车 ', 2: '人 携带/握住物体', 4: '汽车 ', 7: '人 走路', 9: '人 ', 10: '摩托车 '},
        #     9.0: {1: '汽车 ', 2: '人 携带/握住物体', 4: '汽车 ', 7: '人 骑车（例如，自行车、汽车、马）', 11: '摩托车 '},
        #     10.0: {1: '汽车 ', 2: '人 ', 3: '人 ', 4: '汽车 ', 5: '人 ', 7: '人 '},
        #     11.0: {1: '汽车 ', 2: '人 携带/握住物体', 4: '汽车 ', 7: '人 走路', 9: '人 ', 10: '摩托车 '},
        #     12.0: {1: '汽车 ', 2: '人 携带/握住物体', 4: '汽车 ', 7: '人 骑车（例如，自行车、汽车、马）', 11: '摩托车 '}
        # }
        # 如果全局变量 Globals.dict_text 为空，则返回
        if not Globals.dict_text:
            return
        # 从 Globals.dict_text 加载数据到 self.frame_dict
        self.frame_dict = self.loadFrameDict(Globals.dict_text)

        # 如果 self.frame_dict 为空，则返回
        if not self.frame_dict:
            return
        self._drawLineChart(False)
        self._drawLineChart(True)

    def _drawLineChart(self, isflow=False):
        if qconfig.theme == Theme.LIGHT:
            style = 'light'
        else:
            style = 'dark'
        with plt.style.context(matplotx.styles.pitaya_smoothie[style]):
            plt.rcParams['font.family'] = 'Microsoft YaHei'
            plt.tight_layout()
            frame_dict = {}
            if isflow:
                ax = self.ax3
                canvas = self.canvas3
                frame_dict = self.frame_dict
                # 遍历frame_dict字典
                for t, action_dict in frame_dict.items():
                    for order, action in action_dict.items():
                        if "人" in action:
                            frame_dict[t][order] = "人流量"
                        elif "车" in action:
                            frame_dict[t][order] = "车流量"

            else:
                ax = self.ax
                canvas = self.canvas
                frame_dict = self.frame_dict
            # 清除之前的绘图
            ax.clear()
            # 创建一个字典来存储每个唯一动作的计数
            action_counts = {}
            # 遍历 frame_dict 并计算每个唯一动作的发生次数
            for t, action_dict in frame_dict.items():
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
                ax.plot(last_10_data['t'], last_10_data['count'], marker='o', label=action)
                if action == "车流量":
                    self.carflow = last_10_data['count'][-1]
                    self.toolbar3.setText(f"车流量：{self.carflow} 辆/秒     人流量：{self.menflow} 人/秒")
                elif action == "人流量":
                    self.menflow = last_10_data['count'][-1]
                    self.toolbar3.setText(f"车流量：{self.carflow} 辆/秒     人流量：{self.menflow} 人/秒")

            # 设置标签和标题
            ax.set_xlabel('时间')
            ax.set_ylabel('数目')
            if isflow:
                ax.set_title('流量图')
            else:
                ax.set_title('每秒动作数目趋势')

            # 添加图例
            ax.legend()
            # 重新绘制画布
            canvas.draw()

    # 绘制饼图
    def drawPieChart(self):
        if qconfig.theme == Theme.LIGHT:
            style = 'light'
        else:
            style = 'dark'
        with plt.style.context(matplotx.styles.pitaya_smoothie[style]):
            plt.rcParams['font.family'] = 'Microsoft YaHei'
            plt.tight_layout()
            # 如果全局变量 Globals.dict_text 为空，则返回
            if not Globals.dict_text:
                return

            # 从 Globals.dict_text 加载数据到 self.frame_dict
            self.frame_dict = self.loadFrameDict(Globals.dict_text)

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
            self.ax2.set_title('动作分布')

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
            self.timer_cv.stop()

    @staticmethod
    def displayUseInformation():
        # 显示软件使用信息
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

    @staticmethod
    def displayVersionInformation():
        # 显示软件版本信息
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

    @staticmethod
    def modelView():
        # 显示模型简介
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

    @staticmethod
    def changePath():
        # 改变默认视频文件、保存路径
        try:
            with open('Default settings.txt', 'r', encoding='utf-8') as f:
                content = f.read()
            seted = loads(content)
            video_path = seted['video_path']
            save_path = seted['save_path']

        except FileNotFoundError:
            print("文件不存在")

    @staticmethod
    def on_item_changed(item, old_name, current_folder_path):
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


if __name__ == "__main__":
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
    app = QApplication([])
    # app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    user = User_management()
    user.show()
    app.exec()
