import json
import os
import subprocess
import sys
import threading
import qdarkstyle
import cv2
# import ffmpeg
from datetime import datetime

import pygame.camera
from PyQt5.QtCore import QUrl, Qt, QTimer, QThread
from PyQt5.QtGui import QIcon, QImage, QPixmap, QStandardItem
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtWidgets import QFileDialog, QApplication, QListWidgetItem, QSplitter, QTreeWidgetItem, QHeaderView, \
    QListView, QTreeWidget, QMainWindow, QMenu, QAction
from PyQt5 import uic, QtWidgets
from qdarkstyle import LightPalette

import detect
import detect_yolov5
from model_settings import ModelSettings
from utils.myutil import file_is_pic, Globals


class Pyqt5Window(QMainWindow):
    def __init__(self):
        # 加载designer设计的ui程序
        super().__init__()
        self.ui = uic.loadUi('pyqt5.ui')
        self.ui.resize(1000, 600)
        self.ui.showMaximized()
        self.ui.setWindowTitle("视频检测")
        # 播放器
        self.player = QMediaPlayer()
        self.player.setVideoOutput(self.ui.player)
        self.player_2 = QMediaPlayer()
        self.player_2.setVideoOutput(self.ui.player_2)
        # 选择文件夹
        self.ui.video_select.triggered.connect(self.openVideoFolder)
        self.ui.new_file.clicked.connect(self.openVideoFolder)
        # 加入工作区
        self.ui.add_workspace.clicked.connect(self.addWorkspace)
        # 暂停
        self.ui.play_pause.clicked.connect(self.playPause)
        self.ui.play_pause_2.clicked.connect(self.playPause)
        # 双击播放
        self.ui.video_tree.setSortingEnabled(False)
        self.ui.video_tree.itemDoubleClicked.connect(self.CameraVideo)
        self.ui.work_list.itemDoubleClicked.connect(self.WorkListPreview)
        # 文件树展开
        self.ui.video_tree.itemExpanded.connect(self.loadSubtree)
        # 右键菜单
        self.ui.video_tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.ui.video_tree.customContextMenuRequested.connect(self.showContextMenu)
        # 进度条
        self.player.durationChanged.connect(self.getDuration)
        self.player.positionChanged.connect(self.getPosition)
        self.player_2.durationChanged.connect(self.getDuration)
        self.player_2.positionChanged.connect(self.getPosition)
        self.ui.video_slider.sliderMoved.connect(self.updatePosition)
        self.ui.video_slider_2.sliderMoved.connect(self.updatePosition)
        # 保存标签
        self.ui.save_label.clicked.connect(self.saveLabel)
        # 动作列表
        self.frame_dict = {}
        # 搜索动作
        self.ui.item_search_button.clicked.connect(self.search_action)
        # 获取摄像头id列表
        self.camera_id_list = None
        camera_thread = threading.Thread(target=self.initialize_camera)
        camera_thread.daemon = True  # 主界面关闭时自动退出此线程
        camera_thread.start()
        self.capture = None
        # 视频/摄像头
        self.ui.v_d_comboBox.currentIndexChanged.connect(self.select_V_D)
        self.ui.v_d_comboBox.setView(QListView())
        # 启动识别
        self.ui.start_identify.clicked.connect(self.startIdentifyClicked)
        self.settings_window = None
        # 使用样式表来设置项的高度
        self.ui.v_d_comboBox.setStyleSheet('QComboBox QAbstractItemView::item { height: 20px; }')
        self.ui.v_d_comboBox.setMaxVisibleItems(50)
        self.ui.camera.setVisible(False)
        self.ui.play_pause_2.setVisible(False)
        # tab切换
        self.ui.tabWidget.currentChanged.connect(self.tabChanged)
        # 定时器
        self.timer_cv = QTimer()
        self.timer_cv.timeout.connect(self.updateFrame)
        self.timer_cv.setInterval(30)  # 1000毫秒 = 1秒
        # 启用多选
        self.ui.video_tree.setSelectionMode(QTreeWidget.ExtendedSelection)
        self.ui.video_tree.setStyleSheet("""
        QTreeWidget::branch:has-siblings:!adjoins-item{ \
            border-image:None 0;\
        }\
        QTreeWidget::branch:has-siblings:adjoins-item{\
            border-image:None 0;\
        }   \
        QTreeWidget::branch:!has-children:!has-siblings:adjoins-item{\
            border-image:None 0;\
        }\
        """)
        self.ui.work_list.setSelectionMode(QTreeWidget.ExtendedSelection)

        self.video_tree = []
        self.selected_folder = ""
        self.selected_path = ""
        self.selected_item = None

        # 分割器
        splitter_list = QSplitter(Qt.Vertical)
        splitter_list.addWidget(self.ui.video_tree)
        splitter_list.addWidget(self.ui.work_list)
        self.ui.verticalLayout.addWidget(splitter_list)
        self.ui.verticalLayout.setStretch(0, 1)  # 第一个部件的伸缩因子为1
        self.ui.verticalLayout.setStretch(1, 40)  # 第二个部件的伸缩因子为2
        self.ui.verticalLayout.setStretch(2, 40)  # 第三个部件的伸缩因子为3

        splitter_tab = QSplitter(Qt.Horizontal)
        splitter_tab.addWidget(self.ui.widget_list)
        splitter_tab.addWidget(self.ui.tabWidget)
        splitter_tab.setStretchFactor(0, 8)
        splitter_tab.setStretchFactor(1, 10)
        splitter_tab.setStyleSheet("""
            QSplitter {
                background-color: white;
            }
            QSplitter::handle {
                background-color: white;
            }
        """)

        self.ui.horizontalLayout.addWidget(splitter_tab)

        splitter_video = QSplitter(Qt.Horizontal)
        splitter_video.addWidget(self.ui.video_widget)
        splitter_video.addWidget(self.ui.video_label_widget)
        splitter_video.setStretchFactor(0, 5)
        splitter_video.setStretchFactor(1, 2)
        self.ui.horizontalLayout_5.addWidget(splitter_video)

        splitter_video = QSplitter(Qt.Horizontal)
        splitter_video.addWidget(self.ui.video_widget_2)
        splitter_video.addWidget(self.ui.video_label_widget_2)
        splitter_video.setStretchFactor(0, 9)
        splitter_video.setStretchFactor(1, 1)
        self.ui.horizontalLayout_7.addWidget(splitter_video)
        # 图标
        self.file_icon = QIcon("resources/file_ico.png")
        self.folder_icon = QIcon("resources/folder_ico.ico")
        self.image_icon = QIcon("resources/img_ico.ico")
        self.text_icon = QIcon("resources/text_ico.png")
        self.video_icon = QIcon("resources/video_ico.png")
        self.camera_icon = QIcon("resources/cam_ico.png")
        self.play_ico = QIcon("resources/play_ico.png")
        self.pause_ico = QIcon("resources/pause_ico.png")

    def showContextMenu(self, position):

        item = self.ui.video_tree.itemAt(position)
        if item is not None:
            context_menu = QMenu(self.ui.video_tree)

            context_menu.setStyleSheet("background-color: white")

            action = QAction("右键菜单项", context_menu)
            context_menu.addAction(action)
            action.triggered.connect(lambda: self.onContextMenuClick(item.text(0)))

            action = QAction("右键菜单项", context_menu)
            context_menu.addAction(action)
            action.triggered.connect(lambda: self.onContextMenuClick(item.text(0)))

            action = QAction("右键菜单项", context_menu)
            context_menu.addAction(action)
            action.triggered.connect(lambda: self.onContextMenuClick(item.text(0)))

            context_menu.exec_(self.ui.video_tree.mapToGlobal(position))

    def onContextMenuClick(self, item_text):
        print(f"右键菜单项被点击: {item_text}")

    def addWorkspace(self):
        selected_items = self.ui.video_tree.selectedItems()
        for item in selected_items:
            # 复制选中的项到目标 TreeWidget
            if item.childCount() <= 0:
                cloned_item = item.clone()
                cloned_item.isCamera = item.isCamera
                if not item.isCamera:
                    cloned_item.path = self.getFullPath(item)
                else:
                    cloned_item.device = int(item.text(0))
                self.ui.work_list.addTopLevelItem(cloned_item)

    def startIdentifyClicked(self):
        if self.settings_window is None:
            Globals.settings['saved'] = False
            self.settings_window = ModelSettings(self)
            self.settings_window.ui.show()
            self.ui.setEnabled(False)  # 暂停主窗口活动

    def startIdentifyThread(self):
        if Globals.settings['saved']:
            self.timer_cv.stop()
            Globals.camera_running = True
            identify_thread = threading.Thread(target=self.startIdentify)
            identify_thread.daemon = True  # 主界面关闭时自动退出此线程
            identify_thread.start()

    def startIdentify(self):
        if self.selected_item.isCamera:
            if Globals.settings['model_select'] == 'yolov5':
                detect_yolov5.run(source=self.selected_item.device, weights=Globals.settings['pt_path'],
                                  show_label=self.ui.camera_2,
                                  save_img=False, use_camera=True, show_labellist=self.ui.action_list)
            elif Globals.settings['model_select'] == 'yolo_slowfast':
                detect.run(source=self.selected_item.device, weights=Globals.settings['pt_path'],
                           show_label=self.ui.camera_2,
                           save_img=False, use_camera=True, show_labellist=self.ui.action_list)
        else:
            if Globals.settings['model_select'] == 'yolov5':
                detect_yolov5.run(source=self.selected_item.path, weights=Globals.settings['pt_path'],
                                  show_label=self.ui.camera_2,
                                  save_img=False, use_camera=True, show_labellist=self.ui.action_list)
            elif Globals.settings['model_select'] == 'yolo_slowfast':
                detect.run(source=self.selected_item.path, weights=Globals.settings['pt_path'],
                           show_label=self.ui.camera_2,
                           save_img=False, use_camera=True, show_labellist=self.ui.action_list)
        # detect.run(source=self.selected_path, weights=model_path, show_label=self.ui.camera_2,
        # save_img=True, show_labellist=self.ui.action_list)

    # 视频/设备切换时触发
    def select_V_D(self):
        selected_option = self.ui.v_d_comboBox.currentText()
        if selected_option == '视频列表':
            self.ui.player.setVisible(True)
            self.ui.camera.setVisible(False)
            if self.capture is not None:
                self.capture.release()
            self.timer_cv.stop()
            self.ui.video_tree.clear()
            if self.selected_folder != "":
                self.addFolderToTree(self.ui.video_tree, self.selected_folder)
        elif selected_option == '设备列表':
            self.ui.player.setVisible(False)
            self.ui.camera.setVisible(True)
            self.ui.video_tree.clear()
            if self.camera_id_list is not None:
                for camera_id in self.camera_id_list:
                    item = QTreeWidgetItem(self.ui.video_tree)
                    item.isCamera = True
                    item.setText(0, str(camera_id))
                    item.setIcon(0, self.camera_icon)

    # 初始化检测摄像头线程
    def initialize_camera(self):
        pygame.camera.init()
        self.camera_id_list = pygame.camera.list_cameras()
        selected_option = self.ui.v_d_comboBox.currentText()
        if selected_option == '设备列表':
            self.ui.player.setVisible(False)
            self.ui.camera.setVisible(True)
            self.ui.video_tree.clear()
            if self.camera_id_list is not None:
                for camera_id in self.camera_id_list:
                    item = QTreeWidgetItem(self.ui.video_tree)
                    item.isCamera = True
                    item.setText(0, str(camera_id))
                    item.setIcon(0, self.camera_icon)

    # 文件树展开时调用
    def loadSubtree(self, item):
        self.ui.video_tree.resizeColumnToContents(0)
        selected_video_path = self.getFullPath(item)
        item.setData(0, Qt.UserRole, True)
        self._addFilesToTree(item, selected_video_path, 0)

    def openVideoFolder(self):
        # 选择文件夹
        folder_path = QFileDialog.getExistingDirectory()
        if folder_path:
            self.selected_folder = folder_path
            self.ui.video_tree.clear()
            self.addFolderToTree(self.ui.video_tree, folder_path)

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
        if not frame_path:
            return {
                2: [("talk", 0.9), ("stand", 0.8)],
                4: [("talk", 0.7), ("smoke", 0.9)]
            }
        else:
            frame_dict = {}
            for k in frame_path:
                frame_dict[float(k)] = frame_path[k]
            return frame_dict

    def CameraVideo(self, item):
        selected_option = self.ui.v_d_comboBox.currentText()
        if selected_option == '视频列表':
            self.playSelectedVideo(item, False)

            self.ui.player.setVisible(True)
            self.ui.camera.setVisible(False)
        elif selected_option == '设备列表':
            self.ui.player.setVisible(False)
            self.ui.camera.setVisible(True)

            self.capture = cv2.VideoCapture(int(item.text(0)))
            self.timer_cv.start()

    def WorkListPreview(self, item):
        self.selected_item = item
        self.ui.start_identify.setEnabled(True)
        if item.isCamera:
            self.ui.player.setVisible(False)
            self.ui.camera.setVisible(True)
            self.capture = cv2.VideoCapture(0)
            self.timer_cv.start()
        else:
            self.playSelectedVideo(item, True)
            self.ui.player.setVisible(True)
            self.ui.camera.setVisible(False)

    def updateFrame(self):
        # 从OpenCV捕获摄像头获取一帧图像
        flag, image = self.capture.read()
        show = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)
        label = self.ui.camera
        if self.ui.tabWidget.currentIndex() == 0:
            label = self.ui.camera
        elif self.ui.tabWidget.currentIndex() == 1:
            label = self.ui.camera_2
        label_size = label.size()
        label_size.setWidth(label_size.width() - 10)
        label_size.setHeight(label_size.height() - 10)
        scaled_image = showImage.scaled(label_size, Qt.KeepAspectRatio)
        pixmap = QPixmap.fromImage(scaled_image)
        label.setPixmap(pixmap)
        label.setAlignment(Qt.AlignCenter)

    def playSelectedVideo(self, item, isworklist):
        if self.capture is not None:
            self.capture.release()
        self.timer_cv.stop()
        # 重置标签
        self.ui.video_label_edit.setText("")
        # 解禁设置视频标签
        self.ui.video_label_edit.setEnabled(True)
        self.ui.save_label.setEnabled(True)
        # 获取所选项的完整路径，包括文件夹结构
        if isworklist:
            selected_video_path = item.path
        else:
            selected_video_path = self.getFullPath(item)

        self.selected_path = selected_video_path
        file_extension = os.path.splitext(selected_video_path)[1]

        # 播放或禁用播放
        player = self.player
        play_pause = self.ui.play_pause
        if self.ui.tabWidget.currentIndex() == 0:
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
                    self.ui.video_label_edit.setText(file_content)
                # 加载动作列表
                # new_file_name = os.path.splitext(file_name)[0] + ".json"
                # new_path = os.path.join(directory, new_file_name)
                # if os.path.exists(new_path):
                #     self.frame_dict = self.load_frame_dict(new_path)
            else:
                print("err")
            player = self.player
            play_pause = self.ui.play_pause
            if os.path.isfile(selected_video_path) and file_extension:
                player.setMedia(QMediaContent(QUrl.fromLocalFile(selected_video_path)))
                # 根据文件后缀选择不同处理方式
                if file_extension == ".avi" or file_extension == ".mp4":
                    play_pause.setEnabled(True)
                    player.play()
                elif file_extension == ".jpg" or file_extension == ".png":
                    play_pause.setEnabled(False)
                    player.play()
                else:
                    play_pause.setEnabled(False)
                self.getVideoinfo(selected_video_path)
        elif self.ui.tabWidget.currentIndex() == 1:
            video_path = selected_video_path
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                # 读取第一帧
                flag, image = cap.read()
                if flag:
                    show = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)
                    label_size = self.ui.camera_2.size()
                    label_size.setWidth(label_size.width() - 10)
                    label_size.setHeight(label_size.height() - 10)
                    scaled_image = showImage.scaled(label_size, Qt.KeepAspectRatio)
                    pixmap = QPixmap.fromImage(scaled_image)
                    self.ui.camera_2.setPixmap(pixmap)
                    self.ui.camera_2.setAlignment(Qt.AlignCenter)
            cap.release()
            player = self.player_2
            play_pause = self.ui.play_pause_2

    def getFullPath(self, item):
        # 从所选项递归构建完整路径
        path_components = [item.text(0)]
        while item.parent() is not None and item.parent().parent() is not None:
            item = item.parent()
            path_components.insert(0, item.text(0))
        # 将路径组件连接起来
        full_path = os.path.join(self.selected_folder, *path_components)
        return full_path

    # 视频总时长获取
    def getDuration(self, d):
        if self.ui.tabWidget.currentIndex() == 0:
            self.ui.video_slider.setRange(0, d)
            self.ui.video_slider.setEnabled(True)
        elif self.ui.tabWidget.currentIndex() == 1:
            self.ui.video_slider_2.setRange(0, d)
            self.ui.video_slider_2.setEnabled(True)
        self.displayTime(d)

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
            info_text += f"修改日期:{formatted_date} \n\n文件大小: {file_size}\n\n分辨率: {width}x{height}"
            self.ui.video_info.setPlainText(info_text)

        except Exception as e:
            # 处理异常情况
            self.ui.video_info.setPlainText(f"获取视频信息时发生错误: {str(e)}")

    # 单位转换
    def convert_bytes_to_readable(self, size_in_bytes):
        units = ["B", "KB", "MB", "GB", "TB"]
        unit_index = 0
        size = size_in_bytes
        while size >= 1024 and unit_index < len(units) - 1:
            size /= 1024
            unit_index += 1
        formatted_size = "{:.2f}".format(size)
        return f"{formatted_size} {units[unit_index]}"

    # 视频实时位置获取
    def getPosition(self):
        if self.ui.tabWidget.currentIndex() == 0:
            p = self.player.position()
            self.ui.video_slider.setValue(p)
            self.displayTime(p)
        elif self.ui.tabWidget.currentIndex() == 1:
            p = self.player_2.position()
            self.ui.video_slider_2.setValue(p)
            self.displayTime(p)

    # 显示剩余时间
    def displayTime(self, ms):
        print(ms)
        minutes = int(ms / 60000)
        seconds = int((ms % 60000) / 1000)
        milliseconds = int(ms % 1000)
        if self.ui.tabWidget.currentIndex() == 0:
            self.ui.video_time.setText('{}:{}'.format(minutes, seconds))
        elif self.ui.tabWidget.currentIndex() == 1:
            self.ui.video_time_2.setText('{}:{}'.format(minutes, seconds))
            self.ui.action_list.clear()
            # 启动输出动作标签
            # for t, action_list in self.frame_dict.items():
            #     if t == seconds:
            #         # print(action_list[action])
            #         # item = QListWidgetItem(video_file)
            #         for action in action_list:
            #             self.ui.action_list.addItem(f"时间：{t} 动作：{action}-{action_list[action]}")

    # 用进度条更新视频位置
    def updatePosition(self, v):
        if self.ui.tabWidget.currentIndex() == 0:
            self.displayTime(self.ui.video_slider.maximum() - v)
            self.player.setPosition(v)
        elif self.ui.tabWidget.currentIndex() == 1:
            self.displayTime(self.ui.video_slider_2.maximum() - v)
            self.player_2.setPosition(v)

    # 暂停/播放
    def playPause(self):
        if self.ui.tabWidget.currentIndex() == 0:
            if self.player.state() == 1:
                self.ui.play_pause.setIcon(self.play_ico)
                self.player.pause()
            else:
                self.ui.play_pause.setIcon(self.pause_ico)
                self.player.play()
        elif self.ui.tabWidget.currentIndex() == 1:
            if self.player_2.state() == 1:
                self.ui.play_pause_2.setIcon(self.play_ico)
                self.player_2.pause()
            else:
                self.ui.play_pause_2.setIcon(self.pause_ico)
                self.player_2.play()

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
            content_to_write = self.ui.video_label_edit.toPlainText()
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
        item = self.ui.item_search.text()
        self.ui.search_result.clear()
        if item:
            for t, action_list in self.frame_dict.items():
                for action in action_list:
                    if item in action_list[action]:
                        # print(action_list[action])
                        # item = QListWidgetItem(video_file)
                        self.ui.search_result.addItem(f"时间：{t} 动作：{action}-{action_list[action]}")
            # self.ui.item_search.setText("")

    def tabChanged(self):
        self.player.pause()
        self.timer_cv.stop()
        if self.capture is not None:
            self.capture.release()


if __name__ == "__main__":
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
    app = QApplication([])
    app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyqt5', palette=LightPalette()))
    pyqt5 = Pyqt5Window()
    pyqt5.ui.show()
    app.exec()