import sys

import cv2
from PyQt5 import uic
from PyQt5.QtCore import pyqtSlot, Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QProgressBar, QApplication, QLabel, QWidget


class Video(QWidget):
    def __init__(self):
        super(Video, self).__init__()
        # self.ui.pushButton.clicked.connect(self.play)
        # self.ui.pushButton_2.clicked.connect(self.play_pause)
        self.player = cv2.VideoCapture(0)
        # self.ui.horizontalSlider.valueChanged.connect(self.set_frame)
        self.video_path = ""
        self.output = None
        self.fps = self.player.get(cv2.CAP_PROP_FPS)
        self.playing = False  # 用于表示当前播放状态
        # self.update_interval = 100  # 更新进度条的时间间隔，单位毫秒

        # # 设置定时器，定期更新进度条
        # self.timer = QTimer(self)
        # self.timer.timeout.connect(self.update_progress_bar)
        # self.ui.comboBox.currentIndexChanged.connect(self.speed_change)

    def state(self):
        if self.playing:
            return 1
        else:
            return 0

    def setVideoOutput(self, output):
        self.output = output

    def speed_change(self, speed):
        # 改变帧率
        self.fps = int(self.player.get(cv2.CAP_PROP_FPS) * float(speed))

    def play_pause(self):
        # 切换播放状态
        self.playing = not self.playing

    def update_progress_bar(self):
        # 获取当前帧数
        current_frame = int(self.player.get(cv2.CAP_PROP_POS_FRAMES))

        # 获取总帧数
        total_frames = int(self.player.get(cv2.CAP_PROP_FRAME_COUNT))

        # 计算进度百分比
        progress_percentage = int((current_frame / total_frames) * 100)

        # 更新进度条的值
        self.ui.horizontalSlider.setValue(progress_percentage)

    def set_frame(self, video_slider):
        # 获取滑块的比例值（0到1之间）
        slider_value = video_slider.value() / video_slider.maximum()

        # 获取总帧数
        total_frames = int(self.player.get(cv2.CAP_PROP_FRAME_COUNT))

        # 计算对应的帧数
        frame_number = int(slider_value * total_frames)

        # 设置视频播放器的当前帧
        self.player.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    def play(self):
        self.playing = True
        while True:
            if self.playing:
                ret, frame = self.player.read()
                if ret:
                    self.show_frame(frame)
                else:
                    break

            k = cv2.waitKey(int(1000 / self.fps))
            if k == Qt.Key_Space:
                self.play_pause()


    def setMedia(self, video_path=None):
        if video_path:
            self.video_path = video_path
        self.player = cv2.VideoCapture(self.video_path)
        self.fps = self.player.get(cv2.CAP_PROP_FPS)

    def show_frame(self, frame):
        # 将BGR图像转换为RGB格式
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 将图像编码为二进制格式
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # 调整图像大小以适应 QLabel
        qt_image = qt_image.scaled(self.output.width(), self.output.height(), Qt.KeepAspectRatio)

        # 创建 QPixmap，并在 QLabel 中居中显示图像
        pixmap = QPixmap.fromImage(qt_image)
        self.output.setPixmap(pixmap)

        # 设置 QLabel 的对齐方式为居中
        self.output.setAlignment(Qt.AlignCenter)

