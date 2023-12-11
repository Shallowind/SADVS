import sys

import cv2
from PyQt5 import uic
from PyQt5.QtCore import pyqtSlot, Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QProgressBar, QApplication, QLabel, QWidget


class Video(QWidget):
    def __init__(self):
        super(Video, self).__init__()
        self.player = None
        self.video_slider = None
        self.video_path = ""
        self.output = None
        self.fps = 25
        self.playing = False  # 用于表示当前播放状态
        self.update_interval = 100  # 更新进度条的时间间隔，单位毫秒
        self.video_time = None
        self.cut_time = None
        # 设置定时器，定期更新进度条
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_progress_bar)
        # self.ui.comboBox.currentIndexChanged.connect(self.speed_change)

    def setPosition(self, position):
        self.video_slider.setValue(position)

        # 获取当前播放的时间（毫秒）
        current_time = self.player.get(cv2.CAP_PROP_POS_MSEC)
        self.displayTime(current_time)


    def displayTime(self, ms):
        # print(ms)
        minutes = int(ms / 60000)
        seconds = int((ms % 60000) / 1000)
        self.video_time.setText('{}:{}'.format(minutes, seconds))
        if self.cut_time is not None:
            self.cut_time.setText('{}:{}'.format(minutes, seconds))

    def state(self):
        if self.playing:
            return 1
        else:
            return 0

    def setVideoOutput(self, output, video_slider, video_time, cut_time=None):
        self.output = output
        self.video_slider = video_slider
        self.video_slider.valueChanged.connect(self.set_frame)
        self.video_time = video_time
        self.cut_time = cut_time

    def speed_change(self, speed):
        # 改变帧率
        self.fps = int(self.player.get(cv2.CAP_PROP_FPS) * float(speed))

    def pause(self):
        # 切换播放状态
        self.timer.stop()
        self.playing = False

    def update_progress_bar(self):
        # 获取当前帧数
        current_frame = int(self.player.get(cv2.CAP_PROP_POS_FRAMES))

        # 获取总帧数
        total_frames = int(self.player.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            return
        # 计算进度百分比
        progress_percentage = int((current_frame / total_frames) * 100)

        # 更新进度条的值
        try:
            self.video_slider.setValue(progress_percentage)
        except:
            pass

    def set_frame(self):
        # 获取滑块的比例值（0到1之间）
        slider_value = self.video_slider.value() / self.video_slider.maximum()

        # 获取总帧数
        total_frames = int(self.player.get(cv2.CAP_PROP_FRAME_COUNT))

        # 计算对应的帧数
        frame_number = int(slider_value * total_frames)

        # 设置视频播放器的当前帧
        self.player.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.player.read()
        if ret:
            self.show_frame(frame)
        else:
            return

    def play(self):
        self.timer.start(self.update_interval)
        self.playing = True
        # 图片文件
        if self.is_image_file(self.video_path):
            img = cv2.imread(self.video_path)
            self.player = cv2.VideoCapture(self.video_path)
            self.show_frame(img)
        # 视频文件
        elif self.is_video_file(self.video_path):
            while self.playing:
                ret, frame = self.player.read()
                if ret:
                    self.show_frame(frame)
                else:
                    break

                # 获取当前播放的时间（毫秒）
                current_time = self.player.get(cv2.CAP_PROP_POS_MSEC)
                self.displayTime(current_time)

                cv2.waitKey(int(1000 / self.fps))

    @staticmethod
    def is_video_file(file_path):
        video_extensions = ['mp4', 'avi', 'mkv', 'mov', 'flv']  # 常见视频文件扩展名

        # 检查文件扩展名
        file_ext = file_path.lower().split('.')[-1]
        if file_ext in video_extensions:
            return True

        # 尝试打开文件
        cap = cv2.VideoCapture(file_path)

        # 检查视频是否成功打开
        if not cap.isOpened():
            print(f"Error: Could not open file at {file_path}")
            return False

        # 释放资源
        cap.release()

        return True

    @staticmethod
    def is_image_file(file_path):
        image_extensions = ['jpg', 'jpeg', 'png', 'gif', 'bmp']  # 常见图像文件扩展名

        # 检查文件扩展名
        file_ext = file_path.lower().split('.')[-1]
        if file_ext in image_extensions:
            return True

        return False

    def setMedia(self, video_path=None):
        if video_path:
            self.video_path = video_path
        self.player = cv2.VideoCapture(self.video_path)
        self.fps = self.player.get(cv2.CAP_PROP_FPS)

    def show_frame(self, frame):
        show = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 将图像转换为QImage对象
        showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)
        # # 缩小尺寸并保持宽高比
        # label_size = self.output.size()
        # label_size.setWidth(label_size.width() - 10)
        # label_size.setHeight(label_size.height() - 10)
        # scaled_image = showImage.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        #
        # pixmap = QPixmap.fromImage(scaled_image)
        # self.output.setPixmap(pixmap)
        # self.output.setAlignment(Qt.AlignCenter)

        scale_factor = min(self.output[1].width() / showImage.width(),
                           self.output[1].height() / showImage.height())

        # 计算新的宽度和高度
        new_width = int(showImage.width() * scale_factor)
        new_height = int(showImage.height() * scale_factor)

        # 设置新的最大大小
        self.output[0].setMaximumSize(new_width, new_height)

        self.output[0].setPixmap(QPixmap(showImage))
        self.output[0].setScaledContents(True)

        # # 将BGR图像转换为RGB格式
        # rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #
        # # 将图像编码为二进制格式
        # h, w, ch = rgb_frame.shape
        # bytes_per_line = ch * w
        # qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        #
        # # 调整图像大小以适应 QLabel
        # qt_image = qt_image.scaled(self.output.width(), self.output.height(), Qt.KeepAspectRatio)
        #
        # # 创建 QPixmap，并在 QLabel 中居中显示图像
        # pixmap = QPixmap.fromImage(qt_image)
        # self.output.setPixmap(pixmap)
        #
        # # 设置 QLabel 的对齐方式为居中
        # self.output.setAlignment(Qt.AlignCenter)

