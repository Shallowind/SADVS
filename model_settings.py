import sys
import os

import qdarkstyle
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog, QMessageBox
from PyQt5 import uic, QtWidgets

from utils.myutil import Globals
from labels_settings import LabelsSettings
from json import loads

class ModelSettings(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window  # 传入主窗口的引用
        self.ui = uic.loadUi('model_settings.ui')
        self.ui.resize(800, 400)
        self.ui.setWindowTitle("识别设置")
        self.ui.show()  # 显示窗口
        self.ui.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())

        self.ui.closeEvent = self.closeEvent
        self.path = ''
        self.ui.label_combox.clear()
        self.init_labels_combox()

        # 获得保存路径
        self.ui.save_address_button.clicked.connect(self.saveAddress)
        # 获得权重文件
        self.ui.pt_select_button.clicked.connect(self.ptSelectButton)
        # 模型选择按钮
        self.ui.model_select_button.clicked.connect(self.modelSelect)
        # 保存模型参数
        self.ui.save_button.clicked.connect(self.saveSettings)
        # 保存模型参数链接‘enter’键
        self.ui.save_button.setShortcut('enter')
        # 开始识别按钮
        self.ui.begin.clicked.connect(self.beginIdentify)
        # 显示最大识别数量信息
        self.ui.max_size.clicked.connect(self.max_size_clicked)
        # 显示置信度信息
        self.ui.conf.clicked.connect(self.conf_clicked)
        # 显示交并比信息
        self.ui.iou.clicked.connect(self.iou_clicked)
        # 设置识别框线条粗细
        self.ui.line_thickness_button.clicked.connect(self.line_thickness_clicked)
        # 初始化标签列表
        self.ui.model_combox.currentIndexChanged.connect(self.init_labels_combox)
        # 启动标签自定义设置
        self.ui.labels_select_button.clicked.connect(self.labelsetting)
        # 加载模型默认参数 上一次识别的参数
        try:
            with open('Default settings.txt', 'r', encoding='utf-8') as f:
                content = f.read()
            seted = loads(content)
            self.ui.save_address_edit.setText(seted['save_path'])
            self.ui.pt_edit.setText(seted['pt_path'])
            self.ui.model_combox.setCurrentText(seted['model_select'])
            self.ui.label_combox.setCurrentText(seted['labels'])
            self.ui.max_size_comboBox.setCurrentText(seted['max_det'])
            self.ui.conf_doubleSpinBox.setValue(seted['conf'])
            self.ui.iou_doubleSpinBox.setValue(seted['iou'])
            self.ui.line_thickness.setValue(seted['line_thickness'])
        except FileNotFoundError:
            print("文件不存在")

    # 显示识别框线条粗细信息
    def line_thickness_clicked(self):
        message_box = QMessageBox()
        # 设置对话框的标题
        message_box.setWindowTitle("设置线条粗细")
        # 设置对话框的文本内容
        message = "线条粗细表示检测框的粗细。\n\n"
        message += "yolo_slowfast：默认线条粗细是2。\n"
        message += "yolo5：默认线条粗细是3。\n"
        message_box.setText(message)
        # 添加 OK 按钮
        message_box.addButton(QMessageBox.Ok)
        # 显示对话框
        message_box.exec_()

    # 显示交并比信息
    def iou_clicked(self):
        message_box = QMessageBox()
        # 设置对话框的标题
        message_box.setWindowTitle("设置IoU")
        # 设置对话框的文本内容
        message = "IoU表示两个矩形的交集面积除以它们的并集面积。IoU值越高，表示两个矩形之间的重叠程度越大。\n\n"
        message += "yolo_slowfast：默认IoU是0.4。\n"
        message += "yolo5：默认iou是0.45。\n"
        message_box.setText(message)
        # 添加 OK 按钮
        message_box.addButton(QMessageBox.Ok)
        # 显示对话框
        message_box.exec_()

    # 显示置信度信息
    def conf_clicked(self):
        message_box = QMessageBox()
        # 设置对话框的标题
        message_box.setWindowTitle("设置置信度")
        # 设置对话框的文本内容
        message = "置信度通常用于表示分类器对于某个样本属于某个类别的信心水平。置信度越高，表示分类器对该样本属于该类别的确定程度越高。\n\n"
        message += "yolo_slowfast：默认置信度是0.4。\n"
        message += "yolo5：默认置信度是0.25。\n"
        message_box.setText(message)
        # 添加 OK 按钮
        message_box.addButton(QMessageBox.Ok)
        # 显示对话框
        message_box.exec_()

    # 显示最大识别数量信息
    def max_size_clicked(self):
        message_box = QMessageBox()
        # 设置对话框的标题
        message_box.setWindowTitle("设置最大识别数量")
        # 设置对话框的文本内容
        message = "最大识别数量表示每张图片允许的最大检测数。\n\n"
        message += "yolo_slowfast：默认最大识别数量是100。\n"
        message += "yolo5：默认最大识别数量是1000。\n"
        message_box.setText(message)
        # 添加 OK 按钮
        message_box.addButton(QMessageBox.Ok)
        # 显示对话框
        message_box.exec_()

    # 标签自定义设置
    def labelsetting(self):
        self.settings_window = LabelsSettings(self)
        # self.settings_window.setWindowZOrder(Qt.TopMost)
        self.settings_window.ui.show()

    # 初始化标签列表
    def init_labels_combox(self):
        self.ui.label_combox.clear()
        path = os.getcwd()
        path = os.path.join(path, "labels")
        self.path = os.path.join(path, self.ui.model_combox.currentText())

        for filename in os.listdir(self.path):
            base, ext = os.path.splitext(filename)
            # 如果文件后缀不是 "bptxt"，则跳过
            if ext != ".pbtxt":
                continue
            self.ui.label_combox.insertItem(0, base)

    # 读取默认/自定义标签
    def labelsSelect(self):
        class_ids = self.read_label_map(os.path.join(self.path, self.ui.label_combox.currentText() + '.pbtxt'))
        print(os.path.join(self.path, self.ui.label_combox.currentText()))
        Globals.select_labels = class_ids

    # 读取默认/自定义标签
    def read_label_map(self, label_map_file):
        class_ids = []
        with open(label_map_file, "r") as f:
            for line in f:
                if line.startswith("  id:") or line.startswith("  label_id:"):
                    class_id = int(line.strip().split(" ")[-1])
                    class_ids.append(class_id)
        return class_ids

    # 关闭窗口
    def closeEvent(self, event):
        self.main_window.ui.setEnabled(True)  # 关闭第二个窗口时恢复主窗口活动状态
        self.main_window.settings_window = None  # 将第二个窗口的引用设置为 None
        self.main_window.startIdentifyThread()

    # 保存地址
    def saveAddress(self):
        folder_path = QFileDialog.getExistingDirectory()
        self.ui.save_address_edit.setText(folder_path)

    # 选择权重
    def ptSelectButton(self):
        model_path = QtWidgets.QFileDialog.getOpenFileName(self, "选择权重", "weights", "Model files(*.pt)")
        self.ui.pt_edit.setText(model_path[0])

    # 模型选择
    def modelSelect(self):
        # 创建一个 QMessageBox 对话框
        message_box = QMessageBox()
        # 设置对话框的标题
        message_box.setWindowTitle("模型选择")
        # 设置对话框的文本内容
        message = "请在这里选择你需要的模型。\n\n"
        message += "模型1：yolo_slowfast。\n"
        message += "模型2：yolo。\n"
        message_box.setText(message)
        # 添加 OK 按钮
        message_box.addButton(QMessageBox.Ok)
        # 显示对话框
        message_box.exec_()

    # 开始识别
    def beginIdentify(self):
        # 判断是否缺少信息
        if not self.ui.save_address_edit.text():
            # 如果文本内容为空，显示提示消息
            QMessageBox.warning(self, "警告", "保存地址不能为空")
        elif not self.ui.pt_edit.text():
            # 如果文本内容为空，显示提示消息
            QMessageBox.warning(self, "警告", "权重地址不能为空")
        else:
            settings_data = {
                'saved': True,
                'save_path': self.ui.save_address_edit.text(),
                'pt_path': self.ui.pt_edit.text(),
                'model_select': self.ui.model_combox.currentText(),
                'labels': self.ui.label_combox.currentText(),
                'max_det': self.ui.max_size_comboBox.currentText(),
                'conf': self.ui.conf_doubleSpinBox.value(),
                'iou': self.ui.iou_doubleSpinBox.value(),
                'line_thickness': self.ui.line_thickness.value()
            }
            Globals.settings = settings_data
            print(Globals.settings)
            self.labelsSelect()

            import json

            # 假设这是从数据库中获取的数据
            data = {
                'video_path' : self.main_window.path,
                'save_path': self.ui.save_address_edit.text(),
                'pt_path': self.ui.pt_edit.text(),
                'model_select': self.ui.model_combox.currentText(),
                'labels': self.ui.label_combox.currentText(),
                'max_det': self.ui.max_size_comboBox.currentText(),
                'conf': self.ui.conf_doubleSpinBox.value(),
                'iou': self.ui.iou_doubleSpinBox.value(),
                'line_thickness': self.ui.line_thickness.value()
            }

            # 将数据写入文件
            with open('Default settings.txt', 'w') as f:
                json.dump(data, f)

            self.ui.close()

    # 保存模型参数
    def saveSettings(self):
        return

if __name__ == "__main__":
    app = QApplication([])
    modelsettings = ModelSettings()
    modelsettings.ui.show()
    app.exec()
