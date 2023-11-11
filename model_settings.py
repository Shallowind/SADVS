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

        self.id = 1

        self.ui.closeEvent = self.closeEvent
        self.path = ''
        self.ui.label_combox.clear()
        self.init_labels_combox()

        self.ui.save_address_button.clicked.connect(self.saveAddress)
        self.ui.pt_select_button.clicked.connect(self.ptSelectButton)
        self.ui.model_select_button.clicked.connect(self.modelSelect)
        self.ui.save_button.clicked.connect(self.saveSettings)
        self.ui.save_button.setShortcut('enter')
        self.ui.begin.clicked.connect(self.beginIdentify)

        self.ui.model_combox.currentIndexChanged.connect(self.init_labels_combox)
        self.ui.labels_select_button.clicked.connect(self.labelsetting)

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

            # 'save_path': self.ui.save_address_edit.text(),
            # 'pt_path': self.ui.pt_edit.text(),
            # 'model_select': self.ui.model_combox.currentText(),
            # 'labels': self.ui.label_combox.currentText(),
            # 'max_det': self.ui.max_size_comboBox.currentText(),
            # 'conf': self.ui.conf_doubleSpinBox.value(),
            # 'iou': self.ui.iou_doubleSpinBox.value(),
            # 'line_thickness': self.ui.line_thickness.value()
        except FileNotFoundError:
            print("文件不存在")

    def labelsetting(self):
        self.settings_window = LabelsSettings(self)
        # self.settings_window.setWindowZOrder(Qt.TopMost)
        self.settings_window.ui.show()

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

    def labelsSelect(self):
        class_ids = self.read_label_map(os.path.join(self.path, self.ui.label_combox.currentText() + '.pbtxt'))
        print(os.path.join(self.path, self.ui.label_combox.currentText()))
        Globals.select_labels = class_ids
        print(class_ids)

    def read_label_map(self, label_map_file):
        class_ids = []
        with open(label_map_file, "r") as f:
            for line in f:
                if line.startswith("  id:") or line.startswith("  label_id:"):
                    class_id = int(line.strip().split(" ")[-1])
                    class_ids.append(class_id)
        return class_ids

    def closeEvent(self, event):
        self.main_window.ui.setEnabled(True)  # 关闭第二个窗口时恢复主窗口活动状态
        self.main_window.settings_window = None  # 将第二个窗口的引用设置为 None
        self.main_window.startIdentifyThread()

    def saveAddress(self):
        folder_path = QFileDialog.getExistingDirectory()
        self.ui.save_address_edit.setText(folder_path)

    def ptSelectButton(self):
        model_path = QtWidgets.QFileDialog.getOpenFileName(self, "选择权重", "weights", "Model files(*.pt)")
        self.ui.pt_edit.setText(model_path[0])

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

    def beginIdentify(self):
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

    def saveSettings(self):
        return

if __name__ == "__main__":
    app = QApplication([])
    modelsettings = ModelSettings()
    modelsettings.ui.show()
    app.exec()
