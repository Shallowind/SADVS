import json
import os
from json import loads

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QFileDialog, QMessageBox
from qfluentwidgets import MessageBox, FluentIcon
from qfluentwidgets.components.widgets.frameless_window import FramelessWindow

from UI.labels_settings import LabelsSettings
from UI.model_settings_ui import Ui_model_settings
from common.style_sheet import StyleSheet
from utils.myutil import Globals


class ModelSettings(Ui_model_settings, FramelessWindow):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window  # 传入主窗口的引用
        # 新建的窗口始终位于当前屏幕的最前面
        self.setWindowFlags(Qt.WindowStaysOnTopHint)
        # 阻塞父类窗口不能点击
        self.setWindowModality(Qt.ApplicationModal)

        self.setupUi(self)
        self.resize(800, 500)
        self.setWindowTitle("识别设置")
        # self.setStyleSheet("background: rgb(39, 39, 39);")

        StyleSheet.Demo.apply(self)
        self.model_combox.addItem("yolov5")
        self.model_combox.addItem("yolov8_BRA_DCNv3")
        self.model_combox.addItem("yolo_slowfast")

        self.id = 1

        self.closeEvent = self.closeEvent
        self.path = ''
        self.config_dict = {}
        self.label_combox.clear()
        self.init_labels_combox()

        self.save_address_button.clicked.connect(self.saveAddress)
        self.pt_select_button.clicked.connect(self.ptSelectButton)
        self.pdf_button.clicked.connect(self.pdfSelect)
        self.model_select_button.clicked.connect(self.modelSelect)
        self.save_button.clicked.connect(self.saveSettings)
        self.save_button.setShortcut('enter')
        self.begin.clicked.connect(self.beginIdentify)
        self.object_button.clicked.connect(self.object_clicked)
        self.max_size.clicked.connect(self.max_size_clicked)
        self.conf.clicked.connect(self.conf_clicked)
        self.iou.clicked.connect(self.iou_clicked)
        self.line_thickness_button.clicked.connect(self.line_thickness_clicked)

        self.model_combox.currentIndexChanged.connect(self.init_labels_combox)
        self.labels_select_button.clicked.connect(self.labelsetting)
        self.define_conbox.currentIndexChanged.connect(self.define_conbox_change)
        self.preinf.clicked.connect(self.pre_edit)
        self.pt_edit.setReadOnly(True)

        if os.path.exists('labels/yolov5'):
            path = 'labels/yolov5'
            for file in os.listdir(path):
                if file.endswith('.pbtxt'):
                    self.object_combox.addItem(file.split('.')[0])
            self.object_combox.setCurrentText('yolov5_re')

        try:
            with open("config.json", "r") as json_file:
                self.config_dict = json.load(json_file)
        except FileNotFoundError:
            self.config_dict = {}
        for defaultName in self.config_dict.keys():
            self.define_conbox.addItem(defaultName)
        self.define_conbox.setCurrentIndex(0)

        if self.model_combox.currentText() != 'yolo_slowfast':
            self.object_combox.setVisible(False)
            self.object_button.setVisible(False)
            self.CaptionLabel.setVisible(False)


        self.save_address_button.setIcon(FluentIcon.LABEL)
        self.pdf_button.setIcon(FluentIcon.LABEL)
        self.pt_select_button.setIcon(FluentIcon.LABEL)
        self.model_select_button.setIcon(FluentIcon.HELP)
        self.labels_select_button.setIcon(FluentIcon.LABEL)
        self.object_button.setIcon(FluentIcon.HELP)
        self.max_size.setIcon(FluentIcon.HELP)
        self.conf.setIcon(FluentIcon.HELP)
        self.iou.setIcon(FluentIcon.HELP)
        self.line_thickness_button.setIcon(FluentIcon.HELP)
        self.preinf.setIcon(FluentIcon.LABEL)


    def define_conbox_change(self):
        define_dict = self.config_dict[self.define_conbox.currentText()]
        self.save_address_edit.setText(define_dict['save_path'])
        self.pt_edit.setText('weights/yolov5s.pt')
        self.pdf_save_path.setText(define_dict['pdf_save_path'])
        self.model_combox.setCurrentText(define_dict['model_select'])
        if self.model_combox.currentText() == 'yolov8_BRA_DCNv3':
            self.pt_edit.setText('weights/yolov8_BRA_DCNv3_crowdhuman.pt')
        self.label_combox.setCurrentText(define_dict['labels'])
        self.max_size_comboBox.setValue(define_dict['max_det'])
        self.conf_doubleSpinBox.setValue(define_dict['conf'])
        self.iou_doubleSpinBox.setValue(define_dict['iou'])
        self.line_thickness.setValue(define_dict['line_thickness'])

    def object_clicked(self):
        message_box = MessageBox(
            "设置正常物体",
            "在识别时，选择之外的物体，即为异常物体。\n\n",
            self
        )
        message_box.yesButton.setText("OK")
        # 显示对话框
        message_box.exec_()

    def max_size_clicked(self):
        message_box = MessageBox(
            "设置最大识别数量",
            "最大识别数量表示每张图片允许的最大检测数。\n\n"
            "yolo_slowfast：默认最大识别数量是100。\n"
            "yolo5：默认最大识别数量是1000。\n",
            self
        )
        message_box.yesButton.setText("OK")
        # 显示对话框
        message_box.exec_()

    def line_thickness_clicked(self):
        message_box = MessageBox(
            "设置线条粗细",
            "线条粗细表示检测框的粗细。\n\n"
            "yolo_slowfast：默认线条粗细是2。\n"
            "yolo5：默认线条粗细是3。",
            self
        )
        message_box.yesButton.setText("OK")
        # 显示对话框
        message_box.exec_()

    def iou_clicked(self):
        message_box = MessageBox(
            "设置IoU",
            "IoU表示两个矩形的交集面积除以它们的并集面积。IoU值越高，表示两个矩形之间的重叠程度越大。\n\n"
            "yolo_slowfast：默认IoU是0.4。\n"
            "yolo5：默认iou是0.45。\n",
            self
        )
        message_box.yesButton.setText("OK")
        # 显示对话框
        message_box.exec_()

    def conf_clicked(self):
        # 创建一个 MessageBox 对话框
        message_box = MessageBox(
            "设置置信度",
            "置信度通常用于表示分类器对于某个样本属于某个类别的信心水平。置信度越高，表示分类器对该样本属于该类别的确定程度越高。\n\n"
            "yolo_slowfast：默认置信度是0.4。\n"
            "yolo5：默认置信度是0.25。\n",
            self
        )
        message_box.yesButton.setText("OK")
        # 显示对话框
        message_box.exec_()

    def labelsetting(self):
        self.main_window.father.switchTo(self.main_window.father.labelsInterface)
        self.close()
        # self.settings_window = LabelsSettings(self)
        # # self.settings_window.setWindowZOrder(Qt.TopMost)
        # self.settings_window.show()

    def init_labels_combox(self):
        self.label_combox.clear()
        path = os.getcwd()
        path = os.path.join(path, "labels")
        self.path = os.path.join(path, self.model_combox.currentText())
        try:
            for filename in os.listdir(self.path):
                base, ext = os.path.splitext(filename)
                # 如果文件后缀不是 "bptxt"，则跳过
                if ext != ".pbtxt":
                    continue
                self.label_combox.insertItem(0, base)
        except Exception as e:
            print(e)
        self.label_combox.setCurrentText(self.model_combox.currentText())
        if self.model_combox.currentText() != 'yolo_slowfast':
            flag = False
        else:
            flag = True
        self.object_combox.setVisible(flag)
        self.object_button.setVisible(flag)
        self.CaptionLabel.setVisible(flag)

    def labelsSelect(self):
        Globals.settings['select_labels'] = self.label_combox.currentText()
        class_ids = self.read_label_map(os.path.join(self.path, self.label_combox.currentText() + '.pbtxt'))
        print(os.path.join(self.path, self.label_combox.currentText()))
        Globals.select_labels = class_ids
        # 如果使用yolo_slowfast，则获取正常物体的标签
        if self.model_combox.currentText() == 'yolo_slowfast' or self.model_combox.currentText() == 'yolov8':
            Globals.select_objects = self.read_label_map(
                os.path.join('labels/yolov5', self.object_combox.currentText() + '.pbtxt'))

    @staticmethod
    def read_label_map(label_map_file):
        class_ids = []
        try:
            with open(label_map_file, "r") as f:
                for line in f:
                    if line.startswith("  id:") or line.startswith("  label_id:"):
                        class_id = int(line.strip().split(" ")[-1])
                        class_ids.append(class_id)
        except Exception as e:
            print(e)
        return class_ids

    def closeEvent(self, event):
        # self.main_window.setEnabled(True)  # 关闭第二个窗口时恢复主窗口活动状态
        self.main_window.settings_window = None  # 将第二个窗口的引用设置为 None
        self.main_window.startIdentifyThread()

    def saveAddress(self):
        folder_path = QFileDialog.getExistingDirectory()
        self.save_address_edit.setText(folder_path)

    def ptSelectButton(self):
        model_path = QtWidgets.QFileDialog.getOpenFileName(self, "选择权重", "weights", "Model files(*.pt)")
        self.pt_edit.setText(model_path[0])

    def pdfSelect(self):
        pdf_path = QFileDialog.getExistingDirectory()
        self.pdf_save_path.setText(pdf_path)

    def pre_edit(self):
        self.main_window.father.switchTo(self.main_window.father.SettingInterface)
        if not self.main_window.father.SettingInterface.informationCard.isExpand:
            self.main_window.father.SettingInterface.informationCard.toggleExpand()
        self.close()
    def modelSelect(self):
        # 创建一个 MessageBox 对话框
        message_box = MessageBox(
            "模型选择",
            "请在这里选择你需要的模型。\n\n"
            "模型1：yolo_slowfast。\n"
            "模型2：yolo。\n",
            self
        )
        message_box.yesButton.setText("OK")
        # 显示对话框
        message_box.exec_()

    def beginIdentify(self):
        if not self.save_address_edit.text():
            # 如果文本内容为空，显示提示消息
            # 创建一个 MessageBox 对话框
            message_box = MessageBox(
                "警告",
                "保存地址不能为空",
                self
            )
            message_box.yesButton.setText("OK")
            # 显示对话框
            message_box.exec_()
        elif not self.pt_edit.text():
            # 如果文本内容为空，显示提示消息
            message_box = MessageBox(
                "警告",
                "权重文件地址不能为空",
                self
            )
            message_box.yesButton.setText("OK")
            # 显示对话框
            message_box.exec_()
        elif not self.pdf_save_path.text():
            message_box = MessageBox(
                "警告",
                "PDF保存地址不能为空",
                self
            )
            message_box.yesButton.setText("OK")
            # 显示对话框
            message_box.exec_()
        else:
            settings_data = {
                'saved': True,
                'save_path': self.save_address_edit.text(),
                'pdf_save_path': self.pdf_save_path.text(),
                'pt_path': self.pt_edit.text(),
                'model_select': self.model_combox.currentText(),
                'labels': self.label_combox.currentText(),
                'max_det': self.max_size_comboBox.value(),
                'conf': self.conf_doubleSpinBox.value(),
                'iou': self.iou_doubleSpinBox.value(),
                'line_thickness': self.line_thickness.value()
            }
            Globals.settings = settings_data
            print(Globals.settings)
            self.labelsSelect()

            data = {
                'video_path': self.main_window.path,
                'save_path': self.save_address_edit.text(),
                'pdf_save_path': self.pdf_save_path.text(),
                'pt_path': self.pt_edit.text(),
                'model_select': self.model_combox.currentText(),
                'labels': self.label_combox.currentText(),
                'max_det': self.max_size_comboBox.value(),
                'conf': self.conf_doubleSpinBox.value(),
                'iou': self.iou_doubleSpinBox.value(),
                'line_thickness': self.line_thickness.value()
            }

            with open('Default settings.txt', 'w') as f:
                json.dump(data, f)

            self.close()

    def saveSettings(self):
        return
