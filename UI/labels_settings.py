import os
from json import loads

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QPushButton, QCheckBox, QListWidgetItem, QMenu, QAction, QMessageBox, \
    QMainWindow
from qfluentwidgets import CheckBox, RoundMenu, RadioButton, PillToolButton, TogglePushButton, InfoBarPosition, InfoBar, \
    setCustomStyleSheet, MessageBox

from UI.labels_settings_ui import Ui_labels_settings
from common.config import MyQuestionMessageBox
from common.style_sheet import StyleSheet


class LabelsSettings(Ui_labels_settings, QMainWindow):
    def __init__(self, controller=None):
        super().__init__()
        self.controller = controller
        self.setupUi(self)
        self.resize(1000, 600)
        self.setWindowTitle("标签设置")
        # 新建的窗口始终位于当前屏幕的最前面
        self.setWindowFlags(Qt.WindowStaysOnTopHint)
        # 阻塞父类窗口不能点击
        self.setWindowModality(Qt.ApplicationModal)
        # self.setStyleSheet("background-color: #202020;")
        # self.closeEvent = self.closeEvent

        self.buttons = []
        self.checkboxes = []
        self.checkboxes_data = {}
        self.id = []
        self.FullCollection_path = ''
        self.translations = {}
        self.full_buttons = []

        self.open.setVisible(False)
        self.pushButton.setVisible(False)
        self.pushButton_2.clicked.connect(self.save)
        self.sets_list.itemClicked.connect(self.select)
        self.models_list.itemClicked.connect(self.visit_sets)
        self.pushButton.clicked.connect(self.displays_selection)
        self.delete_2.clicked.connect(self.on_delete_clicked)
        self.open.clicked.connect(self.on_open_clicked)
        self.fan.clicked.connect(self.on_fan_clicked)
        self.fan.setVisible(False)
        # 设置顶部显示

        self.labels_part_2.setAlignment(Qt.AlignTop)
        self.SmoothScrollArea.setStyleSheet("border: none;background-color: transparent;")
        # 右键菜单栏
        self.sets_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.sets_list.customContextMenuRequested.connect(self.show_context_menu)
        StyleSheet.Demo.apply(self)
        self.modellist()

    def on_fan_clicked(self):
        if self.checkboxes is None:
            return
        for checkbox in self.checkboxes:
            if checkbox.isChecked():
                checkbox.setChecked(False)
            else:
                checkbox.setChecked(True)

    def visit_sets(self, item):
        self.fan.setVisible(False)
        script_directory = os.path.dirname(os.path.abspath(__file__))
        parent_directory = os.path.dirname(script_directory)
        # labels文件夹的路径
        self.FullCollection_path = os.path.join(parent_directory, 'labels', item.text())
        if self.FullCollection_path:
            self.open_file()

    def modellist(self):
        # 清空 listwidget，以便重新加载文件夹名
        self.models_list.clear()

        # 获取当前脚本所在目录的路径
        script_directory = os.path.dirname(os.path.abspath(__file__))
        parent_directory = os.path.dirname(script_directory)

        # labels文件夹的路径
        labels_directory = os.path.join(parent_directory, 'labels')

        # 检查labels文件夹是否存在
        if os.path.exists(labels_directory) and os.path.isdir(labels_directory):
            # 遍历labels文件夹内的文件夹
            for folder_name in os.listdir(labels_directory):
                # 确保是文件夹而不是文件
                folder_path = os.path.join(labels_directory, folder_name)
                if os.path.isdir(folder_path):
                    # 将文件夹名添加到listwidget
                    self.models_list.addItem(folder_name)
        else:
            print("labels文件夹不存在或者不是一个文件夹")

    def rename(self, item, old_name):
        new_name = os.path.join(self.FullCollection_path, item.text() + '.pbtxt')
        item.setData(Qt.UserRole, new_name)
        # 检查新名称是否与旧名称不同
        if old_name != new_name:
            # 执行重命名操作，将旧名称改为新名称
            try:
                os.rename(old_name, new_name)
            except OSError as e:
                return

    def rename_file(self):
        item = self.sets_list.currentItem()  # 获取当前选中的项目

        if item:
            # 如果文件名与标签集名相同不可删除
            filename, extension = os.path.splitext(os.path.basename(item.data(Qt.UserRole)))
            if os.path.basename(self.FullCollection_path) == filename or os.path.basename(
                    self.FullCollection_path) + '_re' == filename:
                w = MessageBox(
                    title='警告',
                    content='不能对基础标签集进行操作',
                    parent=self
                )
                w.show()
                return

        old_name = item.data(Qt.UserRole)  # 获取该项目的旧名称
        # 将该项目设为可编辑状态并开始编辑
        item.setFlags(item.flags() | Qt.ItemIsEditable)
        self.sets_list.editItem(item)

        self.sets_list.itemChanged.connect(lambda item: self.rename(item, old_name))
        item.setFlags(item.flags() & ~Qt.ItemIsEditable)

    def on_open_clicked(self):
        self.FullCollection_path = None
        self.FullCollection_path = QtWidgets.QFileDialog.getExistingDirectory(self, "选择权重标签子集", "labels")
        if self.FullCollection_path:
            self.open_file()

    def open_file(self):
        # 清空list
        self.translations = {}
        self.sets_list.clear()
        # 清空工作区
        while self.labels_part_2.count():
            item = self.labels_part_2.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        # 将文本内容转换为字典
        folder_path = os.path.join(self.FullCollection_path, "字典.txt")
        with open(folder_path, 'r', encoding='utf-8') as f:
            content = f.read()
        self.translations = loads(content)

        path = os.path.join(self.FullCollection_path, os.path.basename(self.FullCollection_path) + ".pbtxt")
        self.full_buttons, id = self.read_label_map(path)
        self.full_buttons = self.create_buttons(self.full_buttons)

        for filename in os.listdir(self.FullCollection_path):
            file_path = os.path.join(self.FullCollection_path, filename)
            base, ext = os.path.splitext(filename)
            # 如果文件后缀不是 "bptxt"，则跳过
            if ext != ".pbtxt":
                continue

            item = QListWidgetItem(base)
            item.setData(Qt.UserRole, file_path)
            self.sets_list.addItem(item)

    def show_context_menu(self, position):
        item = self.sets_list.itemAt(position)
        if item is not None:
            menu = RoundMenu(self.sets_list)

            action = QAction("重命名", menu)
            action.triggered.connect(self.rename_file)
            menu.addAction(action)

            action = QAction("新建", menu)
            action.triggered.connect(self.save)
            menu.addAction(action)

            action = QAction("删除", menu)
            action.triggered.connect(self.on_delete_clicked)
            menu.addAction(action)

            menu.exec_(self.sets_list.mapToGlobal(position))

    def on_delete_clicked(self):
        selected_item = self.sets_list.currentItem()
        if selected_item:
            # 如果文件名与标签集名相同不可删除
            filename, extension = os.path.splitext(os.path.basename(selected_item.data(Qt.UserRole)))
            if os.path.basename(self.FullCollection_path) == filename or os.path.basename(
                    self.FullCollection_path) + '_re' == filename:
                w = MessageBox(
                    title='警告',
                    content='不能对基础标签集进行操作',
                    parent=self
                )
                w.show()
                return

        if selected_item:
            row = self.sets_list.row(selected_item)
            self.buttons = []
            self.checkboxes = []
            self.checkboxes_data = {}
            while self.labels_part_2.count():
                item = self.labels_part_2.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
            item = self.sets_list.takeItem(row)
            del item
            os.remove(selected_item.data(Qt.UserRole))

    def save(self):
        base = "新建标签子集"
        suffix = ""
        counter = 1

        while True:
            path = os.path.join(self.FullCollection_path, base + suffix + '.pbtxt')
            if not os.path.exists(path):
                break
            suffix = f"_{counter}"
            counter += 1

        with open(path, "w", encoding="utf-8") as f:
            for checkbox in self.checkboxes:
                item = self.checkboxes_data[checkbox.text()]
                if checkbox.isChecked():
                    item = self.checkboxes_data[checkbox.text()]
                    f.write(f"item {{\n  name: \"{item[1]}\"\n  id: {item[0]}\n}}\n")

            item = QListWidgetItem(base + suffix)
            item.setData(Qt.UserRole, path)
            if self.sets_list.findItems(base + suffix, Qt.MatchExactly) != []:
                item = self.sets_list.takeItem(self.sets_list.row(item))
            self.sets_list.addItem(item)
            self.sets_list.setCurrentItem(item)
            self.rename_file()

        InfoBar.success(
            title='成功！',
            content='成功新建标签集',
            orient=Qt.Horizontal,
            isClosable=True,
            position=InfoBarPosition.BOTTOM_RIGHT,
            # position='Custom',   # NOTE: use custom info bar manager
            duration=2000,
            parent=self.parent().parent().parent()
        )

    def displays_selection(self):
        for checkbox in self.checkboxes:
            if checkbox.isChecked():
                print('选中的复选框：', checkbox.text())
                print(self.checkboxes_data[checkbox.text()])

    # 选择标签子集并显示
    def select(self):
        self.fan.setVisible(True)
        selected_item = self.sets_list.currentItem()
        if selected_item:
            # 清空工作区
            self.id = []
            self.buttons = []
            self.checkboxes = []
            self.checkboxes_data = {}
            while self.labels_part_2.count():
                item = self.labels_part_2.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
            # 显示选中标签集
            label_map, self.id = self.read_label_map(selected_item.data(Qt.UserRole))
            self.buttons = self.create_buttons(label_map)
            self.display_checkboxes(self.buttons)

    # 字典转按钮
    def create_buttons(self, dictionary):
        buttons = []
        for key, value in dictionary.items():
            button_data = (key, value)
            button = QPushButton(text="")
            button.data = button_data
            button.selected = False
            buttons.append(button)
        return buttons

    # 展示标签集
    def display_checkboxes(self, buttons):
        row = 0
        col = 0
        for button in self.full_buttons:
            # 创建复选框并添加到布局中，使用 button 变量作为标签
            checkbox_name = self.translations[str(button.data[1])]
            checkbox = TogglePushButton(checkbox_name)

            if button.data[0] in self.id:
                checkbox.setChecked(True)
            else:
                checkbox.setChecked(False)

            self.checkboxes_data[checkbox_name] = button.data
            self.labels_part_2.addWidget(checkbox, row, col)  # 添加到指定的行和列
            self.checkboxes.append(checkbox)

            # 更新行和列，以便下一个复选框被放在正确的位置
            col += 1
            if col > 2:  # 如果列数大于2，换行并重置列数
                col = 0
                row += 1

    # 新建标签子集
    def save_to_file(self, button_data_list):
        with open("button_data.pbtxt", "w", encoding="utf-8") as f:
            for data in button_data_list:
                f.write(f"item {{ name: \"{data[0]}\", id: {data[1]} }}\n")

    # 读文件转字典
    def read_label_map(self, label_map_file):
        label_map = {}
        class_ids = set()
        name = ""
        class_id = ""
        with open(label_map_file, "r") as f:
            for line in f:
                if line.startswith("  name:"):
                    name = line.split('"')[1]
                elif line.startswith("  id:") or line.startswith("  label_id:"):
                    class_id = int(line.strip().split(" ")[-1])
                    label_map[class_id] = name
                    class_ids.add(class_id)
        return label_map, class_ids


if __name__ == "__main__":
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
    app = QApplication([])
    # app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyqt5', palette=LightPalette()))
    labels_settings = LabelsSettings()
    labels_settings.ui.show()
    app.exec()
