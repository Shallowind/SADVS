import sys
import os
from json import loads
import qdarkstyle
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QCheckBox, QListWidgetItem, QMenu, QAction, QMessageBox
from PyQt5 import uic, QtWidgets
from PyQt5.QtCore import Qt
from qdarkstyle import LightPalette

from utils.myutil import Globals


# from tensorflow.python.platform import gfile

class LabelsSettings(QWidget):
    def __init__(self, modelset):
        super().__init__()
        self.modelset = modelset
        self.ui = uic.loadUi("labels_settings.ui")
        self.ui.resize(1000, 600)
        self.ui.setWindowTitle("标签设置")
        self.ui.show()  # 显示窗口
        self.ui.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        # self.ui.closeEvent = self.closeEvent

        self.buttons = []
        self.checkboxes = []
        self.checkboxes_data = {}
        self.FullCollection_path = ''
        self.translations = {}

        if self.modelset.id == 1:
            path = os.getcwd()
            path = os.path.join(path, "labels")
            self.FullCollection_path = os.path.join(path, self.modelset.ui.model_combox.currentText())
            self.open_file()
        self.ui.open.setVisible(False)
        self.ui.pushButton.setVisible(False)
        self.ui.pushButton_2.clicked.connect(self.save)
        self.ui.sets_list.itemClicked.connect(self.select)
        self.ui.models_list.itemClicked.connect(self.visit_sets)
        self.ui.pushButton.clicked.connect(self.displays_selection)
        self.ui.delete_2.clicked.connect(self.on_delete_clicked)
        self.ui.open.clicked.connect(self.on_open_clicked)
        # self.ui.sets_list.itemChanged.connect(lambda item, old_name: self.on_item_changed(item, old_name))
        # 设置顶部显示
        self.ui.labels_part_2.setAlignment(Qt.AlignTop)

        # 右键菜单栏
        self.ui.sets_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.ui.sets_list.customContextMenuRequested.connect(self.show_context_menu)

        self.modellist()

    # def on_item_changed(self, item, old_name):
    #     print(222)
    #     print(old_name)
    #     name = item.text() + '.pbtxt'
    #     new_name = os.path.join(self.FullCollection_path, name)
    #     item.setData(Qt.UserRole, new_name)
    #     print(333)
    #     print(new_name)
    #     os.rename(old_name, new_name)

    def visit_sets(self, item):
        script_directory = os.path.dirname(os.path.abspath(__file__))
        # labels文件夹的路径
        self.FullCollection_path = os.path.join(script_directory, 'labels', item.text())
        if self.FullCollection_path:
            self.open_file()

    def modellist(self):
        # 清空 listwidget，以便重新加载文件夹名
        self.ui.models_list.clear()

        # 获取当前脚本所在目录的路径
        script_directory = os.path.dirname(os.path.abspath(__file__))

        # labels文件夹的路径
        labels_directory = os.path.join(script_directory, 'labels')

        # 检查labels文件夹是否存在
        if os.path.exists(labels_directory) and os.path.isdir(labels_directory):
            # 遍历labels文件夹内的文件夹
            for folder_name in os.listdir(labels_directory):
                # 确保是文件夹而不是文件
                folder_path = os.path.join(labels_directory, folder_name)
                if os.path.isdir(folder_path):
                    # 将文件夹名添加到listwidget
                    self.ui.models_list.addItem(folder_name)
        else:
            print("labels文件夹不存在或者不是一个文件夹")

    def rename_file(self):
        # get the currently selected item and its old name
        item = self.ui.sets_list.currentItem()
        old_name = item.data(Qt.UserRole)

        # make the item editable and start editing
        item.setFlags(item.flags() | Qt.ItemIsEditable)
        self.ui.sets_list.editItem(item)

        # wait until editing is finished
        QApplication.instance().processEvents()

        # get the new name from the edited item
        new_name = os.path.join(self.FullCollection_path, item.text() + '.pbtxt')

        # check if the new name is different from the old name
        if old_name != new_name:
            # perform the actual renaming operation
            os.rename(old_name, new_name)

    def on_open_clicked(self):
        self.FullCollection_path = None
        self.FullCollection_path = QtWidgets.QFileDialog.getExistingDirectory(self, "选择权重标签子集", "labels")
        if self.FullCollection_path:
            self.open_file()

    def open_file(self):
        # 清空list
        self.translations = {}
        self.ui.sets_list.clear()
        # 清空工作区
        while self.ui.labels_part_2.count():
            item = self.ui.labels_part_2.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        # 将文本内容转换为字典
        folder_path = os.path.join(self.FullCollection_path, "字典.txt")
        with open(folder_path, 'r', encoding='utf-8') as f:
            content = f.read()
        self.translations = loads(content)

        for filename in os.listdir(self.FullCollection_path):
            file_path = os.path.join(self.FullCollection_path, filename)
            base, ext = os.path.splitext(filename)
            # 如果文件后缀不是 "bptxt"，则跳过
            if ext != ".pbtxt":
                continue

            item = QListWidgetItem(base)
            item.setData(Qt.UserRole, file_path)
            self.ui.sets_list.addItem(item)

    def show_context_menu(self, position):
        item = self.ui.sets_list.itemAt(position)
        if item is not None:
            menu = QMenu(self.ui.sets_list)

            action = QAction("重命名", menu)
            action.triggered.connect(self.rename_file)
            menu.addAction(action)

            action = QAction("新建", menu)
            action.triggered.connect(self.save)
            menu.addAction(action)

            action = QAction("删除", menu)
            action.triggered.connect(self.on_delete_clicked)
            menu.addAction(action)

            menu.exec_(self.ui.sets_list.mapToGlobal(position))

    def on_delete_clicked(self):
        selected_item = self.ui.sets_list.currentItem()
        if selected_item:
            # 如果文件名与标签集名相同不可删除
            filename, extension = os.path.splitext(os.path.basename(selected_item.data(Qt.UserRole)))
            if os.path.basename(self.FullCollection_path) == filename:
                QMessageBox.warning(self, "警告", "不能删除此标签集")
                return

        if selected_item:
            row = self.ui.sets_list.row(selected_item)
            self.buttons = []
            self.checkboxes = []
            self.checkboxes_data = {}
            while self.ui.labels_part_2.count():
                item = self.ui.labels_part_2.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
            item = self.ui.sets_list.takeItem(row)
            del item
            os.remove(selected_item.data(Qt.UserRole))

    def save(self):
        base = "新建标签子集"
        path = os.path.join(self.FullCollection_path, base + '.pbtxt')
        with open(path, "w", encoding="utf-8") as f:
            for checkbox in self.checkboxes:
                if checkbox.isChecked():
                    item = self.checkboxes_data[checkbox.text()]
                    f.write(f"item {{\n  name: \"{item[1]}\"\n  id: {item[0]}\n}}\n")

            item = QListWidgetItem(base)
            item.setData(Qt.UserRole, path)
            self.ui.sets_list.addItem(item)

    def displays_selection(self):
        for checkbox in self.checkboxes:
            if checkbox.isChecked():
                print('选中的复选框：', checkbox.text())
                print(self.checkboxes_data[checkbox.text()])

    # 选择标签子集并显示
    def select(self):
        selected_item = self.ui.sets_list.currentItem()
        if selected_item:
            # 清空工作区
            self.buttons = []
            self.checkboxes = []
            self.checkboxes_data = {}
            while self.ui.labels_part_2.count():
                item = self.ui.labels_part_2.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
            # 显示选中标签集
            label_map, class_ids = self.read_label_map(selected_item.data(Qt.UserRole))
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
        for button in buttons:
            # 创建复选框并添加到布局中，使用 button 变量作为标签
            checkbox_name = self.translations[str(button.data)]
            checkbox = QCheckBox(checkbox_name)
            self.checkboxes_data[checkbox_name] = button.data
            self.ui.labels_part_2.addWidget(checkbox)
            self.checkboxes.append(checkbox)

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
    app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyqt5', palette=LightPalette()))
    labels_settings = LabelsSettings()
    labels_settings.ui.show()
    app.exec()
