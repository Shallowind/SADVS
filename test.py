from PyQt5.QtWidgets import QApplication, QMainWindow, QTreeWidget, QTreeWidgetItem
from PyQt5.QtCore import Qt

app = QApplication([])

# 创建一个主窗口
main_window = QMainWindow()

# 创建一个QTreeWidget
tree_widget = QTreeWidget(main_window)
main_window.setCentralWidget(tree_widget)

# 添加一些示例项目
item1 = QTreeWidgetItem(tree_widget, ["Item 1"])
item2 = QTreeWidgetItem(tree_widget, ["Item 2"])
item3 = QTreeWidgetItem(tree_widget, ["Item 3"])

# 创建一个槽函数来处理双击事件
def rename_item(item, column):
    if item and column == 0:  # 确保是项目的第一列
        item.setFlags(item.flags() | Qt.ItemIsEditable)
        tree_widget.editItem(item, column)  # 启动编辑模式

# 连接itemDoubleClicked信号到槽函数
tree_widget.itemDoubleClicked.connect(rename_item)

# 自定义样式表以调整文本的外观
tree_widget.setStyleSheet("QTreeWidget::Item { padding: 2px; }")

# 显示主窗口
main_window.show()

app.exec_()
