from PyQt5.QtWidgets import QApplication, QMainWindow, QTreeWidget, QTreeWidgetItem

app = QApplication([])

# 创建一个主窗口
main_window = QMainWindow()

# 创建一个QTreeWidget
tree_widget = QTreeWidget(main_window)
main_window.setCentralWidget(tree_widget)

# 添加一些示例项目
item1 = QTreeWidgetItem(tree_widget, ["Item 1"])
item2 = QTreeWidgetItem(tree_widget, ["Item 2"])
item3 = QTreeWidgetItem(item2, ["Subitem 1"])
item4 = QTreeWidgetItem(tree_widget, ["Item 3"])

# 删除item2
index = tree_widget.indexOfTopLevelItem(item2)
if index != -1:
    tree_widget.takeTopLevelItem(index)
else:
    index = item2.parent().indexOfChild(item2)
    if index != -1:
        item2.parent().takeChild(index)

# 显示主窗口
main_window.show()

app.exec_()
