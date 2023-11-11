import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
class PieChart(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Pie Chart in PyQt5")
        self.setGeometry(100, 100, 800, 600)

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)

        # 创建Matplotlib图形
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        # 添加到布局中
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        # 调用绘制饼图的函数
        self.plot_pie_chart()

    def plot_pie_chart(self):
        # 示例数据
        labels = ('Aaa',)
        sizes = [100]

        # 绘制饼图
        self.ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        self.ax.axis('equal')  # 使饼图保持圆形

        # 在Matplotlib图形上绘制饼图后，更新Qt窗口
        self.canvas.draw()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = PieChart()
    window.show()
    sys.exit(app.exec_())
