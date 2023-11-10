import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt

class MyMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Matplotlib in PyQt")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout(self.central_widget)

        # 创建Matplotlib的FigureCanvas和NavigationToolbar
        self.canvas = FigureCanvas(plt.Figure())
        self.toolbar = NavigationToolbar(self.canvas, self)

        # 将它们添加到布局中
        self.layout.addWidget(self.toolbar)
        self.layout.addWidget(self.canvas)

        # 在FigureCanvas上绘制折线图
        self.draw_line_chart()

    def draw_line_chart(self):
        # 这里放入你的Matplotlib绘图代码
        # 注意：要绘制的图形应该是 self.canvas.figure 中的Figure对象
        ax = self.canvas.figure.add_subplot(111)
        ax.plot([1, 2, 3, 4], [10, 20, 25, 30], label='Example Line')
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.legend()

        # 绘制完成后刷新图形
        self.canvas.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyMainWindow()
    window.show()
    sys.exit(app.exec_())
