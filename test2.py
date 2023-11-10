import sys

import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QApplication, QMainWindow
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class MyMainWindow(QMainWindow):
    def __init__(self):
        super(QMainWindow, self).__init__()
        self.setWindowTitle("Matplotlib in PyQt")
        self.setGeometry(100, 100, 800, 600)
        # Assuming you have a Qt layout called verticalLayout_6
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout(self.central_widget)

        # Assuming your data is stored in self.frame_dict
        self.frame_dict = {
            0.0: {1: 'person Unknow', 2: 'person Unknow',3: 'person Unknow'},
            1.0: {1: 'person stand', 2: 'person carry/hold', 3: 'person Unknow'},
            2.0: {1: 'person Unknow', 2: 'person carry/hold', 3: 'person Unknow'},
            3.0: {1: 'person stand', 2: 'person carry/hold', 3: 'person 123'},
            # ... and so on
        }


        # Create a Matplotlib figure and canvas
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)

        # Call the method to update the plot
        self.update_plot()

    def update_plot(self):
        # Clear the previous plot
        self.ax.clear()

        # Create a dictionary to store counts for each unique action
        action_counts = {}

        # Iterate through the frame_dict and count occurrences of each unique action
        for t, action_dict in self.frame_dict.items():
            # Initialize counts dictionary for each timestamp
            counts = {}

            for order, action in action_dict.items():
                if action not in counts:
                    counts[action] = 0

                counts[action] += 1

                if action not in action_counts:
                    action_counts[action] = {'t': [], 'count': []}

                action_counts[action]['t'].append(t)
                action_counts[action]['count'].append(counts[action])

        # Plot each unique action as a vertical line with markers only at the endpoints
        for action, data in action_counts.items():
            for i in range(len(data['t']) - 1):
                t_start, t_end = data['t'][i], data['t'][i + 1]
                count_start, count_end = data['count'][i], data['count'][i + 1]

                # Plot a vertical line segment
                self.ax.plot([t_start, t_start], [count_start, count_end], marker='o', color='b', label=action)
                self.ax.plot([t_end, t_end], [count_start, count_end], marker='o', color='b')

        # Set labels and title
        self.ax.set_xlabel('t')
        self.ax.set_ylabel('Count')
        self.ax.set_title('Action Counts Over Time')

        # Add legend
        self.ax.legend()

        # Redraw the canvas
        self.canvas.draw()
# Example usage:
app = QApplication(sys.argv)
window = MyMainWindow()
window.show()
sys.exit(app.exec_())
