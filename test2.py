import sys
import logging

from PyQt5.QtGui import QTextCursor
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QTextEdit, QPushButton, QWidget
from PyQt5.QtCore import Qt, pyqtSlot

class ConsoleRedirector:
    def __init__(self, parent, text_widget, color=None):
        self.parent = parent
        self.text_widget = text_widget
        self.color = color

        # Redirect sys.stdout and sys.stderr
        self.original_stdout = sys.stdout
        sys.stdout = self
        self.original_stderr = sys.stderr
        sys.stderr = self

        # Redirect logging output
        logging.basicConfig(stream=self)

    def write(self, text):
        # Display text in the console
        cursor = self.text_widget.textCursor()
        cursor.movePosition(QTextCursor.End)
        if self.color:
            cursor.insertHtml(f'<font color="{self.color.name()}">{text}</font>')
        else:
            cursor.insertText(text)
        self.text_widget.setTextCursor(cursor)
        self.text_widget.ensureCursorVisible()

    def flush(self):
        pass

    def restore_stdout_stderr(self):
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr

    def restore_logging(self):
        logging.basicConfig(stream=self.original_stdout)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)

        self.output_textedit = QTextEdit(self)
        layout.addWidget(self.output_textedit)

        self.redirect_button = QPushButton('Redirect Console', self)
        self.redirect_button.clicked.connect(self.redirect_console)
        layout.addWidget(self.redirect_button)

    @pyqtSlot()
    def redirect_console(self):
        self.console_redirector = ConsoleRedirector(self, self.output_textedit, color=None)
        print("Console redirected.")

    def closeEvent(self, event):
        if hasattr(self, 'console_redirector'):
            self.console_redirector.restore_stdout_stderr()
            self.console_redirector.restore_logging()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.setGeometry(100, 100, 800, 600)
    mainWindow.show()
    sys.exit(app.exec_())
