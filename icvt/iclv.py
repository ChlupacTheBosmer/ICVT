# Default python packages
import sys
import os

# Extra python packages
import PyQt5
from PyQt5.QtCore import QTimer
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtWidgets import QSplashScreen

# ICVT modules
from modules.labels.image_viewer import ImageViewer

pyqt = os.path.dirname(PyQt5.__file__)
os.environ['QT_PLUGIN_PATH'] = os.path.join(pyqt, "Qt5/plugins")

# Extra packages
from tkinter import filedialog
from PyQt5.QtWidgets import QApplication


if __name__ == '__main__':

    def show_main_window():
        global viewer

        # Specify the folder path
        folder_path = filedialog.askdirectory(title="Select the image folder",
                                              initialdir=os.path.dirname(os.path.abspath(__file__)))

        #app = QApplication(sys.argv)

        viewer = ImageViewer(folder_path, True, "Correct", "Wrong", "ICLV - Insect Communities Label Verifier")
        viewer.run()

        #sys.exit(app.exec_())

    def resource_path(relative_path):
        try:
            base_path = sys._MEIPASS
        except Exception:
            base_path = os.path.abspath(".")

        return os.path.join(base_path, relative_path)

    global app
    app = QApplication([])

    # Splash screen
    splash_pix = QPixmap(resource_path("resources/img/iclv.png"))
    splash = QSplashScreen(splash_pix, Qt.WindowStaysOnTopHint)
    splash.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
    splash.setEnabled(False)

    # Optionally set the window icon for the splash screen
    splash.setWindowIcon(QIcon(resource_path("resources/img/iclv.png")))

    splash.show()

    # Close splash screen after 2 seconds
    QTimer.singleShot(2000, splash.close)
    QTimer.singleShot(2000, show_main_window)

    app.exec_()



