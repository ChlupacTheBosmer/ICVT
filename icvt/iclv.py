# ICVT modules
from modules.labels.image_viewer import ImageViewer

# Default python packages
import sys
import os

import PyQt5

pyqt = os.path.dirname(PyQt5.__file__)
os.environ['QT_PLUGIN_PATH'] = os.path.join(pyqt, "Qt5/plugins")

# Extra packages
from tkinter import filedialog
from PyQt5.QtWidgets import QApplication


if __name__ == '__main__':
    # Specify the folder path
    folder_path = filedialog.askdirectory(title="Select the image folder",
                                          initialdir=os.path.dirname(os.path.abspath(__file__)))
    app = QApplication(sys.argv)

    viewer = ImageViewer(folder_path, True, "Correct", "Wrong", "ICLV - Insect Communities Label Verifier")
    viewer.run()

    sys.exit(app.exec_())