# ICVT modules
from modules.labels.image_viewer import ImageViewer

# Extra packages
from tkinter import filedialog
from PyQt5.QtWidgets import QApplication

# Default python packages
import sys
import os

if __name__ == '__main__':
    # Specify the folder path
    folder_path = filedialog.askdirectory(title="Select the image folder",
                                          initialdir=os.path.dirname(os.path.abspath(__file__)))
    app = QApplication(sys.argv)

    viewer = ImageViewer(folder_path, False, "Visitor", "Empty", "ICV - Insect Communities VisiTinder")
    viewer.run()

    sys.exit(app.exec_())