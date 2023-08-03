from tkinter import filedialog
from PyQt5.QtWidgets import QApplication
import sys
import os
from image_viewer import ImageViewer

if __name__ == '__main__':
    # Specify the folder path
    folder_path = filedialog.askdirectory(title="Select the image folder",
                                          initialdir=os.path.dirname(os.path.abspath(__file__)))
    app = QApplication(sys.argv)

    viewer = ImageViewer(folder_path, True, "Correct", "Wrong", "ICLV - Insect Communities Label Verifier")
    viewer.run()

    sys.exit(app.exec_())