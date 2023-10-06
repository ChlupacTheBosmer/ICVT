from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog, QTreeWidget, QVBoxLayout, QPushButton, QTreeView, QFileSystemModel, QTreeWidgetItem, QMenu, QAction, QInputDialog
from PyQt5.QtCore import QDir
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtCore import QUrl
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtGui import QPixmap, QPainter, QColor, QPen, QImage
from PyQt5.QtWidgets import QDialog, QLabel, QVBoxLayout, QHBoxLayout, QToolBar
from PyQt5.QtWidgets import QMainWindow, QToolBar, QVBoxLayout, QTreeView, QPushButton, QWidget

import shutil
import PyQt5
import os
import sqlite3
import re
pyqt = os.path.dirname(PyQt5.__file__)
os.environ['QT_PLUGIN_PATH'] = os.path.join(pyqt, "Qt5/plugins")

from PyQt5.QtCore import QSortFilterProxyModel, Qt

class FolderFilterProxyModel(QSortFilterProxyModel):
    def __init__(self, filter_string='', *args, **kwargs):
        super(FolderFilterProxyModel, self).__init__(*args, **kwargs)
        self.filter_string = filter_string.lower()

    def setFilterString(self, filter_string):
        self.filter_string = filter_string.lower()
        self.invalidateFilter()

    def filterAcceptsRow(self, source_row, source_parent):
        source_model = self.sourceModel()
        index = source_model.index(source_row, 0, source_parent)
        if self.filter_string in source_model.filePath(index).lower():
            return False
        return True

class ICDM(QMainWindow):
    def __init__(self):
        super(ICDM, self).__init__()
        self.previewLabel = None

        # Create central widget to contain layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        hLayout = QHBoxLayout()

        # Initialize the toolbar
        toolbar = QToolBar()
        self.addToolBar(toolbar)

        openButton = QPushButton('Open Folder', self)
        openButton.clicked.connect(self.showDialog)
        toolbar.addWidget(openButton)

        # Create TreeView for visitor folders
        self.tree_visitor = QTreeView(self)
        self.model_visitor = QFileSystemModel()
        self.proxyModel_visitor = FolderFilterProxyModel("visitor")
        self.proxyModel_visitor.setSourceModel(self.model_visitor)
        self.tree_visitor.setModel(self.proxyModel_visitor)
        hLayout.addWidget(self.tree_visitor)

        self.tree_visitor.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree_visitor.customContextMenuRequested.connect(lambda pos, tree=self.tree_visitor: self.openMenu(pos, tree))

        self.tree_visitor.doubleClicked.connect(lambda: self.itemDoubleClicked(self.tree_visitor))

        # Create TreeView for empty folders
        self.tree_empty = QTreeView(self)
        self.model_empty = QFileSystemModel()
        self.proxyModel_empty = FolderFilterProxyModel("empty")
        self.proxyModel_empty.setSourceModel(self.model_empty)
        self.tree_empty.setModel(self.proxyModel_empty)
        hLayout.addWidget(self.tree_empty)

        self.tree_empty.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree_empty.customContextMenuRequested.connect(lambda pos, tree=self.tree_empty: self.openMenu(pos, tree))

        self.tree_empty.doubleClicked.connect(lambda: self.itemDoubleClicked(self.tree_empty))

        layout.addLayout(hLayout)

        # Set the layout for the central widget
        self.centralWidget().setLayout(layout)

        self.setWindowTitle('ICDM - Insect Communities Dataset Manager')
        self.show()

    def showDialog(self):
        folder_path = QFileDialog.getExistingDirectory(self, 'Select Folder')
        if folder_path:
            self.import_data(folder_path)

    def import_data(self, folder_path):
        self.model_visitor.setRootPath(folder_path)
        self.tree_visitor.setRootIndex(self.proxyModel_visitor.mapFromSource(self.model_visitor.index(folder_path)))

        self.model_empty.setRootPath(folder_path)
        self.tree_empty.setRootIndex(self.proxyModel_empty.mapFromSource(self.model_empty.index(folder_path)))

        self.tree_visitor.setSortingEnabled(True)
        self.tree_empty.setSortingEnabled(True)
        populate_database(folder_path)

    def openMenu(self, position, tree):
        menu = QMenu()
        index = tree.currentIndex()

        # This line is important: we go back to the source model
        source_index = tree.model().mapToSource(index)
        source_model = tree.model().sourceModel()

        # Use the source model to get the file path
        path = source_model.filePath(source_index)

        if path.lower().endswith(('.png', '.jpg', '.jpeg')):
            previewAction = QAction("Preview Image", self)
            previewAction.triggered.connect(lambda: self.previewImage(path))
            menu.addAction(previewAction)

        renameAction = QAction('Rename', self)
        renameAction.triggered.connect(lambda: self.renameItem(tree))
        menu.addAction(renameAction)

        deleteAction = QAction('Delete', self)
        deleteAction.triggered.connect(lambda: self.deleteItem(tree))
        menu.addAction(deleteAction)

        openAction = QAction('Open', self)
        openAction.triggered.connect(lambda: self.openItem(tree))
        menu.addAction(openAction)

        if os.path.isdir(path):
            setRootAction = QAction('Set as Root', self)
            setRootAction.triggered.connect(lambda: self.setAsRoot(tree))
            menu.addAction(setRootAction)

        menu.exec_(tree.viewport().mapToGlobal(position))

    def itemClicked(self):
        pass

    def itemDoubleClicked(self, tree):

        index, model = self.getSourceAttributes(tree)

        file_path = model.filePath(index)  # Use the source index to get the file path
        if file_path.lower().endswith('.jpg'):  # Check if the double-clicked item is a .jpg image
            self.previewImage(file_path)

    def previewImage(self, image_path):
        # Get the image
        img = QImage(image_path)
        img_width = img.width()
        img_height = img.height()

        # Initialize painter
        painter = QPainter(img)
        pen = QPen(Qt.red)
        pen.setWidth(3)
        painter.setPen(pen)

        # Find the corresponding label file
        label_path = image_path.replace('.jpg', '.txt')
        if os.path.exists(label_path):
            with open(label_path, 'r') as file:
                for line in file:
                    # Assume that the label file has space-separated values and the coordinates are normalized
                    # Format: <object-class> <x_center> <y_center> <width> <height>
                    x_center, y_center, width, height = map(float, line.split()[1:])

                    # Denormalize and convert to integer
                    x1 = int((x_center - width / 2) * img_width)
                    y1 = int((y_center - height / 2) * img_height)
                    x2 = int((x_center + width / 2) * img_width)
                    y2 = int((y_center + height / 2) * img_height)

                    # Draw rectangle
                    painter.drawRect(x1, y1, x2 - x1, y2 - y1)

        # End painting
        painter.end()

        # Display the image with rectangle
        pixmap = QPixmap.fromImage(img)
        self.previewLabel = QLabel()  # Save the QLabel in the class attribute
        self.previewLabel.setPixmap(pixmap)
        self.previewLabel.show()

    def getSourceAttributes(self, tree):

        # Get index and model for the specific tree
        index = tree.currentIndex()
        source_index = tree.model().mapToSource(index)
        source_model = tree.model().sourceModel()

        return source_index, source_model


    def renameItem(self, tree):

        index, model = self.getSourceAttributes(tree)

        if index.isValid():
            old_name = model.fileName(index)
            path = model.filePath(index)
            directory = os.path.dirname(path)

            new_name, ok = QInputDialog.getText(self, 'Rename', f'Rename {old_name} to:', text=old_name)
            if ok and new_name:
                new_path = os.path.join(directory, new_name)
                os.rename(path, new_path)

    def deleteItem(self, tree):

        index, model = self.getSourceAttributes(tree)

        if index.isValid():
            path = model.filePath(index)
            confirm_msg = QMessageBox()
            confirm_msg.setIcon(QMessageBox.Warning)
            confirm_msg.setWindowTitle('Delete Item')
            confirm_msg.setText(f'Are you sure you want to delete {os.path.basename(path)}?')
            confirm_msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)

            result = confirm_msg.exec_()
            if result == QMessageBox.Yes:
                if os.path.isfile(path):
                    os.remove(path)
                elif os.path.isdir(path):
                    shutil.rmtree(path)

    def openItem(self, tree):

        index, model = self.getSourceAttributes(tree)

        if index.isValid():
            path = model.filePath(index)
            if os.path.isfile(path):
                QDesktopServices.openUrl(QUrl(f"file:///{path}", QUrl.TolerantMode))

    def setAsRoot(self, tree):

        index, model = self.getSourceAttributes(tree)
        folder_path = model.filePath(index)

        print(folder_path)

        if os.path.isdir(folder_path):
            model.setRootPath(folder_path)
            tree.setRootIndex(tree.model().mapFromSource(model.index(folder_path)))
        else:
            QMessageBox.warning(self, "Warning", "Selected item is not a directory.")

def create_database():
    conn = sqlite3.connect("metadata.db")
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS metadata (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        recording_id TEXT,
        video_file_id TEXT,
        frame_no INTEGER,
        visit_no INTEGER,
        crop_no INTEGER,
        x1 INTEGER,
        y1 INTEGER,
        x2 INTEGER,
        y2 INTEGER,
        full_path TEXT,
        label_path TEXT
    );
    """)
    conn.commit()
    conn.close()


def populate_database(root_path, batch_size=500):
    print("Creating database...")
    conn = sqlite3.connect("metadata.db")
    cursor = conn.cursor()

    batch_data = []
    count = 0

    for subdir, _, files in os.walk(root_path):
        for file in files:
            #print(f"Scanning: {file}")
            if file.endswith('.jpg'):
                print(f"Adding a file to the database: {file}")
                full_path = os.path.join(subdir, file)

                # Extracting the core filename without the .jpg extension and any aberrations
                clean_filename = re.sub(r'[^\w\s_,]', '', file[:-4])
                parts = clean_filename.split('_')

                # Validate that we have enough parts to construct the IDs
                if len(parts) < 11:
                    print(f"Skipping file due to unexpected format: {file}")
                    continue

                # Construct IDs and retrieve metadata
                recording_id = f"{parts[0]}_{parts[1]}_{parts[2]}"
                video_file_id = f"{recording_id}_{parts[3]}_{parts[4]}_{parts[5]}"

                # Data validation and conversion to integers
                try:
                    frame_no = int(parts[6])
                    visit_no = int(parts[7])
                    crop_no = int(parts[8])
                    x1, y1 = map(int, parts[9].split(','))
                    x2, y2 = map(int, parts[10].split(','))
                except ValueError as e:
                    print(f"Skipping file due to invalid metadata: {file}, Error: {e}")
                    continue

                label_path = None
                if "visitor" in subdir:
                    label_path = os.path.join(subdir, file.replace('.jpg', '.txt'))

                batch_data.append((recording_id, video_file_id, frame_no, visit_no, crop_no, x1, y1, x2, y2, full_path, label_path))

                count += 1
                if count >= batch_size:
                    cursor.executemany("INSERT INTO metadata (recording_id, video_file_id, frame_no, visit_no, crop_no, x1, y1, x2, y2, full_path, label_path) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", batch_data)
                    conn.commit()
                    batch_data = []
                    count = 0

    if batch_data:
        cursor.executemany("INSERT INTO metadata (recording_id, video_file_id, frame_no, visit_no, crop_no, x1, y1, x2, y2, full_path, label_path) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", batch_data)
        conn.commit()

    conn.close()


if __name__ == '__main__':
    import sys
    create_database()
    app = QApplication(sys.argv)
    ex = ICDM()
    sys.exit(app.exec_())