from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog, QTreeWidget, QVBoxLayout, QPushButton, QTreeView, QFileSystemModel, QTreeWidgetItem, QMenu, QAction, QInputDialog
from PyQt5.QtCore import QDir
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtCore import QUrl
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtGui import QPixmap, QPainter, QColor, QPen, QImage
from PyQt5.QtWidgets import QDialog, QLabel, QVBoxLayout, QHBoxLayout, QToolBar
from PyQt5.QtWidgets import QMainWindow, QToolBar, QVBoxLayout, QTreeView, QPushButton, QWidget
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QLabel, QProgressBar
from PyQt5.QtWidgets import QMainWindow, QDockWidget, QSplitter, QGraphicsView, QGraphicsScene, QTextEdit, QVBoxLayout, QWidget
from pathlib import Path
import shutil
import PyQt5
import os
import sqlite3
import re
pyqt = os.path.dirname(PyQt5.__file__)
os.environ['QT_PLUGIN_PATH'] = os.path.join(pyqt, "Qt5/plugins")

from PyQt5.QtCore import QSortFilterProxyModel, Qt
from PyQt5.QtCore import QThread, pyqtSignal
from pathlib import Path
import random
from pathlib import Path

from PyQt5.QtWidgets import QDialog, QFormLayout, QLineEdit, QDialogButtonBox


from PyQt5.QtWidgets import QDialog, QFormLayout, QLineEdit, QDialogButtonBox, QSlider, QLabel, QComboBox
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDialog, QFormLayout, QLineEdit, QDialogButtonBox, QSlider, QLabel, QComboBox, QTimeEdit
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QTimeEdit
from PyQt5.QtCore import QTime

class ExportDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Export Options")

        layout = QFormLayout()

        # Dataset Size Input
        self.datasetSizeInput = QLineEdit()
        layout.addRow("Dataset Size", self.datasetSizeInput)

        # Empty/Visitor Ratio Slider
        self.emptyVisitorRatioSlider = QSlider(Qt.Horizontal)
        self.emptyVisitorRatioSlider.setMinimum(0)
        self.emptyVisitorRatioSlider.setMaximum(100)
        self.emptyVisitorRatioSlider.setValue(50)
        self.emptyVisitorRatioLabel = QLabel(f"Visitor: 50%, Empty: 50%")
        layout.addRow("Empty/Visitor Ratio", self.emptyVisitorRatioSlider)
        layout.addRow("", self.emptyVisitorRatioLabel)

        # Day/Night Ratio Slider
        self.dayNightRatioSlider = QSlider(Qt.Horizontal)
        self.dayNightRatioSlider.setMinimum(0)
        self.dayNightRatioSlider.setMaximum(100)
        self.dayNightRatioSlider.setValue(50)
        self.dayNightRatioLabel = QLabel(f"Day: 50%, Night: 50%")
        layout.addRow("Day/Night Ratio", self.dayNightRatioSlider)
        layout.addRow("", self.dayNightRatioLabel)

        # Day Begin/End Time
        self.dayBeginInput = QTimeEdit(QTime.currentTime())
        self.dayEndInput = QTimeEdit(QTime.currentTime())
        layout.addRow("Day Begins At", self.dayBeginInput)
        layout.addRow("Day Ends At", self.dayEndInput)

        # Method Selection Dropdown
        self.methodDropdown = QComboBox()
        self.methodDropdown.addItems(["Method 1", "Method 2", "Method 3"])
        layout.addRow("Select Method", self.methodDropdown)

        # Button Box
        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)
        layout.addRow(buttonBox)

        # Update Labels on Slider Value Change
        self.emptyVisitorRatioSlider.valueChanged.connect(self.update_emptyVisitorRatioLabel)
        self.dayNightRatioSlider.valueChanged.connect(self.update_dayNightRatioLabel)

        self.setLayout(layout)

    def update_emptyVisitorRatioLabel(self, value):
        self.emptyVisitorRatioLabel.setText(f"Visitor: {value}%, Empty: {100-value}%")

    def update_dayNightRatioLabel(self, value):
        self.dayNightRatioLabel.setText(f"Day: {value}%, Night: {100-value}%")

class DatasetExporter(QThread):
    progress_signal = pyqtSignal(int)
    progress_total_signal = pyqtSignal(int)
    indeterminate_progress_signal = pyqtSignal(bool)

    def __init__(self, dataset_size, empty_visitor_ratio, root_folder_path, destination_folder_path, database_path: str = None):
        QThread.__init__(self)
        self.dataset_size = dataset_size
        self.empty_visitor_ratio = empty_visitor_ratio
        self.root_folder_path = root_folder_path
        self.destination_folder_path = destination_folder_path
        self.database_path = database_path

    def get_files_from_database(self):

        # Connect to the database
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()

        # Create empty lists to hold the file paths
        empty_files = []
        visitor_files = []

        # Query to get files that do not have a label (label_path is NULL)
        cursor.execute("SELECT full_path FROM metadata WHERE label_path IS NULL")
        empty_result = cursor.fetchall()

        # Query to get files that do have a label (label_path is NOT NULL)
        cursor.execute("SELECT full_path FROM metadata WHERE label_path IS NOT NULL")
        visitor_result = cursor.fetchall()

        # Close the database connection
        conn.close()

        # Extract the file paths from the query results and populate the lists
        empty_files = [row[0] for row in empty_result]
        visitor_files = [row[0] for row in visitor_result]

        return empty_files, visitor_files

    def get_files_default(self):

        # Create empty lists to hold the file paths
        empty_files = []
        visitor_files = []

        for subdir, _, files in os.walk(self.root_folder_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    full_path = os.path.join(subdir, file)
                    txt_path = os.path.join(subdir, file.rsplit('.', 1)[0] + '.txt')

                    if os.path.exists(txt_path):
                        visitor_files.append(full_path)
                    else:
                        empty_files.append(full_path)

        return empty_files, visitor_files

    def run(self):

        # Scan subfolders and separate filenames into "empty" and "visitor" lists
        empty_files = []
        visitor_files = []

        self.indeterminate_progress_signal.emit(True)

        if not os.path.isfile(self.database_path):
            empty_files, visitor_files = self.get_files_default()
        elif self.database_path.lower().endswith(".db"):
            empty_files, visitor_files = self.get_files_from_database()

        total_files = self.dataset_size
        progress = int(total_files*0.1)
        self.progress_total_signal.emit(int(total_files+progress))
        self.progress_signal.emit(progress)

        # Shuffle files
        random.shuffle(empty_files)
        random.shuffle(visitor_files)

        # Calculate the number of "empty" and "visitor" images to include
        num_empty = int(self.dataset_size * self.empty_visitor_ratio / (1 + self.empty_visitor_ratio))
        num_visitor = self.dataset_size - num_empty

        # Select files
        selected_empty = empty_files[:num_empty]
        selected_visitor = visitor_files[:num_visitor]

        # Copy files
        # Create subfolders in the destination folder
        empty_folder = os.path.join(self.destination_folder_path, "empty")
        visitor_folder = os.path.join(self.destination_folder_path, "visitor")
        os.makedirs(empty_folder, exist_ok=True)
        os.makedirs(visitor_folder, exist_ok=True)

        # Copy selected empty files
        for file in selected_empty:
            file_name = os.path.basename(file)
            source = file
            destination = os.path.join(empty_folder, file_name)
            shutil.copy(source, destination)

            # Update progress
            progress += 1
            self.progress_signal.emit(progress)


        # Copy selected visitor files
        for file in selected_visitor:
            file_name = os.path.basename(file)
            source = file
            destination = os.path.join(visitor_folder, file_name)
            shutil.copy(source, destination)

            # Move also the txt
            source = file.rsplit('.', 1)[0] + '.txt'
            destination = os.path.join(visitor_folder, os.path.basename(source))
            shutil.copy(source, destination)

            # Update progress
            progress += 1
            self.progress_signal.emit(progress)

class DatabasePopulator(QThread):
    progress_signal = pyqtSignal(int)
    progress_total_signal = pyqtSignal(int)

    def __init__(self, root_path, batch_size, database_path):
        QThread.__init__(self)
        self.root_path = root_path
        self.batch_size = batch_size
        self.database_path = database_path

    def run(self):
        print("Creating database...")
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()

        batch_data = []
        count = 0

        # Set the maximum for the progressbar
        total_files = sum(1 for _ in Path(self.root_path).rglob('*'))
        self.progress_total_signal.emit(total_files)

        progress_counter = 0
        # Process the files
        for subdir, _, files in os.walk(self.root_path):
            progress_counter += 1
            for i, file in enumerate(files):
                # print(f"Scanning: {file}")
                progress_counter += 1
                self.progress_signal.emit(progress_counter)  # Or however you wish to compute progress.
                if file.endswith('.jpg'):
                    print(f"Adding a file to the database: {file}")
                    full_path = os.path.normpath(os.path.join(subdir, file)).replace(r"\\", r"/")

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

                    batch_data.append((recording_id, video_file_id, frame_no, visit_no, crop_no, x1, y1, x2, y2,
                                       full_path, label_path))

                    count += 1
                    if count >= self.batch_size:
                        cursor.executemany(
                            "INSERT INTO metadata (recording_id, video_file_id, frame_no, visit_no, crop_no, x1, y1, x2, y2, full_path, label_path) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            batch_data)
                        conn.commit()
                        batch_data = []
                        count = 0

        if batch_data:
            cursor.executemany(
                "INSERT INTO metadata (recording_id, video_file_id, frame_no, visit_no, crop_no, x1, y1, x2, y2, full_path, label_path) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                batch_data)
            conn.commit()

        self.progress_signal.emit(total_files)  # Or however you wish to compute progress
        conn.close()
        self.progress_signal.emit(0)

class FileCounter(QThread):
    started_counting = pyqtSignal(str, object)
    finished_counting = pyqtSignal(int, int, int, object)

    def __init__(self, folder_path, label):
        QThread.__init__(self)
        self.folder_path = folder_path
        self.label = label

    def run(self):
        self.started_counting.emit("File count in progress...", self.label)
        path = Path(self.folder_path)
        total_files = sum(1 for _ in path.rglob('*'))
        image_files = sum(1 for _ in path.rglob('*.png')) + sum(1 for _ in path.rglob('*.jpg')) + sum(
            1 for _ in path.rglob('*.jpeg'))
        label_files = sum(1 for _ in path.rglob('*.txt'))

        self.finished_counting.emit(total_files, image_files, label_files, self.label)

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

class CustomDockWidget(QDockWidget):
    def __init__(self, graphics_view, pixmap_item, parent=None):
        super(CustomDockWidget, self).__init__(parent)
        self.graphics_view = graphics_view
        self.pixmap_item = pixmap_item

    def resizeEvent(self, event):
        if self.pixmap_item:
            new_size = self.graphics_view.size()
            scale_factor = min(new_size.width() / self.pixmap_item.pixmap().width(),
                               new_size.height() / self.pixmap_item.pixmap().height())
            self.pixmap_item.setScale(scale_factor)

class ICDM(QMainWindow):
    def __init__(self):
        super(ICDM, self).__init__()

        # Init variables
        self.file_counter = None
        self.root_path = None
        self.previewLabel = None
        self.pixmap_item = None
        self.database_path = ""

        # Build the GUI
        self.initialize_UI()

    def initialize_UI(self):

        # Create central widget to contain layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout()
        h_layout = QHBoxLayout()

        # Initialize the toolbar
        toolbar = QToolBar()
        self.addToolBar(toolbar)

        self.open_button = QPushButton('Open Folder', self)
        self.open_button.clicked.connect(self.show_open_folder_dialog)
        toolbar.addWidget(self.open_button)

        self.db_button = QPushButton('Generate Database', self)
        self.db_button.clicked.connect(lambda: self.import_data(self.root_path))
        self.db_button.setEnabled(False)
        toolbar.addWidget(self.db_button)

        self.export_button = QPushButton('Export Dataset', self)
        self.export_button.clicked.connect(self.show_export_dialog)
        self.export_button.setEnabled(False)
        toolbar.addWidget(self.export_button)

        # Create TreeView for visitor folders
        self.tree_visitor = QTreeView(self)
        self.model_visitor = QFileSystemModel()
        self.proxyModel_visitor = FolderFilterProxyModel("visitor")
        self.proxyModel_visitor.setSourceModel(self.model_visitor)
        self.tree_visitor.setModel(self.proxyModel_visitor)

        self.visitor_file_count_label = QLabel("")

        # Vertical layout for visitor view related widgets
        visitor_layout = QVBoxLayout()
        visitor_layout.addWidget(self.tree_visitor)
        visitor_layout.addWidget(self.visitor_file_count_label)

        h_layout.addLayout(visitor_layout)

        # Connecting actions and events to the visitor view
        self.tree_visitor.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree_visitor.customContextMenuRequested.connect(lambda pos, tree=self.tree_visitor: self.open_context_menu(
            pos, tree))

        self.tree_visitor.doubleClicked.connect(lambda: self.item_double_clicked(self.tree_visitor))
        self.tree_visitor.clicked.connect(lambda: self.initiate_file_count(self.tree_visitor,
                                                                           self.visitor_file_count_label)) #TODO: This should connect to the clickedTree function and that should trigger functions based on whether it was a fodler or a file

        # Create TreeView for empty folders
        self.tree_empty = QTreeView(self)
        self.model_empty = QFileSystemModel()
        self.proxyModel_empty = FolderFilterProxyModel("empty")
        self.proxyModel_empty.setSourceModel(self.model_empty)
        self.tree_empty.setModel(self.proxyModel_empty)

        self.empty_file_count_label = QLabel("")

        # Vertical layout for empty view related widgets
        empty_layout = QVBoxLayout()
        empty_layout.addWidget(self.tree_empty)
        empty_layout.addWidget(self.empty_file_count_label)

        h_layout.addLayout(empty_layout)

        # Connecting actions and events to the empty view
        self.tree_empty.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree_empty.customContextMenuRequested.connect(lambda pos, tree=self.tree_empty: self.open_context_menu(pos,
                                                                                                                    tree))

        self.tree_empty.doubleClicked.connect(lambda: self.item_double_clicked(self.tree_empty))
        self.tree_empty.clicked.connect(lambda: self.item_clicked(self.tree_empty)) #TODO: This should connect to the clickedTree function and that should trigger functions based on whether it was a fodler or a file


        # Make the dockable part for editing

        # Initialize Splitter
        self.splitter = QSplitter(Qt.Horizontal)

        # Initialize QGraphicsView
        self.image_view = QGraphicsView()
        self.scene = QGraphicsScene()
        self.image_view.setScene(self.scene)

        # Initialize Dock Widget
        self.dock_widget = CustomDockWidget(self.image_view, self.pixmap_item)
        self.addDockWidget(Qt.TopDockWidgetArea, self.dock_widget)

        # Initialize QTextEdit
        self.text_edit = QTextEdit()

        # Add widgets to splitter
        self.splitter.addWidget(self.image_view)
        self.splitter.addWidget(self.text_edit)

        # Set the initial sizes to make the splitter handle appear at the center
        initial_size = self.width() // 2  # Assuming 'self' is the QMainWindow
        self.splitter.setSizes([initial_size, initial_size])

        # Create layout and central widget for dock_widget
        # layout_split_preview = QVBoxLayout()
        # layout_split_preview.addWidget(self.splitter)
        self.dock_widget.setWidget(self.splitter)

        # Connect signal
        self.text_edit.textChanged.connect(self.on_text_changed)

        # Create a common progress bar
        self.common_progress_bar = QProgressBar()

        # Add the main horizontal layout to the layout = central Widget
        layout.addLayout(h_layout)

        # add progressbar to the layout
        layout.addWidget(self.common_progress_bar)

        # Set the layout for the central widget
        self.centralWidget().setLayout(layout)

        # Set window properties and show
        self.setWindowTitle('ICDM - Insect Communities Dataset Manager')
        self.show()

    def show_export_dialog(self):

        visitor_percentage = 100

        dialog = ExportDialog()
        result = dialog.exec_()

        if result == QDialog.Accepted:
            dataset_size = int(dialog.datasetSizeInput.text())
            empty_visitor_ratio = dialog.emptyVisitorRatioSlider.value()
            day_night_ratio = dialog.dayNightRatioSlider.value()
            day_start_time = dialog.dayBeginInput.time().toString("HH:mm")
            day_end_time = dialog.dayEndInput.time().toString("HH:mm")
            selection_method = dialog.methodDropdown.currentText()

            print(
                f"Retrieved values: Dataset Size: {dataset_size}, Empty/Visitor Ratio: {visitor_percentage}, Day/Night Ratio: {day_night_ratio}, Day Start Time: {day_start_time}, Day End Time: {day_end_time}, Selection Method: {selection_method}")
            empty_visitor_ratio = (100 - visitor_percentage) / visitor_percentage

            self.perform_export(dataset_size, empty_visitor_ratio)

    def perform_export(self, dataset_size, empty_visitor_ratio):

        destination_folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if not destination_folder:
            return

        self.dataset_exporter = DatasetExporter(dataset_size, empty_visitor_ratio, self.root_path, destination_folder, "metadata.db")
        self.dataset_exporter.progress_total_signal.connect(self.set_progress)
        self.dataset_exporter.progress_signal.connect(self.update_progress)
        self.dataset_exporter.indeterminate_progress_signal.connect(self.set_progress_indeterminate)
        self.dataset_exporter.start()

    def update_progress(self, value):
        self.common_progress_bar.setValue(value)

    def set_progress(self, value):
        self.common_progress_bar.setRange(0, value)

    def set_progress_indeterminate(self, value):
        if value:
            self.common_progress_bar.setRange(0, 0)

    # Slot function to update the UI
    def update_tree_view_label(self, total_files, image_files, label_files, label):
        label.setText(f"Total Files: {total_files}, Image Files: {image_files}, Label Files: {label_files}")

    def load_tree_view_label(self, text, label):
        label.setText(text)

    # Function to initiate the counting
    def initiate_file_count(self, tree, label):
        index, model = self.get_tree_model_source_attributes(tree)
        folder_path = model.filePath(index)

        if os.path.isdir(folder_path):
            self.file_counter = FileCounter(folder_path, label)
            self.file_counter.finished_counting.connect(self.update_tree_view_label)
            self.file_counter.started_counting.connect(self.load_tree_view_label)
            self.file_counter.start()

    def show_open_folder_dialog(self):
        folder_path = QFileDialog.getExistingDirectory(self, 'Select Folder')
        if folder_path:
            self.set_tree_roots(folder_path)
            self.root_path = folder_path
            self.database_path = f'{os.path.basename(folder_path)}.db'
            print(self.database_path)
            self.db_button.setEnabled(True)
            self.export_button.setEnabled(True)

    def set_tree_roots(self, folder_path):
        self.model_visitor.setRootPath(folder_path)
        self.tree_visitor.setRootIndex(self.proxyModel_visitor.mapFromSource(self.model_visitor.index(folder_path)))

        self.model_empty.setRootPath(folder_path)
        self.tree_empty.setRootIndex(self.proxyModel_empty.mapFromSource(self.model_empty.index(folder_path)))

        self.tree_visitor.setSortingEnabled(True)
        self.tree_empty.setSortingEnabled(True)

        total_width = self.tree_empty.width()
        self.tree_empty.setColumnWidth(0, int(total_width * 0.6))  # 60% of total width

        total_width = self.tree_visitor.width()
        self.tree_visitor.setColumnWidth(0, int(total_width * 0.6))  # 60% of total width

    def import_data(self, folder_path):

        # Create the database
        self.create_database(folder_path)

        # Initiate
        self.database_populator = DatabasePopulator(root_path=folder_path, batch_size=50, database_path=self.database_path)
        self.database_populator.progress_total_signal.connect(self.set_progress)
        self.database_populator.progress_signal.connect(self.update_progress)

        # Populate database and update progress
        self.database_populator.start()

    def open_context_menu(self, position, tree):
        menu = QMenu()
        index = tree.currentIndex()

        # This line is important: we go back to the source model
        source_index = tree.model().mapToSource(index)
        source_model = tree.model().sourceModel()

        # Use the source model to get the file path
        path = source_model.filePath(source_index)

        if path.lower().endswith(('.png', '.jpg', '.jpeg')):
            preview_action = QAction("Preview Image", self)
            preview_action.triggered.connect(lambda: self.preview_image(path))
            menu.addAction(preview_action)

        rename_action = QAction('Rename', self)
        rename_action.triggered.connect(lambda: self.rename_item(tree))
        menu.addAction(rename_action)

        delete_action = QAction('Delete', self)
        delete_action.triggered.connect(lambda: self.delete_item(tree))
        menu.addAction(delete_action)

        open_action = QAction('Open', self)
        open_action.triggered.connect(lambda: self.open_item(tree))
        menu.addAction(open_action)

        if os.path.isdir(path):
            set_root_action = QAction('Set as Root', self)
            set_root_action.triggered.connect(lambda: self.set_as_root(tree))
            menu.addAction(set_root_action)

        menu.exec_(tree.viewport().mapToGlobal(position))

    def item_clicked(self, tree):

        # Get the string
        filter_string = tree.model().filter_string
        print(filter_string)

        # get the source attributes
        index, model = self.get_tree_model_source_attributes(tree)

        file_path = model.filePath(index)  # Use the source index to get the file path
        if os.path.isdir(file_path):  # Check if the double-clicked item is a .jpg image
            label = self.empty_file_count_label if filter_string == "empty" else self.visitor_file_count_label
            self.initiate_file_count(tree, label)
        elif file_path.lower().endswith(('.png', '.jpg', '.jpeg')) and os.path.isfile(self.database_path):

            self.current_metadata = self.load_metadata_from_db(file_path)

            image_path = file_path
            text_path = self.current_metadata['label_path']

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
            label_path = text_path
            if os.path.exists(label_path):
                lines = []
                with open(label_path, 'r') as file:
                    for line in file:
                        lines.append(line)
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

            bbox_str = "".join(lines)
            # End painting
            painter.end()

            # Display the image with rectangle
            pixmap = QPixmap.fromImage(img)

            self.scene.clear()
            self.pixmap_item = self.scene.addPixmap(pixmap)
            self.dock_widget.pixmap_item = self.pixmap_item

            # Fit the view to the pixmap's bounding rectangle
            new_size = self.dock_widget.graphics_view.size()
            scale_factor = min(new_size.width() / self.pixmap_item.pixmap().width(),
                               new_size.height() / self.pixmap_item.pixmap().height())
            self.pixmap_item.setScale(scale_factor)

            # Load and display the text
            self.text_edit.setPlainText(bbox_str)



    # def resizeEvent(self, event):
    #     if self.pixmap_item:
    #         new_size = self.image_view.size()
    #         scale_factor = min(new_size.width() / self.pixmap_item.pixmap().width(),
    #                            new_size.height() / self.pixmap_item.pixmap().height())
    #         self.pixmap_item.setScale(scale_factor)

    def load_metadata_from_db(self, file_path):
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()

        file_path = os.path.normpath(file_path).replace(r"\\", r"/")
        print(file_path)

        query = "SELECT * FROM metadata WHERE full_path=?"  # Using 'full_path' to match your database schema
        cursor.execute(query, (file_path,))

        row = cursor.fetchone()
        print(row)
        if row:
            # Map the database row to a metadata dictionary
            metadata = {
                'id': row[0],
                'recording_id': row[1],
                'video_file_id': row[2],
                'frame_no': row[3],
                'visit_no': row[4],
                'crop_no': row[5],
                'x1': row[6],
                'y1': row[7],
                'x2': row[8],
                'y2': row[9],
                'full_path': row[10],
                'label_path': row[11]
            }
            return metadata
        conn.close()

    def on_text_changed(self):

        # Read the new text content from QTextEdit
        text_content = self.text_edit.toPlainText()

        # Update the corresponding .txt file
        txt_path = self.current_metadata['label_path']
        if txt_path:
            with open(txt_path, 'w') as f:
                    f.write(text_content)

    def item_double_clicked(self, tree):

        index, model = self.get_tree_model_source_attributes(tree)

        file_path = model.filePath(index)  # Use the source index to get the file path
        if file_path.lower().endswith('.jpg'):  # Check if the double-clicked item is a .jpg image
            self.preview_image(file_path)

    def preview_image(self, image_path):
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

    def get_tree_model_source_attributes(self, tree):

        # Get index and model for the specific tree
        index = tree.currentIndex()
        source_index = tree.model().mapToSource(index)
        source_model = tree.model().sourceModel()

        return source_index, source_model


    def rename_item(self, tree):

        index, model = self.get_tree_model_source_attributes(tree)

        if index.isValid():
            old_name = model.fileName(index)
            path = model.filePath(index)
            directory = os.path.dirname(path)

            new_name, ok = QInputDialog.getText(self, 'Rename', f'Rename {old_name} to:', text=old_name)
            if ok and new_name:
                new_path = os.path.join(directory, new_name)
                os.rename(path, new_path)

    def delete_item(self, tree):

        index, model = self.get_tree_model_source_attributes(tree)

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

    def open_item(self, tree):

        index, model = self.get_tree_model_source_attributes(tree)

        if index.isValid():
            path = model.filePath(index)
            if os.path.isfile(path):
                QDesktopServices.openUrl(QUrl(f"file:///{path}", QUrl.TolerantMode))

    def set_as_root(self, tree):

        index, model = self.get_tree_model_source_attributes(tree)
        folder_path = model.filePath(index)

        if os.path.isdir(folder_path):
            model.setRootPath(folder_path)
            tree.setRootIndex(tree.model().mapFromSource(model.index(folder_path)))
        else:
            QMessageBox.warning(self, "Warning", "Selected item is not a directory.")

    def create_database(self, folder_path):
        conn = sqlite3.connect(self.database_path)
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


if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    ex = ICDM()
    sys.exit(app.exec_())