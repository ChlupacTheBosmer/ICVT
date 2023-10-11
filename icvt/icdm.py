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
from PyQt5.QtWidgets import QMainWindow, QDockWidget, QSplitter, QGraphicsView, QGraphicsScene, QTextEdit, QVBoxLayout, QWidget, QTableView
from PyQt5.QtSql import QSqlTableModel
from pathlib import Path
import shutil
import PyQt5
import os
import sqlite3
import re
import datetime
from collections import defaultdict
pyqt = os.path.dirname(PyQt5.__file__)
os.environ['QT_PLUGIN_PATH'] = os.path.join(pyqt, "Qt5/plugins")

from PyQt5.QtCore import QSortFilterProxyModel, Qt
from PyQt5.QtCore import QThread, pyqtSignal
from pathlib import Path
import random
from pathlib import Path
import hashlib

from PyQt5.QtWidgets import QDialog, QFormLayout, QLineEdit, QDialogButtonBox


from PyQt5.QtWidgets import QDialog, QFormLayout, QLineEdit, QDialogButtonBox, QSlider, QLabel, QComboBox
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDialog, QFormLayout, QLineEdit, QDialogButtonBox, QSlider, QLabel, QComboBox, QTimeEdit
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QTimeEdit
from PyQt5.QtCore import QTime
from PyQt5.QtWidgets import QApplication, QVBoxLayout, QSizePolicy, QWidget, QPushButton, QDialog, QTabWidget
from PyQt5.QtSql import QSqlDatabase, QSqlTableModel
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

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
    export_database_created_signal = pyqtSignal(str)

    def __init__(self, dataset_size, empty_visitor_ratio, day_night_ratio, day_start_time, day_end_time, root_folder_path, destination_folder_path, database_path: str = None):
        QThread.__init__(self)
        self.dataset_size = dataset_size
        self.daytime_nighttime_ratio = day_night_ratio
        self.daytime_start = day_start_time
        self.daytime_end = day_end_time
        self.empty_visitor_ratio = empty_visitor_ratio
        self.root_folder_path = root_folder_path
        self.destination_folder_path = destination_folder_path
        self.database_path = database_path

    def update_export_database(self):
        # Modify database path by adding '_export' before '.db'
        old_database_path = self.database_path
        new_database_path = os.path.join(os.path.dirname(self.database_path),
                                         os.path.splitext(os.path.basename(self.database_path))[0] + '_export.db')
        # Copy database
        shutil.copy2(old_database_path, new_database_path)
        self.database_path = new_database_path

        # Modify the database
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()

        # Add new columns
        cursor.execute("ALTER TABLE metadata ADD COLUMN time TEXT")
        cursor.execute("ALTER TABLE metadata ADD COLUMN parent_folder TEXT")

        conn.commit()
        cursor.execute("SELECT id, video_file_id, full_path FROM metadata")

        for row in cursor.fetchall():
            id_, video_file_id, full_path = row
            time = self.extract_time_from_video_file_id(video_file_id)
            parent_folder = self.extract_parent_folder_from_full_path(full_path)

            cursor.execute("UPDATE metadata SET time = ?, parent_folder = ? WHERE id = ?", (time, parent_folder, id_))

        conn.commit()
        conn.close()

    def extract_time_from_video_file_id(self, video_file_id: str) -> str:
        # Extract time using regular expression
        time_match = re.search(r'_(\d{2}_\d{2})', video_file_id)
        if time_match:
            return time_match.group(1).replace('_', ':')  # Replace underscore with colon
        return None  # Return None if the pattern is not found

    def extract_parent_folder_from_full_path(self, full_path: str) -> str:
        # Initialize folder_path with the parent directory
        folder_path = os.path.dirname(full_path)

        while True:
            # Extract folder name from the folder path
            folder_name = os.path.basename(folder_path)

            # Check if folder name is neither "empty" nor "visitor"
            if folder_name.lower() not in ["empty", "visitor"]:
                return folder_name

            # Move up the directory tree
            folder_path = os.path.dirname(folder_path)

            # If we're at the root directory, break the loop
            if folder_path == os.path.dirname(folder_path):
                break

        return None  # Return None if no such folder is found

    def get_unique_parent_folder_count(self, database_path: str) -> int:
        # Connect to the database
        conn = sqlite3.connect(database_path)
        cursor = conn.cursor()

        # Execute SQL query to count unique parent_folder entries
        cursor.execute("SELECT COUNT(DISTINCT parent_folder) FROM metadata")

        # Fetch the result and extract the count
        count = cursor.fetchone()[0]

        # Close the database connection
        conn.close()

        return count

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

    def fetch_files_per_parent_folder(self, database_path: str, total_files_per_folder: int, empty_visitor_ratio: float,
                                      daytime_nighttime_ratio: float, daytime_start: str, daytime_end: str):
        # Connect to the database
        conn = sqlite3.connect(database_path)
        cursor = conn.cursor()

        # Convert daytime strings to datetime objects
        # daytime_start = datetime.datetime.strptime(daytime_start, '%H:%M').time()
        # daytime_end = datetime.datetime.strptime(daytime_end, '%H:%M').time()

        # Calculate counts per category
        num_empty_per_folder = int(total_files_per_folder * empty_visitor_ratio / (1 + empty_visitor_ratio))
        num_visitor_per_folder = total_files_per_folder - num_empty_per_folder

        num_daytime_per_type = int(num_empty_per_folder * daytime_nighttime_ratio / (1 + daytime_nighttime_ratio))
        num_nighttime_per_type = num_empty_per_folder - num_daytime_per_type

        # Initialize result dictionary
        files_by_folder = defaultdict(dict)

        cursor.execute("SELECT DISTINCT parent_folder FROM metadata")
        parent_folders = [row[0] for row in cursor.fetchall()]

        print(f"tot:{total_files_per_folder}, emp:{num_empty_per_folder}, vis:{num_visitor_per_folder}")
        for parent_folder in parent_folders:
            for label_condition, count_day, count_night, count in [("IS NULL", num_daytime_per_type, num_nighttime_per_type, num_empty_per_folder),
                                                            ("IS NOT NULL", num_daytime_per_type,
                                                             num_nighttime_per_type, num_visitor_per_folder)]:
                if False:
                    # Query for 'daytime' files
                    query = """
                        SELECT full_path
                        FROM metadata
                        WHERE parent_folder = ?
                        AND label_path {}
                        AND time(time) BETWEEN ? AND ?
                        LIMIT ?
                    """.format(label_condition)

                    cursor.execute(query, (parent_folder, daytime_start, daytime_end, count_day))
                    daytime_files = [row[0] for row in cursor.fetchall()]

                    # Query for 'nighttime' files
                    query = """
                        SELECT full_path
                        FROM metadata
                        WHERE parent_folder = ?
                        AND label_path {}
                        AND (time(time) < ? OR time(time) > ?)
                        LIMIT ?
                    """.format(label_condition)
                    cursor.execute(query, (parent_folder, daytime_start, daytime_end, count_night))
                    nighttime_files = [row[0] for row in cursor.fetchall()]

                    # Update result dictionary
                    type_label = 'empty' if label_condition == "IS NULL" else 'visitor'
                    files_by_folder[parent_folder][type_label] = {
                        'daytime': daytime_files,
                        'nighttime': nighttime_files
                    }
                else:
                    print("doing this")
                    # Fetch files ignoring day/night ratio
                    query = """
                                    SELECT full_path
                                    FROM metadata
                                    WHERE parent_folder = ?
                                    AND label_path {}
                                    LIMIT ?
                                """.format(label_condition)
                    print(count)
                    cursor.execute(query, (parent_folder, count))
                    all_files = [row[0] for row in cursor.fetchall()]

                    type_label = 'empty' if label_condition == "IS NULL" else 'visitor'
                    files_by_folder[parent_folder][type_label] = {
                        'all': all_files
                    }

        # Add a new colum nto the database
        cursor.execute("ALTER TABLE metadata ADD COLUMN chosen_for_export BOOLEAN DEFAULT 0")
        conn.commit()

        # Here, let's update the 'chosen_for_export' flag for the selected files.
        for parent_folder, types in files_by_folder.items():
            for type_label, time_dict in types.items():
                for file in time_dict.get('all', []):
                    update_query = "UPDATE metadata SET chosen_for_export = 1 WHERE full_path = ?"
                    cursor.execute(update_query, (file,))
        conn.commit()

        # Close the database connection
        conn.close()

        # Generate a unique hash (for example, from the current time)
        unique_hash = hashlib.sha1(str(datetime.datetime.now()).encode()).hexdigest()[:8]

        # Create a new database name with the unique hash
        new_database_path = f"{database_path.split('.db')[0]}_{unique_hash}.db"

        # Copy the current database to the new unique name
        shutil.copy(database_path, new_database_path)

        return files_by_folder

    def copy_files(self, result):

        # Setup progress tracking
        total_files = 0
        for parent_folder, types in result.items():
            for file_type, time_dict in types.items():
                total_files += len(time_dict['all'])
        progress = int(total_files * 0.1)
        self.progress_total_signal.emit(int(total_files + progress))
        self.progress_signal.emit(progress)

        # Create destination folders for each parent folder, if needed
        parent_folder_destination = self.destination_folder_path
        empty_folder = os.path.join(parent_folder_destination, 'empty')
        visitor_folder = os.path.join(parent_folder_destination, 'visitor')
        os.makedirs(empty_folder, exist_ok=True)
        os.makedirs(visitor_folder, exist_ok=True)

        # Loop through each parent folder
        for parent_folder, types_data in result.items():
            print("parent")
            # Copy selected empty files
            for file in types_data['empty']['all']:
                print(file)
                file_name = os.path.basename(file)
                source = file
                destination = os.path.join(empty_folder, file_name)
                shutil.copy(source, destination)

                # Update progress
                progress += 1
                self.progress_signal.emit(progress)

            # Copy selected visitor files
            for file in types_data['visitor']['all']:
                print(file)
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

    def inspect_time_distribution(self):

        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        cursor.execute("SELECT time FROM metadata WHERE chosen_for_export = 1")
        times = [row[0] for row in cursor.fetchall()]
        conn.close()

        import matplotlib.pyplot as plt

        # Convert times into a format that can be plotted, such as minutes since midnight
        minutes_since_midnight = [(int(time.split(':')[0]) * 60 + int(time.split(':')[1])) for time in times]

        plt.hist(minutes_since_midnight, bins=48)  # 48 bins for 30-minute intervals
        plt.xlabel('Minutes Since Midnight')
        plt.ylabel('Number of Images')
        plt.title('Distribution of Daytimes')
        plt.show()

    def run(self):

        # Scan subfolders and separate filenames into "empty" and "visitor" lists
        empty_files = []
        visitor_files = []

        self.indeterminate_progress_signal.emit(True)

        if not os.path.isfile(self.database_path):
            return
        elif self.database_path.lower().endswith(".db"):
            self.update_export_database()
            number_of_parent_folders = self.get_unique_parent_folder_count(self.database_path)
            print(f"parent_folders:{number_of_parent_folders}")
            files_per_parent_folder = self.dataset_size // number_of_parent_folders
            result = self.fetch_files_per_parent_folder(self.database_path, files_per_parent_folder, self.empty_visitor_ratio,
                                                   self.daytime_nighttime_ratio, self.daytime_start, self.daytime_end)
            print(result)
            self.copy_files(result)
            self.export_database_created_signal.emit(self.database_path)
            # empty_files, visitor_files = self.get_files_from_database()

        # total_files = self.dataset_size
        # progress = int(total_files*0.1)
        # self.progress_total_signal.emit(int(total_files+progress))
        # self.progress_signal.emit(progress)
        #
        # # Shuffle files
        # random.shuffle(empty_files)
        # random.shuffle(visitor_files)
        #
        # # Calculate the number of "empty" and "visitor" images to include
        # num_empty = int(self.dataset_size * self.empty_visitor_ratio / (1 + self.empty_visitor_ratio))
        # num_visitor = self.dataset_size - num_empty
        #
        # # Select files
        # selected_empty = empty_files[:num_empty]
        # selected_visitor = visitor_files[:num_visitor]
        #
        # # Copy files
        # # Create subfolders in the destination folder
        # empty_folder = os.path.join(self.destination_folder_path, "empty")
        # visitor_folder = os.path.join(self.destination_folder_path, "visitor")
        # os.makedirs(empty_folder, exist_ok=True)
        # os.makedirs(visitor_folder, exist_ok=True)
        #
        # # Copy selected empty files
        # for file in selected_empty:
        #     file_name = os.path.basename(file)
        #     source = file
        #     destination = os.path.join(empty_folder, file_name)
        #     shutil.copy(source, destination)
        #
        #     # Update progress
        #     progress += 1
        #     self.progress_signal.emit(progress)
        #
        #
        # # Copy selected visitor files
        # for file in selected_visitor:
        #     file_name = os.path.basename(file)
        #     source = file
        #     destination = os.path.join(visitor_folder, file_name)
        #     shutil.copy(source, destination)
        #
        #     # Move also the txt
        #     source = file.rsplit('.', 1)[0] + '.txt'
        #     destination = os.path.join(visitor_folder, os.path.basename(source))
        #     shutil.copy(source, destination)
        #
        #     # Update progress
        #     progress += 1
        #     self.progress_signal.emit(progress)

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

class CustomWidget(QWidget):
    def __init__(self, graphics_view, pixmap_item, parent=None):
        super(CustomWidget, self).__init__(parent)
        self.graphics_view = graphics_view
        self.pixmap_item = pixmap_item

        # Initialize layout and add the graphics_view
        layout = QVBoxLayout()
        layout.addWidget(self.graphics_view)
        self.setLayout(layout)

    def resizeEvent(self, event):
        if self.pixmap_item:
            new_size = self.graphics_view.size()
            scale_factor = min(new_size.width() / self.pixmap_item.pixmap().width(),
                               new_size.height() / self.pixmap_item.pixmap().height())
            self.pixmap_item.setScale(scale_factor)

class TimeDistributionPlot(FigureCanvas):
    def __init__(self, database_path, parent=None, width=5, height=4, dpi=100):
        self.database_path = database_path
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        if os.path.isfile(self.database_path):
            self.plot()

    def plot(self):
        # Clear the existing plot
        self.axes.clear()

        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        cursor.execute("SELECT time FROM metadata WHERE chosen_for_export = 1")
        times = [row[0] for row in cursor.fetchall()]
        conn.close()

        minutes_since_midnight = [(int(time.split(':')[0]) * 60 + int(time.split(':')[1])) for time in times]

        # Plot histogram
        self.axes.hist(minutes_since_midnight, bins=48, color='g')

        # Convert specific minutes since midnight back to time format for labeling
        tick_positions = list(range(0, 24 * 60, 60 * 3))  # Every 180 minutes (3 hours) from 0 to 24*60
        tick_labels = [f"{int(minute // 60):02d}:{int(minute % 60):02d}" for minute in tick_positions]

        # Set ticks and custom tick labels
        self.axes.set_xticks(tick_positions)
        self.axes.set_xticklabels(tick_labels, rotation=45)  # 45 degree rotation for better visibility

        # Set other axis labels and title
        self.axes.set_title('Distribution of Daytimes')
        self.axes.set_xlabel('Time of the Day')
        self.axes.set_ylabel('Number of Images')

        # Add this line to adjust layout to fit axis labels
        self.axes.figure.tight_layout()

        self.draw()

    def update_database(self, new_database_path):
        self.database_path = new_database_path
        self.plot()


class FolderDistributionPlot(FigureCanvas):
    def __init__(self, database_path, parent=None, width=5, height=4, dpi=100):
        self.database_path = database_path
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)

        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

        if os.path.isfile(self.database_path):
            self.plot()

    def get_folder_distribution(self):
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()

        # Use a CASE statement to classify based on label_path
        cursor.execute(
            """SELECT parent_folder, 
                CASE 
                    WHEN label_path IS NULL THEN 'empty' 
                    ELSE 'visitor' 
                END AS file_type,
                COUNT(*) 
            FROM metadata 
            WHERE chosen_for_export = 1 
            GROUP BY parent_folder, file_type""")

        raw_counts = cursor.fetchall()
        conn.close()

        folder_counts = {}
        for folder, file_type, count in raw_counts:
            if folder not in folder_counts:
                folder_counts[folder] = {'visitor': 0, 'empty': 0}
            folder_counts[folder][file_type] = count

        return folder_counts

    def plot(self):
        folder_counts = self.get_folder_distribution()
        self.axes.clear()

        folder_names = list(folder_counts.keys())
        visitor_counts = [folder_counts[name].get('visitor', 0) for name in folder_names]
        empty_counts = [folder_counts[name].get('empty', 0) for name in folder_names]

        # Create stacked bar chart
        self.axes.barh(folder_names, visitor_counts, label='Visitor Files', color='g')
        self.axes.barh(folder_names, empty_counts, left=visitor_counts, label='Empty Files', color='r')

        self.axes.set_xlabel('Number of Files')
        self.axes.set_title('File Distribution Among Folders')
        self.axes.legend()

        # To adjust layout to fit axis labels
        self.axes.figure.tight_layout()
        self.draw()

    def update_database(self, new_database_path):
        self.database_path = new_database_path
        self.plot()

class FrameProportionPieChart(FigureCanvas):
    def __init__(self, database_path, parent=None, width=5, height=4, dpi=100):
        self.database_path = database_path
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

        if os.path.isfile(self.database_path):
            self.plot()

    def plot(self):
        frame_proportions = self.get_frame_proportions()

        # Create pie chart
        labels = list(frame_proportions.keys())
        sizes = [frame_proportions[label] for label in labels]

        self.axes.clear()
        self.axes.pie(sizes, labels=labels, autopct='%1.1f%%', colors=("r","g"))
        self.axes.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        # Add this line to adjust layout to fit axis labels
        self.axes.figure.tight_layout()

        self.draw()

    def get_frame_proportions(self):
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()

        # Count empty frames that are chosen for export
        cursor.execute(
            "SELECT COUNT(*) FROM metadata WHERE label_path IS NULL AND chosen_for_export = 1")
        empty_frames = cursor.fetchone()[0]

        # Count visitor frames that are chosen for export
        cursor.execute(
            "SELECT COUNT(*) FROM metadata WHERE label_path IS NOT NULL AND chosen_for_export = 1")
        visitor_frames = cursor.fetchone()[0]

        conn.close()

        return {'Empty Frames': empty_frames, 'Visitor Frames': visitor_frames}

    def update_database(self, new_database_path):
        self.database_path = new_database_path
        self.plot()

class ICDM(QMainWindow):
    def __init__(self):
        super(ICDM, self).__init__()

        # Init variables
        self.file_counter = None
        self.root_path = None
        self.previewLabel = None
        self.pixmap_item = None
        self.database_path = ""
        self.export_database_path = ""

        # Build the GUI
        self.initialize_UI()

    def initialize_UI(self):

        # Create central widget to contain layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout()
        tab_widget = QTabWidget()
        main_layout.addWidget(tab_widget)
        vertical_layout = QVBoxLayout()
        horizontal_layout = QHBoxLayout()

        first_tab = QWidget()
        existing_layout = QVBoxLayout(first_tab)
        existing_layout.addLayout(vertical_layout)  # Your existing horizontal layout
        tab_widget.addTab(first_tab, "Raw Data View")

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

        horizontal_layout.addLayout(visitor_layout)

        # Connecting actions and events to the visitor view
        self.tree_visitor.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree_visitor.customContextMenuRequested.connect(lambda pos, tree=self.tree_visitor: self.open_context_menu(
            pos, tree))

        self.tree_visitor.doubleClicked.connect(lambda: self.item_double_clicked(self.tree_visitor))
        self.tree_visitor.clicked.connect(lambda: self.item_clicked(self.tree_visitor)) #TODO: This should connect to the clickedTree function and that should trigger functions based on whether it was a fodler or a file

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

        horizontal_layout.addLayout(empty_layout)

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

        # Initialize custom widget (instead of Dock Widget)
        self.custom_widget = CustomWidget(self.image_view, self.pixmap_item)

        # Initialize QTextEdit
        self.text_edit = QTextEdit()

        # Initialize container_widget
        text_edit_container_layout = QVBoxLayout()
        text_edit_container_widget = QWidget()
        text_edit_container_layout.addWidget(self.text_edit)
        text_edit_container_widget.setLayout(text_edit_container_layout)

        # Add widgets to splitter
        self.splitter.addWidget(self.custom_widget)  # Replace 'self.image_view' with 'self.custom_widget'
        self.splitter.addWidget(text_edit_container_widget)

        # Set the initial sizes to make the splitter handle appear at the center
        initial_size = self.width() // 2  # Assuming 'self' is the QMainWindow
        self.splitter.setSizes([initial_size, initial_size])

        # Connect signal
        self.text_edit.textChanged.connect(self.on_text_changed)

        # Create a common progress bar
        self.common_progress_bar = QProgressBar()

        # Create a vertical splitter
        self.vertical_splitter = QSplitter(Qt.Vertical)
        vertical_layout.addWidget(self.vertical_splitter)

        # Add the preview layout splitter to the main layout
        self.vertical_splitter.addWidget(self.splitter)

        # Add the main horizontal layout to the layout = central Widget
        # Create a container widget and set its layout
        container_widget = QWidget()
        container_widget.setLayout(horizontal_layout)
        self.vertical_splitter.addWidget(container_widget)

        # add progressbar to the layout
        main_layout.addWidget(self.common_progress_bar)

        # SECOND TAB
        new_tab = QWidget()
        new_layout = QVBoxLayout(new_tab)
        tab_widget.addTab(new_tab, "Dataset View")

        # Initialize main vertical splitter for the second tab
        main_vertical_splitter = QSplitter(Qt.Vertical)

        # Initialize first horizontal splitter
        horizontal_splitter_1 = QSplitter(Qt.Horizontal)
        self.time_plot = TimeDistributionPlot("", width=5, height=4, dpi=100)
        self.folder_plot = FolderDistributionPlot("", width=5, height=4, dpi=100)
        horizontal_splitter_1.addWidget(self.time_plot)
        horizontal_splitter_1.addWidget(self.folder_plot)

        # Initialize second horizontal splitter
        horizontal_splitter_2 = QSplitter(Qt.Horizontal)
        self.pie_chart = FrameProportionPieChart("", width=5, height=4, dpi=100)

        self.table_view = QTableView()
        model = QSqlTableModel()
        model.setTable('metadata')
        model.select()
        self.table_view.setModel(model)
        horizontal_splitter_2.addWidget(self.pie_chart)
        horizontal_splitter_2.addWidget(self.table_view)

        # Add horizontal splitters to main vertical splitter
        main_vertical_splitter.addWidget(horizontal_splitter_1)
        main_vertical_splitter.addWidget(horizontal_splitter_2)

        # Set the initial sizes to make the splitter handle appear at the center
        initial_size = self.height() // 2  # Assuming 'self' is the QMainWindow
        horizontal_splitter_1.setSizes([initial_size, initial_size])
        horizontal_splitter_2.setSizes([initial_size, initial_size])

        # Add main vertical splitter to second tab layout
        new_layout.addWidget(main_vertical_splitter)

        # Set the layout for the central widget
        self.centralWidget().setLayout(main_layout)

        # Set window properties and show
        self.setWindowTitle('ICDM - Insect Communities Dataset Manager')
        self.show()

    def update_database(self, new_database_path):
        # Step 1: Close the existing database connection
        QSqlDatabase.database().close()

        # Step 2: Open a new connection with the new database name
        db = QSqlDatabase.addDatabase("QSQLITE")
        db.setDatabaseName(new_database_path)
        ok = db.open()
        if not ok:
            print("Failed to open new database")  # Handle this more gracefully in your actual code
            return

        self.database_path = new_database_path  # Update the class attribute

        # Step 3: Reinitialize the QSqlTableModel with the new database
        new_model = QSqlTableModel()
        new_model.setTable('metadata')  # Assuming 'metadata' is your table name
        new_model.setFilter("chosen_for_export = 1")  # Optional: apply your filter
        new_model.select()

        # Step 4: Update the QTableView to use the new model
        self.table_view.setModel(new_model)  # Assuming `self.table_view` is your QTableView instance

    def show_export_dialog(self):

        dialog = ExportDialog()
        result = dialog.exec_()

        if result == QDialog.Accepted:
            dataset_size = int(dialog.datasetSizeInput.text())
            visitor_percentage = dialog.emptyVisitorRatioSlider.value()
            day_percentage = dialog.dayNightRatioSlider.value()
            day_start_time = dialog.dayBeginInput.time().toString("HH:mm")
            day_end_time = dialog.dayEndInput.time().toString("HH:mm")
            selection_method = dialog.methodDropdown.currentText()

            print(
                f"Retrieved values: Dataset Size: {dataset_size}, Empty/Visitor Ratio: {visitor_percentage}, Day/Night Ratio: {day_percentage}, Day Start Time: {day_start_time}, Day End Time: {day_end_time}, Selection Method: {selection_method}")
            empty_visitor_ratio = (100 - visitor_percentage) / visitor_percentage
            day_night_ratio = (100 - day_percentage) / day_percentage

            self.perform_export(dataset_size, empty_visitor_ratio, day_night_ratio, day_start_time, day_end_time)

    def perform_export(self, dataset_size, empty_visitor_ratio, day_night_ratio, day_start_time, day_end_time):

        destination_folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if not destination_folder:
            return

        self.dataset_exporter = DatasetExporter(dataset_size, empty_visitor_ratio, day_night_ratio, day_start_time, day_end_time, self.root_path, destination_folder, self.database_path)
        self.dataset_exporter.progress_total_signal.connect(self.set_progress)
        self.dataset_exporter.progress_signal.connect(self.update_progress)
        self.dataset_exporter.indeterminate_progress_signal.connect(self.set_progress_indeterminate)
        self.dataset_exporter.export_database_created_signal.connect(self.update_inspection_graphs)
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

    def update_inspection_graphs(self, database_path):
        self.export_database_path = database_path
        self.time_plot.update_database(self.export_database_path)
        self.folder_plot.update_database(self.export_database_path)
        self.pie_chart.update_database(self.export_database_path)
        self.update_database(self.export_database_path)


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
        bbox_str = ""

        # get the source attributes
        index, model = self.get_tree_model_source_attributes(tree)

        file_path = model.filePath(index)  # Use the source index to get the file path
        if os.path.isdir(file_path):  # Check if the double-clicked item is a .jpg image
            label = self.empty_file_count_label if filter_string == "empty" else self.visitor_file_count_label
            self.initiate_file_count(tree, label)
        elif file_path.lower().endswith(('.png', '.jpg', '.jpeg')) and os.path.isfile(self.database_path):

            self.current_metadata = self.load_metadata_from_db(file_path)

            image_path = file_path
            text_path = self.current_metadata['label_path'] if self.current_metadata['label_path'] is not None else ""

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
            self.custom_widget.pixmap_item = self.pixmap_item  # Changed from self.dock_widget.pixmap_item

            # Fit the view to the pixmap's bounding rectangle
            new_size = self.custom_widget.graphics_view.size()  # Changed from self.dock_widget.graphics_view.size()
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