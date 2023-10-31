from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import PyQt5
import concurrent.futures
import datetime
import hashlib
import os
import random
import re
import shutil
import sqlite3
import yaml
from PyQt5.QtCore import QSortFilterProxyModel, QThread, pyqtSignal, QSize, QUrl, Qt, QTime
from PyQt5.QtGui import QDesktopServices, QPixmap, QPainter, QPen, QImage, QFont, QIcon, QColor
from PyQt5.QtSql import QSqlDatabase, QSqlTableModel, QSqlQuery
from PyQt5.QtWidgets import (QApplication, QSizePolicy, QWidget, QPushButton, QTabWidget, QFrame, QDialog, QFormLayout,
                             QLineEdit, QSlider, QLabel, QTimeEdit, QComboBox, QDialogButtonBox,
                             QCheckBox, QVBoxLayout, QGroupBox, QFileDialog, QFileSystemModel, QMenu, QAction,
                             QInputDialog, QHBoxLayout, QProgressBar, QMainWindow, QDockWidget, QSplitter,
                             QGraphicsView, QGraphicsScene, QTextEdit, QTableView, QMessageBox, QToolBar, QTreeView, QTreeWidget)
from PyQt5.QtWidgets import QGroupBox, QFormLayout, QTreeWidgetItem
from PyQt5.QtCore import Qt, QSettings
import json
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QSizePolicy
import sqlite3
import os
import pandas as pd
import numpy as np

os.environ['QT_PLUGIN_PATH'] = os.path.join(os.path.dirname(PyQt5.__file__), "Qt5/plugins")

# DONE: Add an option to generate dataset so that it comprises as many visits as possible.
        # When selecting images from folders according to criteria make sure to select uniques (with limit) per visits.
# TODO: Incorporate the different methods of selecting the files: Random within folder, Per visit equal
# TODO: Add an option to somehow select which folders will be exclusive to val dataset so we can make sure to have data
#       in val that are not in the train dataset to correctly detect for overfitting.
# DONE: Add a diagnostic graph to see how are the flower morphotypes represented in the datasets.

class ExportDialog(QDialog):
    def __init__(self, main_database_path: str, details_database_path: str =None):
        super().__init__()

        #Set up connections
        self.main_database_path = main_database_path
        self.details_database_path = details_database_path
        self.conn_metadata = sqlite3.connect(self.main_database_path)
        self.conn_details = sqlite3.connect(self.details_database_path)

        self.setWindowTitle("Export Options")

        # Main layout
        mainLayout = QVBoxLayout()

        # Dataset structure settings
        datasetStructureGroup = QGroupBox("Dataset Structure")
        datasetStructureGroupLayout = QFormLayout()

        # Dataset Size Input
        self.datasetSizeInput = QLineEdit()
        datasetStructureGroupLayout.addRow("Dataset Size", self.datasetSizeInput)

        # Training/Validation Ratio Slider
        self.trainValRatioSlider = QSlider(Qt.Horizontal)
        self.trainValRatioSlider.setMinimum(0)
        self.trainValRatioSlider.setMaximum(100)
        self.trainValRatioSlider.setValue(80)
        self.trainValRatioLabel = QLabel(f"Train: 80%, Validation: 20%")
        datasetStructureGroupLayout.addRow("Train/Validation Ratio", self.trainValRatioSlider)
        datasetStructureGroupLayout.addRow("", self.trainValRatioLabel)

        datasetStructureGroup.setLayout(datasetStructureGroupLayout)
        mainLayout.addWidget(datasetStructureGroup)

        # Dataset composition settings
        datasetCompositionGroup = QGroupBox("Dataset Composition")
        datasetCompositionGroupLayout = QFormLayout()

        # Empty/Visitor Ratio Slider
        self.emptyVisitorRatioSlider = QSlider(Qt.Horizontal)
        self.emptyVisitorRatioSlider.setMinimum(0)
        self.emptyVisitorRatioSlider.setMaximum(100)
        self.emptyVisitorRatioSlider.setValue(80)
        self.emptyVisitorRatioLabel = QLabel(f"Visitor: 80%, Empty: 20%")
        datasetCompositionGroupLayout.addRow("Empty/Visitor Ratio", self.emptyVisitorRatioSlider)
        datasetCompositionGroupLayout.addRow("", self.emptyVisitorRatioLabel)

        # Use Day/Night Ratio Checkbox
        self.useDayNightCheckbox = QCheckBox("Use Day/Night Ratio")
        self.useDayNightCheckbox.setChecked(False)
        datasetCompositionGroupLayout.addRow(self.useDayNightCheckbox)

        # Day/Night Ratio Slider
        self.dayNightRatioSlider = QSlider(Qt.Horizontal)
        self.dayNightRatioSlider.setMinimum(0)
        self.dayNightRatioSlider.setMaximum(100)
        self.dayNightRatioSlider.setValue(50)
        self.dayNightRatioSlider.setEnabled(False)
        self.dayNightRatioLabel = QLabel(f"Day: 50%, Night: 50%")
        datasetCompositionGroupLayout.addRow("Day/Night Ratio", self.dayNightRatioSlider)
        datasetCompositionGroupLayout.addRow("", self.dayNightRatioLabel)

        # Day Begin/End Time
        self.dayBeginInput = QTimeEdit(QTime(12, 0))
        self.dayBeginInput.setEnabled(False)
        self.dayEndInput = QTimeEdit(QTime(0, 0))
        self.dayEndInput.setEnabled(False)
        datasetCompositionGroupLayout.addRow("Day Begins At", self.dayBeginInput)
        datasetCompositionGroupLayout.addRow("Day Ends At", self.dayEndInput)

        datasetCompositionGroup.setLayout(datasetCompositionGroupLayout)
        mainLayout.addWidget(datasetCompositionGroup)

        # Dataset Selection Method settings
        datasetSelectionGroup = QGroupBox("Dataset Selection Method")
        datasetSelectionGroupLayout = QFormLayout()

        # Method Selection Dropdown
        self.methodDropdown = QComboBox()
        self.methodDropdown.addItems(["Method 1", "Method 2", "Method 3"])
        datasetSelectionGroupLayout.addRow("Select Method", self.methodDropdown)

        datasetSelectionGroup.setLayout(datasetSelectionGroupLayout)
        mainLayout.addWidget(datasetSelectionGroup)

        # Dataset description file settings
        datasetDescriptionGroup = QGroupBox("Dataset Description File")
        datasetDescriptionGroupLayout = QFormLayout()

        # Use Day/Night Ratio Checkbox
        self.generateYamlCheckbox = QCheckBox("Generate YAML file")
        self.generateYamlCheckbox.setChecked(True)
        datasetDescriptionGroupLayout.addRow(self.generateYamlCheckbox)

        datasetDescriptionGroup.setLayout(datasetDescriptionGroupLayout)
        mainLayout.addWidget(datasetDescriptionGroup)

        # Update Labels on Slider Value Change
        self.emptyVisitorRatioSlider.valueChanged.connect(self.update_emptyVisitorRatioLabel)
        self.dayNightRatioSlider.valueChanged.connect(self.update_dayNightRatioLabel)
        self.trainValRatioSlider.valueChanged.connect(self.update_trainValRatioLabel)

        # Enable/disable day-night related controls based on checkbox
        self.useDayNightCheckbox.toggled.connect(self.toggle_dayNightSettings)

        # Folder Selection settings
        folderSelectionGroup = QGroupBox("Folder Selection")
        folderSelectionGroupLayout = QVBoxLayout()
        trees_layout = QHBoxLayout()

        # Initialize tree widgets for train and validation sets
        self.train_tree = QTreeWidget()
        self.train_tree.setColumnCount(4)
        self.train_tree.setHeaderLabels(["Recording ID", "Family", "Color", "Shape"])
        self.train_tree.setSortingEnabled(True)

        self.val_tree = QTreeWidget()
        self.val_tree.setColumnCount(4)
        self.val_tree.setHeaderLabels(["Recording ID", "Family", "Color", "Shape"])
        self.val_tree.setSortingEnabled(True)

        # Initialize labels for tree views
        train_tree_label = QLabel("Training Dataset")
        val_tree_label = QLabel("Validation Dataset")

        # Initialize vertical layouts for tree views and labels
        train_tree_layout = QVBoxLayout()
        train_tree_layout.addWidget(train_tree_label)
        train_tree_layout.addWidget(self.train_tree)

        val_tree_layout = QVBoxLayout()
        val_tree_layout.addWidget(val_tree_label)
        val_tree_layout.addWidget(self.val_tree)

        # Add vertical layouts to the main horizontal layout
        trees_layout.addLayout(train_tree_layout)
        trees_layout.addLayout(val_tree_layout)

        # Initialize buttons
        to_val_button = QPushButton("To Validation >>")
        to_train_button = QPushButton("<< To Training")

        to_val_button.clicked.connect(self.move_to_val)
        to_train_button.clicked.connect(self.move_to_train)

        save_button = QPushButton("Save Configuration")
        load_button = QPushButton("Load Configuration")

        save_button.clicked.connect(self.save_config)
        load_button.clicked.connect(self.load_config)

        button_layout = QHBoxLayout()
        button_layout.addWidget(to_val_button)
        button_layout.addWidget(save_button)
        button_layout.addWidget(load_button)
        button_layout.addWidget(to_train_button)

        folderSelectionGroupLayout.addLayout(trees_layout)
        folderSelectionGroupLayout.addLayout(button_layout)

        folderSelectionGroup.setLayout(folderSelectionGroupLayout)
        mainLayout.addWidget(folderSelectionGroup)

        # Create a new QGroupBox
        description_group_box = QGroupBox("Description")
        description_layout = QVBoxLayout()

        # Create a QTextEdit widget for multi-line text input
        self.description_text_edit = QTextEdit()
        self.description_text_edit.setFixedHeight(50)
        description_layout.addWidget(self.description_text_edit)
        description_group_box.setLayout(description_layout)
        mainLayout.addWidget(description_group_box)

        # Button Box
        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)
        mainLayout.addWidget(buttonBox)

        self.setLayout(mainLayout)

        self.populate_trees()

    def update_emptyVisitorRatioLabel(self, value):
        self.emptyVisitorRatioLabel.setText(f"Visitor: {value}%, Empty: {100 - value}%")

    def update_dayNightRatioLabel(self, value):
        self.dayNightRatioLabel.setText(f"Day: {value}%, Night: {100 - value}%")

    def update_trainValRatioLabel(self, value):
        self.trainValRatioLabel.setText(f"Train: {value}%, Validation: {100 - value}%")

    def toggle_dayNightSettings(self, checked):
        self.dayNightRatioSlider.setEnabled(checked)
        self.dayBeginInput.setEnabled(checked)
        self.dayEndInput.setEnabled(checked)

    def fetch_data(self):

        cursor_metadata = self.conn_metadata.cursor()
        cursor_metadata.execute("SELECT DISTINCT recording_id FROM metadata")
        recording_ids = [row[0] for row in cursor_metadata.fetchall()]
        recording_ids = [
            '_'.join([part.upper() if i == 0 else part for i, part in enumerate(rec_id.split('_'))]) for rec_id in
            recording_ids]

        print(recording_ids)

        cursor_details = self.conn_details.cursor()

        details_data = {}
        for recording_id in recording_ids:
            cursor_details.execute(
                "SELECT family, color, shape FROM morphotypes WHERE recording_id = ?",
                (recording_id,),
            )
            details = cursor_details.fetchone()
            if details:  # Check if details exist for the recording_id
                details_data[recording_id] = details
                print(details)
        return details_data

# TODO: Check how is it that not all items are loaded into the trees.
    def populate_trees(self, train_ids=None, val_ids=None):
        details_data = self.fetch_data()

        # Function to populate a specific tree
        def populate_tree(tree, ids):
            for recording_id, (family, color, shape) in details_data.items():
                if ids is None or recording_id in ids:
                    parent = QTreeWidgetItem(tree)
                    parent.setText(0, recording_id)  # Setting the recording_id in the first column
                    parent.setText(1, family)  # Setting the family in the second column
                    parent.setText(2, color)  # Setting the color in the third column
                    parent.setText(3, shape)  # Setting the shape in the fourth column
                    parent.setFlags(parent.flags() | Qt.ItemIsTristate | Qt.ItemIsUserCheckable)

        populate_tree(self.train_tree, train_ids)
        populate_tree(self.val_tree, val_ids)

    def move_to_val(self):
        self._move_selected_items(self.train_tree, self.val_tree)

    def move_to_train(self):
        self._move_selected_items(self.val_tree, self.train_tree)

    def _move_selected_items(self, source_tree, dest_tree):
        selected_items = source_tree.selectedItems()

        for item in selected_items:
            if item.parent() is None:  # Ensure it's a top-level item
                recording_id = item.text(0)  # Grab the recording_id from the first column

                if self.item_exists(dest_tree, recording_id):
                    # Remove the item from the source tree
                    index = source_tree.indexOfTopLevelItem(item)
                    source_tree.takeTopLevelItem(index)
                else:
                    # Clone and add to the destination tree
                    clone = item.clone()
                    dest_tree.addTopLevelItem(clone)

        # Update background color for both trees
        self.update_background_color(source_tree, dest_tree)
        self.update_background_color(dest_tree, source_tree)

    # Function to check if a recording_id already exists in a tree
    def item_exists(self, tree, recording_id):
        root = tree.invisibleRootItem()
        child_count = root.childCount()
        for i in range(child_count):
            item = root.child(i)
            if item.text(0) == recording_id:
                return True
        return False

    # Function to set background color based on item's existence in both trees
    def update_background_color(self, tree1, tree2):
        root1 = tree1.invisibleRootItem()
        root2 = tree2.invisibleRootItem()

        for i in range(root1.childCount()):
            item = root1.child(i)
            recording_id = item.text(0)
            if self.item_exists(tree2, recording_id):
                item.setBackground(0, QColor(255, 255, 255))  # White for existing in both
            else:
                item.setBackground(0, QColor(0, 255, 0))  # Green for existing in one

        for i in range(root2.childCount()):
            item = root2.child(i)
            recording_id = item.text(0)
            if self.item_exists(tree1, recording_id):
                item.setBackground(0, QColor(255, 255, 255))  # White for existing in both
            else:
                item.setBackground(0, QColor(0, 255, 0))  # Green for existing in one

    # Function to save the configuration
    def save_config(self):
        options = QFileDialog.Options()
        filePath, _ = QFileDialog.getSaveFileName(self, "Save Configuration", "", "JSON Files (*.json);;All Files (*)",
                                                  options=options)

        if filePath:
            if not filePath.endswith('.json'):
                filePath += '.json'

            train_ids = [self.train_tree.topLevelItem(i).text(0) for i in range(self.train_tree.topLevelItemCount())]
            val_ids = [self.val_tree.topLevelItem(i).text(0) for i in range(self.val_tree.topLevelItemCount())]

            with open(filePath, 'w') as f:
                json.dump({"train": train_ids, "val": val_ids}, f)

    # Function to load the configuration
    def load_config(self):
        options = QFileDialog.Options()
        filePath, _ = QFileDialog.getOpenFileName(self, "Load Configuration", "", "JSON Files (*.json);;All Files (*)",
                                                  options=options)

        if filePath:
            with open(filePath, 'r') as f:
                config_data = json.load(f)

            # Clear existing items from both trees
            self.train_tree.clear()
            self.val_tree.clear()

            # Assuming fetch_data_ids() returns a list of all recording_ids in the database
            database_ids = self.fetch_data_ids()

            # Check if database contains all the recording_ids in the config file
            if all(id in database_ids for id in config_data['train']) and all(
                    id in database_ids for id in config_data['val']):
                self.populate_trees(config_data['train'], config_data['val'])

                # Update background color for both trees
                self.update_background_color(self.train_tree, self.val_tree)
                self.update_background_color(self.val_tree, self.train_tree)
            else:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Critical)
                msg.setText("Error")
                msg.setInformativeText("Some recording_ids in the configuration file are not found in the database.")
                msg.setWindowTitle("Error")
                msg.exec_()

    def fetch_data_ids(self):
        # Establish the database connection
        conn = sqlite3.connect(self.main_database_path)
        cursor = conn.cursor()

        # Execute SQL query to get unique recording_ids
        cursor.execute("SELECT DISTINCT recording_id FROM metadata")

        # Fetch all rows from the query
        rows = cursor.fetchall()

        # Close the database connection
        conn.close()

        # Extract the recording_ids from the rows and return as a list
        return [row[0] for row in rows]

    def _get_folders(self, tree):
        folders = []
        for i in range(tree.topLevelItemCount()):
            parent = tree.topLevelItem(i)
            for j in range(parent.childCount()):
                child = parent.child(j)
                if child.checkState(0) == Qt.Checked:
                    folders.append(child.text(0))
        return folders

    def collect_selected_ids(self, tree_widget):
        selected_ids = []
        for index in range(tree_widget.topLevelItemCount()):
            item = tree_widget.topLevelItem(index)
            selected_ids.append(item.text(0))
        print(f"returning ids: {selected_ids}")
        return selected_ids

    def accept(self):
        self.selected_train_ids = self.collect_selected_ids(self.train_tree)
        self.selected_val_ids = self.collect_selected_ids(self.val_tree)

        # Call the original accept method to close the dialog
        super().accept()

class DatasetExporter(QThread):
    progress_signal = pyqtSignal(int)
    progress_total_signal = pyqtSignal(int)
    indeterminate_progress_signal = pyqtSignal(bool)
    export_database_created_signal = pyqtSignal(str)

    def __init__(self, dataset_size, empty_visitor_ratio, use_day_night, day_night_ratio, day_start_time, day_end_time,
                 train_val_ratio,
                 generate_yaml, selected_train_ids, selected_val_ids, description_text, root_folder_path, destination_folder_path, database_path: str = None):
        QThread.__init__(self)
        self.dataset_size = dataset_size
        self.use_day_night = use_day_night
        self.daytime_nighttime_ratio = day_night_ratio
        self.daytime_start = day_start_time
        self.daytime_end = day_end_time
        self.empty_visitor_ratio = empty_visitor_ratio
        self.train_val_ratio = train_val_ratio
        self.generate_yaml = generate_yaml
        self.selected_train_ids = selected_train_ids
        self.selected_val_ids = selected_val_ids
        self.description_text = description_text
        self.root_folder_path = root_folder_path
        self.destination_folder_path = destination_folder_path
        self.database_path = database_path
        self.dataset_name = ""
        self.progress = 0
        self.export_from_db = self.check_export_from_existing_export_database()

    def create_unique_export_database(self):

        # Generate a unique hash (for example, from the current time)
        unique_hash = hashlib.sha1(str(datetime.datetime.now()).encode()).hexdigest()[:8]
        self.dataset_name = unique_hash

        # Modify the name
        old_database_path = self.database_path
        new_database_path = os.path.join(os.path.dirname(self.database_path),
                                         os.path.splitext(os.path.basename(self.database_path))[
                                             0] + f'_{unique_hash}.db')

        # Copy the current database to the new unique name
        shutil.copy(old_database_path, new_database_path)
        self.database_path = new_database_path

    def check_export_from_existing_export_database(self):

        db_name = self.database_path
        export_from_db = False

        # Connect to the database
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()

        # Execute the query to count rows where chosen_for_export is not 0
        cursor.execute("SELECT COUNT(*) FROM metadata WHERE chosen_for_export != 0")
        count = cursor.fetchone()[0]

        # Check the count and take appropriate action
        if count > 0:
            export_from_db = True

        # Close the database connection
        conn.close()

        return export_from_db

    def setup_progress_tracking(self, result):
        # Setup progress tracking
        total_files = 0
        for parent_folder, types in result.items():
            for file_type, time_dict in types.items():
                total_files += len(time_dict['all'])
                print(total_files)
        progress = int(total_files * 0.1)
        self.progress_total_signal.emit(int(total_files + progress))
        self.progress_signal.emit(progress)
        return progress

    def create_dataset_folders(self):

        # Create destination folders for each parent folder, if needed
        parent_folder_destination = os.path.join(self.destination_folder_path, self.dataset_name)
        images_folder = os.path.join(parent_folder_destination, 'images')
        labels_folder = os.path.join(parent_folder_destination, 'labels')
        images_train_folder = os.path.join(images_folder, 'train')
        images_val_folder = os.path.join(images_folder, 'val')
        labels_train_folder = os.path.join(labels_folder, 'train')
        labels_val_folder = os.path.join(labels_folder, 'val')
        for folder in [images_train_folder, images_val_folder, labels_train_folder, labels_val_folder]:
            os.makedirs(folder, exist_ok=True)

        return parent_folder_destination, images_folder, labels_folder, images_train_folder, images_val_folder, labels_train_folder, labels_val_folder

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

    def fetch_files_per_parent_folder(self):

        def select_files_among_visits(all_relevant_files, count):
            print(f"selecting {len(all_relevant_files)}")
            if len(all_relevant_files) > 0:
                files_by_visit_number = defaultdict(list)
                for row in all_relevant_files:
                    full_path, visit_number = row
                    files_by_visit_number[visit_number].append(full_path)

                selected_files = []
                while len(selected_files) < min(count, len(all_relevant_files)):
                    for visit_number, files in files_by_visit_number.items():
                        if len(selected_files) >= count:
                            break
                        if files:
                            chosen_file = random.choice(files)
                            files.remove(chosen_file)
                            selected_files.append(chosen_file)
            else:
                return []
            return selected_files

        # Connect to the database
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()

        cursor.execute("SELECT DISTINCT parent_folder FROM metadata")
        parent_folders = [row[0] for row in cursor.fetchall()]

        # Calculate the number of parent folders and therefore the number of files per folder
        number_of_parent_folders = len(parent_folders)
        total_files_per_folder = self.dataset_size // number_of_parent_folders

        # Calculate how many files in total should be val and train
        val_number_of_files = int(self.dataset_size * self.train_val_ratio / (1 + self.train_val_ratio))
        train_number_of_files = self.dataset_size - val_number_of_files

        # Read configuration for parent folder use
        self.parent_folder_config = self.generate_parent_folder_config()

        print(self.parent_folder_config)

        print(f"Day/night ratio: {self.daytime_nighttime_ratio}")

        # Calculate counts per category
        num_empty_per_folder = int(total_files_per_folder * self.empty_visitor_ratio / (1 + self.empty_visitor_ratio))
        num_visitor_per_folder = total_files_per_folder - num_empty_per_folder

        # Initialize result dictionary
        files_by_folder = defaultdict(dict)

        print(f"tot:{total_files_per_folder}, emp:{num_empty_per_folder}, vis:{num_visitor_per_folder}")
        for parent_folder in parent_folders:

            folder_usage = self.parent_folder_config.get(parent_folder, 'both')  # Defaults to 'both'

            # Modify the number of files based on folder usage
            num_empty_modified, num_visitor_modified = self.modify_counts_based_on_folder_usage(num_empty_per_folder, num_visitor_per_folder, folder_usage)

            # Calculate for each folder
            num_daytime_per_empty = int(
                num_empty_modified * self.daytime_nighttime_ratio / (1 + self.daytime_nighttime_ratio))
            num_nighttime_per_empty = num_empty_modified - num_daytime_per_empty

            num_daytime_per_visitor = int(
                num_visitor_modified * self.daytime_nighttime_ratio / (1 + self.daytime_nighttime_ratio))
            num_nighttime_per_visitor = num_visitor_modified - num_daytime_per_visitor

            # Step 3: Continue existing logic, but use modified counts
            for label_condition, count_day, count_night, count in [
                ("IS NULL", num_daytime_per_empty, num_nighttime_per_empty, num_empty_modified),
                ("IS NOT NULL", num_daytime_per_visitor, num_nighttime_per_visitor, num_visitor_modified)]:
                if self.use_day_night:

                    print(f"{count_day}, {count_night}, {count}, {self.daytime_start}, {self.daytime_end}")

                    # Query for daytime files, selected evenly across visits.
                    query = """
                        SELECT full_path, visit_no
                        FROM metadata
                        WHERE parent_folder = ?
                        AND label_path {}
                        AND time(time) BETWEEN ? AND ?
                    """.format(label_condition)
                    cursor.execute(query, (parent_folder, self.daytime_start, self.daytime_end))
                    all_relevant_files = cursor.fetchall()

                    # Use the custom funciton to return files selected by visit in defined quantity
                    daytime_files = select_files_among_visits(all_relevant_files, count_day)

                    # # Query for 'daytime' files
                    # query = """
                    #     SELECT full_path
                    #     FROM metadata
                    #     WHERE parent_folder = ?
                    #     AND label_path {}
                    #     AND time(time) BETWEEN ? AND ?
                    #     LIMIT ?
                    # """.format(label_condition)
                    #
                    # cursor.execute(query, (parent_folder, daytime_start, daytime_end, count_day))
                    # daytime_files = [row[0] for row in cursor.fetchall()]

                    # Query for nighttime files, selected evenly across visits.
                    query = """
                        SELECT full_path, visit_no
                        FROM metadata
                        WHERE parent_folder = ?
                        AND label_path {}
                        AND NOT time(time) BETWEEN ? AND ?
                    """.format(label_condition)
                    cursor.execute(query, (parent_folder, self.daytime_start, self.daytime_end))
                    all_relevant_files = cursor.fetchall()

                    nighttime_files = select_files_among_visits(all_relevant_files, count_night)

                    # # Query for 'nighttime' files
                    # query = """
                    #     SELECT full_path
                    #     FROM metadata
                    #     WHERE parent_folder = ?
                    #     AND label_path {}
                    #     AND NOT time(time) BETWEEN ? AND ?
                    #     LIMIT ?
                    # """.format(label_condition)
                    # cursor.execute(query, (parent_folder, daytime_start, daytime_end, count_night))
                    # nighttime_files = [row[0] for row in cursor.fetchall()]

                    # Update result dictionary
                    type_label = 'empty' if label_condition == "IS NULL" else 'visitor'
                    files_by_folder[parent_folder][type_label] = {
                        'daytime': daytime_files,
                        'nighttime': nighttime_files,
                        'all': daytime_files + nighttime_files
                    }
                else:
                    print("doing this")
                    print(f'{parent_folder}')
                    # Fetch files ignoring day/night ratio
                    query = """
                        SELECT full_path, visit_no
                        FROM metadata
                        WHERE parent_folder = ?
                        AND label_path {}
                    """.format(label_condition)
                    cursor.execute(query, (parent_folder,))
                    all_relevant_files = cursor.fetchall()

                    all_files = select_files_among_visits(all_relevant_files, count)

                    # query = """
                    #                 SELECT full_path
                    #                 FROM metadata
                    #                 WHERE parent_folder = ?
                    #                 AND label_path {}
                    #                 LIMIT ?
                    #             """.format(label_condition)
                    # print(count)
                    # cursor.execute(query, (parent_folder, count))
                    # all_files = [row[0] for row in cursor.fetchall()]

                    type_label = 'empty' if label_condition == "IS NULL" else 'visitor'
                    files_by_folder[parent_folder][type_label] = {
                        'all': all_files
                    }

        # Here, let's update the 'chosen_for_export' flag for the selected files.
        for parent_folder, types in files_by_folder.items():
            for type_label, time_dict in types.items():
                for file in time_dict.get('all', []):
                    update_query = "UPDATE metadata SET chosen_for_export = 1 WHERE full_path = ?"
                    cursor.execute(update_query, (file,))
        conn.commit()

        # Close the database connection
        conn.close()

        return files_by_folder

    def generate_parent_folder_config(self):
        config = {}
        for folder_id in self.selected_train_ids:
            config[folder_id] = 'train'

        for folder_id in self.selected_val_ids:
            if folder_id in config:  # Already exists in config
                config[folder_id] = 'both'
            else:
                config[folder_id] = 'val'

        return config

    def modify_counts_based_on_folder_usage(self, num_empty, num_visitor, folder_usage):
        # Initialize modified counts to original counts
        num_empty_modified, num_visitor_modified = num_empty, num_visitor

        if folder_usage == 'train':
            # Implement logic to modify counts for training only
            num_empty_modified = int(num_empty * 1.2)  # Example: Increase by 20%
            num_visitor_modified = int(num_visitor * 1.2)  # Example: Increase by 20%

        elif folder_usage == 'val':
            # Implement logic to modify counts for validation only
            num_empty_modified = int(num_empty * 0.8)  # Example: Decrease by 20%
            num_visitor_modified = int(num_visitor * 0.8)  # Example: Decrease by 20%

        else:
            # No modification needed if it's for both
            pass

        return num_empty_modified, num_visitor_modified

    def copy_files(self, result):

        # Setup progress tracking
        self.progress = self.setup_progress_tracking(result)

        # Create and check folders for the dataset export
        parent_folder_destination, images_folder, labels_folder, images_train_folder, images_val_folder, labels_train_folder, labels_val_folder = self.create_dataset_folders()

        # Initialize database connection
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()

        future_to_file = {}

        with ThreadPoolExecutor() as executor:
            # for parent_folder, types_data in result.items():
            #     # Get number of empty files
            #     number_of_empty_files = len(types_data['empty']['all'])
            #
            #     # Calculate counts per category
            #     num_empty_files_for_val = int(number_of_empty_files * self.train_val_ratio / (1 + self.train_val_ratio))
            #     num_empty_files_for_train = number_of_empty_files - num_empty_files_for_val
            #
            #     # Prepare list of empty files
            #     empty_files = types_data['empty']['all']
            #     random.shuffle(empty_files)
            #
            #     # Get number of visitor files
            #     number_of_visitor_files = len(types_data['visitor']['all'])
            #
            #     # Calculate counts per category
            #     num_visitor_files_for_val = int(
            #         number_of_visitor_files * self.train_val_ratio / (1 + self.train_val_ratio))
            #     num_visitor_files_for_train = number_of_visitor_files - num_visitor_files_for_val
            #
            #     # Prepare list of visitor files
            #     visitor_files = types_data['visitor']['all']
            #     random.shuffle(visitor_files)

            # Loop through each parent folder
            for parent_folder, types_data in result.items():
                print(parent_folder)

                # Get the folder usage
                folder_usage = self.parent_folder_config.get(parent_folder, 'both')

                # Get number of empty files
                number_of_empty_files = len(types_data['empty']['all'])

                # Calculate counts per category
                num_empty_files_for_val = int(number_of_empty_files * self.train_val_ratio / (1 + self.train_val_ratio))
                num_empty_files_for_train = number_of_empty_files - num_empty_files_for_val

                # Prepare list of empty files
                empty_files = types_data['empty']['all']
                random.shuffle(empty_files)

                # Get number of visitor files
                number_of_visitor_files = len(types_data['visitor']['all'])

                # Calculate counts per category
                num_visitor_files_for_val = int(
                    number_of_visitor_files * self.train_val_ratio / (1 + self.train_val_ratio))
                num_visitor_files_for_train = number_of_visitor_files - num_visitor_files_for_val

                # Prepare list of visitor files
                visitor_files = types_data['visitor']['all']
                random.shuffle(visitor_files)

                # Modify the counts based on the usage of the folder
                num_empty_files_for_train, num_empty_files_for_val = self.modify_counts_based_on_folder_usage(
                    num_empty_files_for_train, num_empty_files_for_val, folder_usage)
                num_visitor_files_for_train, num_visitor_files_for_val = self.modify_counts_based_on_folder_usage(
                    num_visitor_files_for_train, num_visitor_files_for_val, folder_usage)

                for i, file in enumerate(empty_files):
                    print(file)
                    file_name = os.path.basename(file)
                    source = file

                    # Modify this part to use folder_usage
                    if folder_usage == 'train':
                        destination = os.path.join(images_train_folder, file_name)
                        dataset_type = 'train'
                    elif folder_usage == 'val':
                        destination = os.path.join(images_val_folder, file_name)
                        dataset_type = 'val'
                    else:  # 'both'
                        if (i + 1) <= num_empty_files_for_train:
                            destination = os.path.join(images_train_folder, file_name)
                            dataset_type = 'train'
                        else:
                            destination = os.path.join(images_val_folder, file_name)
                            dataset_type = 'val'

                    future = executor.submit(self.copy_file_task, source, destination, labels_folder, dataset_type,
                                             'empty')
                    future_to_file[future] = file

                # Copy selected visitor files
                for i, file in enumerate(visitor_files):
                    print(file)
                    file_name = os.path.basename(file)
                    source = file

                    # Modify this part to use folder_usage
                    if folder_usage == 'train':
                        destination = os.path.join(images_train_folder, file_name)
                        dataset_type = 'train'
                    elif folder_usage == 'val':
                        destination = os.path.join(images_val_folder, file_name)
                        dataset_type = 'val'
                    else:  # 'both'
                        if (i + 1) <= num_visitor_files_for_train:
                            destination = os.path.join(images_train_folder, file_name)
                            dataset_type = 'train'
                        else:
                            destination = os.path.join(images_val_folder, file_name)
                            dataset_type = 'val'

                    future = executor.submit(self.copy_file_task, source, destination, labels_folder, dataset_type,
                                             'visitor')
                    future_to_file[future] = file

            # Getting completion messages
            for future in concurrent.futures.as_completed(future_to_file):
                file = future_to_file[future]
                try:
                    data = future.result()  # This will give the return value from your function
                except Exception as exc:
                    print(f"{file} generated an exception: {exc}")
                else:
                    # print(f"{file} returned {data}")
                    self.progress += 1
                    self.progress_signal.emit(self.progress)

        # Commit database changes
        conn.commit()

        # Close database connection
        conn.close()

        # Add the accompanying files to the dataset folder
        self.generate_accompanying_files(parent_folder_destination)

    def copy_file_task(self, source, destination, labels_folder, dataset_type, file_type=None, cursor=None):

        # Create a new SQLite connection
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()

        normalized_source = os.path.normpath(source)
        normalized_destination = os.path.normpath(destination)
        shutil.copy(normalized_source, normalized_destination)
        # subprocess.run(["cp", normalized_source, normalized_destination])

        if file_type == 'visitor':
            source_txt = source.rsplit('.', 1)[0] + '.txt'
            destination_txt = os.path.join(labels_folder, dataset_type, os.path.basename(source_txt))
            normalized_source_txt = os.path.normpath(source_txt)
            normalized_destination_txt = os.path.normpath(destination_txt)
            # subprocess.run(["cp", normalized_source_txt, normalized_destination_txt])
            shutil.copy(normalized_source_txt, normalized_destination_txt)

        if cursor:
            update_val = 1 if dataset_type == 'train' else 2
            cursor.execute("UPDATE metadata SET chosen_for_export=? WHERE full_path=?", (update_val, source))

        conn.commit()

        conn.close()

        return source, update_val

    ## This is for the alternative approach when exporting from existing export database
    #TODO: Check if it works and update to make copying faster

    def fetch_files_for_train_val(self, database_path: str):
        # Connect to the database
        conn = sqlite3.connect(database_path)
        cursor = conn.cursor()

        # Initialize result dictionary
        files_by_folder = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

        # Get distinct parent folders
        cursor.execute("SELECT DISTINCT parent_folder FROM metadata")
        parent_folders = [row[0] for row in cursor.fetchall()]

        for parent_folder in parent_folders:
            all_files_in_folder = []
            for label_condition, type_label in [("IS NULL", "empty"), ("IS NOT NULL", "visitor")]:
                all_files_by_type = []
                for dataset_type in [1, 2]:  # 0 = Don't use, 1 = Train, 2 = Val
                    query = """
                            SELECT full_path
                            FROM metadata
                            WHERE parent_folder = ?
                            AND label_path {}
                            AND chosen_for_export = ?
                        """.format(label_condition)

                    cursor.execute(query, (parent_folder, dataset_type))
                    files = [row[0] for row in cursor.fetchall()]
                    all_files_by_type.extend(files)
                    if dataset_type == 1:
                        files_by_folder[parent_folder][type_label]['train'] = files
                    elif dataset_type == 2:
                        files_by_folder[parent_folder][type_label]['val'] = files

                files_by_folder[parent_folder][type_label]['all'] = all_files_by_type
            #     all_files_in_folder.extend(all_files_by_type)
            #
            # files_by_folder[parent_folder]['all']['all'] = all_files_in_folder

        # Close the database connection
        conn.close()

        print(files_by_folder)

        return files_by_folder

    def copy_files_and_update_database(self, result):

        # Setup progress tracking
        self.progress = self.setup_progress_tracking(result)

        # Create and check folders for the dataset export
        parent_folder_destination, images_folder, labels_folder, images_train_folder, images_val_folder, labels_train_folder, labels_val_folder = self.create_dataset_folders()
        future_to_file = {}
        with ThreadPoolExecutor() as executor:
            for parent_folder, types_data in result.items():
                for dataset_type in ['train', 'val']:
                    for file_type in ['empty', 'visitor']:
                        for file in types_data[file_type][dataset_type]:
                            print(file)
                            file_name = os.path.basename(file)
                            source = file
                            destination = os.path.join(images_folder, dataset_type, file_name)

                            future = executor.submit(self.copy_file_task, source, destination, labels_folder,
                                                     dataset_type, file_type)
                            future_to_file[future] = file

            # Getting completion messages
            for future in concurrent.futures.as_completed(future_to_file):
                file = future_to_file[future]
                try:
                    data = future.result()  # This will give the return value from your function
                except Exception as exc:
                    print(f"{file} generated an exception: {exc}")
                else:
                    # print(f"{file} returned {data}")
                    self.progress += 1
                    self.progress_signal.emit(self.progress)

        # Add the accompanying files to the dataset folder
        self.generate_accompanying_files(parent_folder_destination)

    ## This deals with generating the accompanying files for the dataset

    def generate_accompanying_files(self, parent_folder_destination):

        def get_actual_dataset_size(database_path):
            conn = sqlite3.connect(database_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM metadata WHERE chosen_for_export IN (1, 2)")
            actual_size = cursor.fetchone()[0]
            conn.close()
            return actual_size

        if self.generate_yaml:
            dataset_path = os.path.join('dataset', self.dataset_name)
            train_path = os.path.join('images', 'train')
            val_path = os.path.join('images', 'val')
            num_classes = 2
            class_names = ['pollinator', 'lepidoptera']
            filepath = os.path.join(parent_folder_destination, f'{self.dataset_name}.yaml')

            self.generate_dataset_yaml(dataset_path, train_path, val_path, num_classes, class_names, filepath)

        # Supplement the dataset with the export database for versioning
        old_database_path = self.database_path
        new_database_path = os.path.join(parent_folder_destination, os.path.basename(self.database_path))
        self.database_path = new_database_path

        # Copy the current database to the new unique name
        shutil.copy(old_database_path, new_database_path)

        # Supplement the dataset with a readme.md file which contains the name: hash, set size: self.dataset_size,
        # actual size: total amount of files actually copied, train/val ratio, day/night ratio, day_start, day_end, and the description_text.
        # Create readme.md file
        readme_filepath = os.path.join(parent_folder_destination, 'readme.md')
        actual_size = get_actual_dataset_size(self.database_path)
        with open(readme_filepath, 'w') as f:
            f.write(f"# {self.dataset_name}\n")  # Title (Dataset Name)
            f.write(f"Set Size: {self.dataset_size}\n")
            f.write(f"Actual Size: {actual_size}\n")  # Replace actual_size with the actual size (number of files)
            f.write(f"Train/Val Ratio: {self.train_val_ratio}\n")
            f.write(f"Day/Night Ratio: {self.daytime_nighttime_ratio}\n")
            f.write(f"Day Start: {self.daytime_start}\n")  # Replace day_start with actual start time
            f.write(f"Day End: {self.daytime_end}\n")  # Replace day_end with actual end time
            f.write(f"\n## Description\n")  # Description header
            f.write(f"{self.description_text}\n")  # Replace description_text with the actual description

    def generate_dataset_yaml(self, dataset_path, train_path, val_path, num_classes, class_names, filepath):
        data = {
            'path': dataset_path,
            'train': train_path,
            'val': val_path,
            'nc': num_classes,
            'names': class_names
        }

        with open(filepath, 'w') as f:
            yaml.dump(data, f, sort_keys=False)

    def run(self):

        self.indeterminate_progress_signal.emit(True)

        if not os.path.isfile(self.database_path):
            return
        elif self.database_path.lower().endswith(".db"):
            self.create_unique_export_database()

            # number_of_parent_folders = self.get_unique_parent_folder_count(self.database_path)
            # print(f"parent_folders:{number_of_parent_folders}")
            # files_per_parent_folder = self.dataset_size // number_of_parent_folders
            if self.export_from_db:
                result = self.fetch_files_for_train_val(self.database_path)
                print("coyping")
                self.copy_files_and_update_database(result)
            else:
                result = self.fetch_files_per_parent_folder()
                self.copy_files(result)
            self.export_database_created_signal.emit(self.database_path)


class DatabasePopulator(QThread):
    progress_signal = pyqtSignal(int)
    progress_total_signal = pyqtSignal(int)

    def __init__(self, root_path, batch_size, database_path):
        QThread.__init__(self)
        self.root_path = root_path
        self.batch_size = batch_size
        self.database_path = database_path

    def create_database(self):
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
            label_path TEXT,
            time TEXT,
            parent_folder TEXT,
            chosen_for_export INTEGER
        );
        """)
        conn.commit()
        conn.close()

    def run(self):

        # Check if database exists
        if os.path.exists(self.database_path):
            os.remove(self.database_path)
            print(f"Deleted existing database: {self.database_path}")

        # Create a new one
        print("Creating database...")
        self.create_database()

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

                    time = self.extract_time_from_video_file_id(video_file_id)
                    parent_folder = self.extract_parent_folder_from_full_path(full_path)

                    # Data validation and conversion to integers
                    try:
                        frame_no = int(parts[7])
                        visit_no = int(parts[8])
                        crop_no = int(parts[6])
                        x1, y1 = map(int, parts[9].split(','))
                        x2, y2 = map(int, parts[10].split(','))
                    except ValueError as e:
                        print(f"Skipping file due to invalid metadata: {file}, Error: {e}")
                        continue

                    label_path = None
                    if "visitor" in subdir:
                        label_path = os.path.join(subdir, file.replace('.jpg', '.txt'))

                    batch_data.append((recording_id, video_file_id, frame_no, visit_no, crop_no, x1, y1, x2, y2,
                                       full_path, label_path, time, parent_folder))

                    count += 1
                    if count >= self.batch_size:
                        cursor.executemany(
                            "INSERT INTO metadata (recording_id, video_file_id, frame_no, visit_no, crop_no, x1, y1, x2, y2, full_path, label_path, time, parent_folder, chosen_for_export) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)",
                            batch_data)
                        conn.commit()
                        batch_data = []
                        count = 0

        if batch_data:
            cursor.executemany(
                "INSERT INTO metadata (recording_id, video_file_id, frame_no, visit_no, crop_no, x1, y1, x2, y2, full_path, label_path, time, parent_folder, chosen_for_export) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)",
                batch_data)
            conn.commit()

        self.progress_signal.emit(total_files)  # Or however you wish to compute progress
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


class PlottingThread(QThread):
    finished = pyqtSignal()  # Signal emitted when thread finishes

    def __init__(self, plot_instance, database_path, additional_db_path=None):
        super(PlottingThread, self).__init__()
        self.plot_instance = plot_instance
        self.database_path = database_path
        self.additional_db_path = additional_db_path

    def run(self):
        if self.additional_db_path is not None:
            self.plot_instance.update_database(self.database_path, self.additional_db_path)
        else:
            self.plot_instance.update_database(self.database_path)
        self.finished.emit()  # Emit finished signal


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
    def __init__(self, database_path, dataset_type_filter=1, parent=None, width=5, height=4, dpi=100):
        self.database_path = database_path
        self.dataset_type_filter = dataset_type_filter
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
        cursor.execute("SELECT time FROM metadata WHERE chosen_for_export = ?", (self.dataset_type_filter,))
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
    def __init__(self, database_path, dataset_type_filter=1, parent=None, width=5, height=4, dpi=100):
        self.database_path = database_path
        self.dataset_type_filter = dataset_type_filter
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
            WHERE chosen_for_export = ? 
            GROUP BY parent_folder, file_type""", (self.dataset_type_filter,))

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
    def __init__(self, database_path, dataset_type_filter: int = 1, parent=None, width=5, height=4, dpi=100):
        self.database_path = database_path
        self.dataset_type_filter = dataset_type_filter
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

        if sizes[0] == 0 and sizes[1] == 0:
            return

        # Add absolute numbers to labels
        labels_with_count = [f"{label} ({count})" for label, count in zip(labels, sizes)]

        self.axes.clear()
        self.axes.pie(sizes, labels=labels_with_count, autopct='%1.1f%%', colors=("r", "g"))
        self.axes.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        # Add this line to adjust layout to fit axis labels
        self.axes.figure.tight_layout()

        self.draw()

    def get_frame_proportions(self):
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()

        # Count empty frames that are chosen for export
        cursor.execute(
            "SELECT COUNT(*) FROM metadata WHERE label_path IS NULL AND chosen_for_export=?",
            (self.dataset_type_filter,))
        empty_frames = cursor.fetchone()[0]

        # Count visitor frames that are chosen for export
        cursor.execute(
            "SELECT COUNT(*) FROM metadata WHERE label_path IS NOT NULL AND chosen_for_export = ?",
            (self.dataset_type_filter,))
        visitor_frames = cursor.fetchone()[0]

        conn.close()

        return {'Empty Frames': empty_frames, 'Visitor Frames': visitor_frames}

    def update_database(self, new_database_path):
        self.database_path = new_database_path
        self.plot()


class HeatmapPlot(FigureCanvas):
    def __init__(self, database_path, dataset_type_filter=1, parent=None, width=5, height=4, dpi=100):
        print("init")
        self.database_path = database_path
        self.dataset_type_filter = dataset_type_filter
        fig = plt.figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
        self.number_of_parent_folders = 0
        self.global_max = 0
        self.df = None

        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

        if os.path.isfile(self.database_path):
            print("plot")
            self.plot()

    def get_data(self):
        conn = sqlite3.connect(self.database_path)
        query = """SELECT parent_folder, visit_no, COUNT(*) as freq 
                           FROM metadata 
                           WHERE chosen_for_export = ? 
                           GROUP BY parent_folder, visit_no"""
        df = pd.read_sql_query(query, conn, params=(self.dataset_type_filter,))

        print(f"df: {df}")
        conn.close()

        # Re-index the 'visit_no' within each 'parent_folder'
        df['new_visit_no'] = df.groupby('parent_folder')['visit_no'].rank(method='first').astype(int)

        # Now pivot the DataFrame
        df_pivot = df.pivot(index="parent_folder", columns="new_visit_no", values="freq")

        # Fill NaN values with 0 for the heatmap
        df_pivot.fillna(0, inplace=True)

        self.number_of_parent_folders = len(df_pivot)
        self.global_max = df['freq'].max()
        return df_pivot

    def plot(self, start_idx=0, end_idx=10):
        print(f"indexes {start_idx}, {end_idx}")

        try:
            self.figure.clf()
        except KeyError:
            print("Error occurred while clearing figure.")
        self.axes = self.figure.add_subplot(111)

        # Retrieve the data from the database
        if self.df is None:
            self.df = self.get_data()

        # Use local variable as it will be filtered and so on
        df = self.df

        # Check if DataFrame is empty
        if df.empty:
            print("Dataframe is empty. No plots will be generated.")
            return

        # Filter rows based on the slider range
        df_filtered = df.iloc[start_idx:end_idx]

        print(f"df_filtered: {df_filtered}")
        if not len(df_filtered) == 0:
            sns.heatmap(df_filtered, annot=True, fmt=".0f", ax=self.axes, vmin=0, vmax=self.global_max, cmap='crest')

        self.axes.set_title('Heatmap of Visit Numbers by Parent Folder')
        self.axes.figure.tight_layout()
        self.draw()

    def update_database(self, new_database_path):
        self.database_path = new_database_path
        self.plot()


class PlantTraitStackedBarPlot(FigureCanvas):
    def __init__(self, database_path, morphotype_db_path, dataset_type_filter=1, metric_displayed=0, parent=None, width=5, height=4, dpi=100):
        fig = plt.figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        metrics = ["family", "color", "shape"]
        self.metric = metrics[metric_displayed]
        self.database_path = database_path
        self.dataset_type_filter = dataset_type_filter
        self.morphotype_db_path = morphotype_db_path
        self.df = None

        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

        if os.path.isfile(self.database_path) and os.path.isfile(self.morphotype_db_path):
            self.plot()

    def get_data(self):
        # Connect to your databases
        conn1 = sqlite3.connect(self.database_path)
        conn2 = sqlite3.connect(self.morphotype_db_path)

        # Query 1: Get parent_folder and count of empty and visitor frames
        query1 = """
            SELECT recording_id,
                   CASE 
                       WHEN label_path IS NULL THEN 'empty' 
                       ELSE 'visitor' 
                   END AS file_type, 
                   COUNT(*) as count_
            FROM metadata
            WHERE chosen_for_export = ?
            GROUP BY parent_folder, file_type;
            """
        df1 = pd.read_sql_query(query1, conn1, params=(self.dataset_type_filter,))
        conn1.close()

        print(df1)

        # Query 2: Get family information for each parent_folder
        query2 = """
        SELECT recording_id, {}
        FROM morphotypes;
        """.format(self.metric)
        df2 = pd.read_sql_query(query2, conn2)

        print(df2)

        # Close your database connections
        conn1.close()
        conn2.close()

        # Combine the data
        combined_df = pd.merge(df1, df2, on='recording_id')
        combined_grouped_df = combined_df.groupby([self.metric, 'file_type']).sum().reset_index()

        print(combined_grouped_df)

        return combined_grouped_df

    def plot(self):
        self.figure.clf()
        self.axes = self.figure.add_subplot(111)

        # Retrieve data
        if self.df is None:
            self.df = self.get_data()

        if self.df.empty:
            print("Dataframe is empty. No plots will be generated.")
            return

        # Create the stacked bar plot
        #sns.barplot(data=self.df, x='family', y='count_', hue='file_type', ax=self.axes)

        # self.axes.set_title('Family Distribution in the Dataset')
        # self.axes.figure.tight_layout()
        # self.draw()

        # Assuming `self.df` has columns ['Family', 'Type', 'Region', 'Count']
        df_pivot = self.df.pivot(index=self.metric, columns='file_type', values='count_').fillna(0)

        metric_levels = df_pivot.index.tolist()
        empty_counts = df_pivot['empty'].tolist()
        visitor_counts = df_pivot['visitor'].tolist()

        x = np.arange(len(metric_levels))  # the label locations
        width = 0.5  # the width of the bars

        rects1 = self.axes.bar(x, empty_counts, width, label='empty', color='r')
        rects2 = self.axes.bar(x, visitor_counts, width, label='visitor', bottom=empty_counts, color='g')

        # Calculate the average total count per family and draw a horizontal line
        average_total_count = np.mean(np.array(empty_counts) + np.array(visitor_counts))
        self.axes.axhline(average_total_count, color='gray', linestyle='--', linewidth=1,
                          label=f"Avg Total Count: {int(average_total_count)}")

        # Labels, title, and custom x-axis tick labels
        self.axes.set_ylabel('Counts')
        self.axes.set_title(f'Counts by {self.metric} and frame type')
        x = np.arange(len(metric_levels))

        # Truncate family names to first 5 characters and add a period
        truncated_metric_levels = [f[:5] + '.' for f in metric_levels]
        self.axes.set_xticks(x)
        self.axes.set_xticklabels(truncated_metric_levels, rotation=45, ha='right')
        self.axes.legend()

        # Add labels above or on the bars
        for i, rect in enumerate(rects1):
            height = rect.get_height()
            self.axes.annotate(f'{int(height)}',
                               xy=(rect.get_x() + rect.get_width() / 2, height),
                               xytext=(0, 3),
                               textcoords="offset points",
                               ha='center', va='bottom')

        for i, rect in enumerate(rects2):
            height = rect.get_height() + empty_counts[i]  # Offset by the height of 'empty' bar
            self.axes.annotate(f'{int(height)}',
                               xy=(rect.get_x() + rect.get_width() / 2, height),
                               xytext=(0, 3),
                               textcoords="offset points",
                               ha='center', va='bottom')

        self.draw()

    def update_database(self, new_database_path, new_morphotype_db_path):
        self.database_path = new_database_path
        self.morphotype_db_path = new_morphotype_db_path
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
        self.plotting_threads = []

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

        size = QSize(35, 35)

        self.open_button = QPushButton('Open Folder', self)
        self.open_button.clicked.connect(self.show_open_folder_dialog),
        self.open_button.setIcon(QIcon(QPixmap("resources/img/fl.png")))
        self.open_button.setIconSize(size)
        toolbar.addWidget(self.open_button)

        self.db_button = QPushButton('Generate Database', self)
        self.db_button.clicked.connect(lambda: self.import_data(self.root_path))
        self.db_button.setIcon(QIcon(QPixmap("resources/img/db_plus.png")))
        self.db_button.setIconSize(size)
        self.db_button.setEnabled(False)
        toolbar.addWidget(self.db_button)

        # New button for selecting a database
        self.select_db_button = QPushButton('Select Database', self)
        self.select_db_button.setEnabled(False)
        self.select_db_button.setIcon(QIcon(QPixmap("resources/img/db_tick.png")))
        self.select_db_button.setIconSize(size)

        def select_database():
            options = QFileDialog.Options()
            options |= QFileDialog.ReadOnly
            file_name, _ = QFileDialog.getOpenFileName(self, "Select a database", "",
                                                       "Database Files (*.db);;All Files (*)", options=options)
            if file_name:
                self.db_name_label.setText(f"Selected DB: {os.path.basename(file_name)}")
                self.database_path = file_name
                self.export_button.setEnabled(True)

                db_name = self.database_path

                # Connect to the database
                conn = sqlite3.connect(db_name)
                cursor = conn.cursor()

                # Execute the query to count rows where chosen_for_export is not 0
                cursor.execute("SELECT COUNT(*) FROM metadata WHERE chosen_for_export != 0")
                count = cursor.fetchone()[0]

                # Check the count and take appropriate action
                if count > 0:
                    self.update_inspection_graphs(self.database_path)

                # Close the database connection
                conn.close()

        self.select_db_button.clicked.connect(select_database)
        toolbar.addWidget(self.select_db_button)

        # Create a QLabel to display the selected database name
        self.db_name_label = QLabel(f"Selected DB: ... ")
        self.db_name_label.setFrameShape(QFrame.StyledPanel)  # Set the frame shape
        self.db_name_label.setFrameShadow(QFrame.Sunken)  # Set the frame shadow
        self.db_name_label.setLineWidth(1)  # Set the line width of the frame

        # Add padding and margin if needed
        # self.db_name_label.setContentsMargins(10, 10, 10, 10)  # margin between the text and frame
        toolbar.addWidget(self.db_name_label)

        self.export_button = QPushButton('Export Dataset', self)
        self.export_button.clicked.connect(self.show_export_dialog)
        self.export_button.setEnabled(False)
        self.export_button.setIcon(QIcon(QPixmap("resources/img/ds_export.png")))
        self.export_button.setIconSize(size)
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
        self.tree_visitor.clicked.connect(lambda: self.item_clicked(
            self.tree_visitor))  # DONE: This should connect to the clickedTree function and that should trigger functions based on whether it was a fodler or a file

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
        self.tree_empty.clicked.connect(lambda: self.item_clicked(
            self.tree_empty))  # DONE: This should connect to the clickedTree function and that should trigger functions based on whether it was a fodler or a file

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

        # Create the Vertical Tab Widget
        self.vertical_tabs = QTabWidget()
        self.vertical_tabs.setTabPosition(QTabWidget.West)
        vertical_tabs = []
        self.plots = {
            'time': [],
            'visit': [],
            'family': [],
            'color': [],
            'shape': [],
            'folder': [],
            'pie': [],
            'database': []
        }
        self.sliders = []

        for i, tab in enumerate(['Training Dataset', 'Validation Dataset']):
            print(i)
            # Create tab
            tab_widget = QWidget()
            tab_layout = QVBoxLayout(tab_widget)
            self.vertical_tabs.addTab(tab_widget, tab)
            vertical_tabs.append(tab_widget)

            # Initialize main vertical splitter for the second tab
            tab_vertical_splitter = QSplitter(Qt.Vertical)

            # Initialize QLabel
            label = QLabel(tab)
            font = QFont()
            font.setBold(True)
            font.setPointSize(10)
            label.setFont(font)

            # Set text and alignment
            label.setText(tab)
            label.setAlignment(Qt.AlignCenter)
            tab_layout.addWidget(label)

            # Initialize and set QFrame as a line divider
            divider = QFrame()
            divider.setFrameShape(QFrame.HLine)
            divider.setFrameShadow(QFrame.Sunken)
            divider.setMinimumSize(0, 2)  # Set minimum height to 2 for visibility
            tab_layout.addWidget(divider)

            # Initialize first horizontal splitter
            horizontal_splitter_1 = QSplitter(Qt.Horizontal)

            # Create a QTabWidget to hold multiple plots
            ds_data_tab_widget = QTabWidget()
            ds_data_tab_widget.setTabPosition(QTabWidget.North)

            # Create the original time_plot and add it as the first tab
            time_plot = TimeDistributionPlot("", i + 1, width=5, height=4, dpi=100)
            ds_data_tab_widget.addTab(time_plot, "Time Distribution")

            visit_plot_widget = QWidget()  # Create a QWidget
            visit_plot_layout = QHBoxLayout()  # Create a layout
            visit_plot = HeatmapPlot("", i + 1, width=5, height=4, dpi=100)  # Your heatmap
            visit_plot_layout.addWidget(visit_plot)  # Add heatmap to layout

            # Initialize and configure the slider
            slider = QSlider(Qt.Vertical, self)
            slider.setMinimum(0)
            slider.setMaximum(max(visit_plot.number_of_parent_folders - 10, 0))
            slider.setSingleStep(1)
            slider.setPageStep(10)
            slider.setInvertedAppearance(True)
            slider.setInvertedControls(True)
            print(f"index", i)
            print("Initial slider value:", slider.value())
            self.sliders.append(slider)
            self.sliders[i].valueChanged.connect(lambda val, i=i: self.slider_moved(val, i))


            # Add slider to layout
            visit_plot_layout.addWidget(slider)

            visit_plot_widget.setLayout(visit_plot_layout)  # Set the widget's layout
            ds_data_tab_widget.addTab(visit_plot_widget, "Visit Distribution")  # Add widget as tab

            family_stacked_bar_plot = PlantTraitStackedBarPlot("", os.path.join('resources', 'db', 'morphotypes.db'), i + 1, metric_displayed=0)
            ds_data_tab_widget.addTab(family_stacked_bar_plot, "Family Distribution")

            color_stacked_bar_plot = PlantTraitStackedBarPlot("", os.path.join('resources', 'db', 'morphotypes.db'),
                                                               i + 1, metric_displayed=1)
            ds_data_tab_widget.addTab(color_stacked_bar_plot, "Color Distribution")

            shape_stacked_bar_plot = PlantTraitStackedBarPlot("", os.path.join('resources', 'db', 'morphotypes.db'),
                                                              i + 1, metric_displayed=2)
            ds_data_tab_widget.addTab(shape_stacked_bar_plot, "Shape Distribution")

            folder_plot = FolderDistributionPlot("", i + 1, width=5, height=4, dpi=100)
            horizontal_splitter_1.addWidget(ds_data_tab_widget)
            horizontal_splitter_1.addWidget(folder_plot)

            # Initialize second horizontal splitter
            horizontal_splitter_2 = QSplitter(Qt.Horizontal)
            pie_chart = FrameProportionPieChart("", i + 1, width=5, height=4, dpi=100)

            table_view = QTableView()
            model = QSqlTableModel()
            model.setTable('metadata')
            model.select()
            table_view.setModel(model)
            horizontal_splitter_2.addWidget(pie_chart)
            horizontal_splitter_2.addWidget(table_view)

            # Add horizontal splitters to main vertical splitter
            tab_vertical_splitter.addWidget(horizontal_splitter_1)
            tab_vertical_splitter.addWidget(horizontal_splitter_2)

            # Record plot references
            self.plots['time'].append(time_plot)
            self.plots['visit'].append(visit_plot)
            self.plots['family'].append(family_stacked_bar_plot)
            self.plots['color'].append(color_stacked_bar_plot)
            self.plots['shape'].append(shape_stacked_bar_plot)
            self.plots['folder'].append(folder_plot)
            self.plots['pie'].append(pie_chart)
            self.plots['database'].append(table_view)

            # Set the initial sizes to make the splitter handle appear at the center
            initial_size = self.height() // 2  # Assuming 'self' is the QMainWindow
            horizontal_splitter_1.setSizes([initial_size, initial_size])
            horizontal_splitter_2.setSizes([initial_size, initial_size])

            # Add main vertical splitter to second tab layout
            tab_layout.addWidget(tab_vertical_splitter)

        # Add the vertical tabs widget to the main tab widget
        new_layout.addWidget(self.vertical_tabs)

        # Set the layout for the central widget
        self.centralWidget().setLayout(main_layout)

        # Set window properties and show
        self.setWindowTitle('ICDM - Insect Communities Dataset Manager')
        self.show()

    def slider_moved(self, value, slider_index):
        print(f"Slider moved, i = {slider_index}")
        #start_idx = self.sliders[slider_index].value()
        start_idx = value
        print(value)
        end_idx = start_idx + 10  # Display 10 parent folders at a time
        self.plots['visit'][slider_index].plot(start_idx, end_idx)  # Assuming `heatmap` is your heatmap object

    def update_database(self, database_widget, dataset_type_filter, new_database_path):
        # Step 1: Close the existing database connection
        QSqlDatabase.database().close()

        # Step 2: Open a new connection with the new database name
        db = QSqlDatabase.addDatabase("QSQLITE")
        db.setDatabaseName(new_database_path)
        ok = db.open()
        if not ok:
            print("Failed to open new database")  # Handle this more gracefully
            return

        query = QSqlQuery(db)
        query.prepare("SELECT COUNT(*) FROM metadata WHERE chosen_for_export = ?")
        query.addBindValue(dataset_type_filter)
        query.exec_()
        if query.next():
            record_count = query.value(0)
        print(f"Record count: {record_count}")

        # self.database_path = new_database_path  # Update the class attribute

        # Step 3: Reinitialize the QSqlTableModel with the new database
        new_model = QSqlTableModel()
        new_model.setTable('metadata')  # Assuming 'metadata' is the table name
        new_model.setFilter(f"chosen_for_export = {dataset_type_filter}")  # Optional: apply your filter
        new_model.select()

        # Step 4: Update the QTableView to use the new model
        database_widget.setModel(new_model)  # Assuming `self.table_view` is your QTableView instance

    def show_export_dialog(self):
        dialog = ExportDialog(self.database_path, os.path.join(
                    'resources', 'db', 'morphotypes.db'))
        result = dialog.exec_()

        if result == QDialog.Accepted:
            # Gather values from the fields
            dataset_size = max(int(dialog.datasetSizeInput.text()), 1)
            visitor_percentage = dialog.emptyVisitorRatioSlider.value()
            day_percentage = dialog.dayNightRatioSlider.value()
            day_start_time = dialog.dayBeginInput.time().toString("HH:mm")
            day_end_time = dialog.dayEndInput.time().toString("HH:mm")
            selection_method = dialog.methodDropdown.currentText()
            train_percentage = dialog.trainValRatioSlider.value()
            generate_yaml = dialog.generateYamlCheckbox.isChecked()
            use_day_night = dialog.useDayNightCheckbox.isChecked()
            selected_train_ids = dialog.selected_train_ids
            selected_val_ids = dialog.selected_val_ids
            description_text = dialog.description_text_edit.toPlainText()

            print(
                f"Retrieved values: Dataset Size: {dataset_size}, Empty/Visitor Ratio: {visitor_percentage}, Day/Night Ratio: {day_percentage}, Day Start Time: {day_start_time}, Day End Time: {day_end_time}, Selection Method: {selection_method}, Train/Validation Ratio: {train_percentage}, Generate YAML: {generate_yaml}, Use Day/Night: {use_day_night}"
            )

            empty_visitor_ratio = max((100 - visitor_percentage), 1) / max(visitor_percentage, 1)
            day_night_ratio = max((100 - day_percentage), 1) / max(day_percentage, 1)
            train_val_ratio = max((100 - train_percentage), 1) / max(train_percentage, 1)

            # Pass the new fields as well
            self.perform_export(
                dataset_size,
                empty_visitor_ratio,
                use_day_night,
                day_night_ratio,
                day_start_time,
                day_end_time,
                train_val_ratio,
                generate_yaml,
                selected_train_ids,
                selected_val_ids,
                description_text
            )

    def perform_export(self, dataset_size, empty_visitor_ratio, use_day_night, day_night_ratio, day_start_time,
                       day_end_time, train_val_ratio, generate_yaml, selected_train_ids, selected_val_ids, description_text):

        destination_folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if not destination_folder:
            return

        self.dataset_exporter = DatasetExporter(dataset_size, empty_visitor_ratio, use_day_night, day_night_ratio,
                                                day_start_time, day_end_time, train_val_ratio,
                                                generate_yaml, selected_train_ids, selected_val_ids, description_text, self.root_path, destination_folder, self.database_path)
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

    # def update_inspection_graphs(self, database_path):
    #     self.export_database_path = database_path
    #
    #     for i, tab in enumerate(['Train', 'Val']):
    #         self.plots['time'][i].update_database(self.export_database_path)
    #         self.plots['visit'][i].update_database(self.export_database_path)
    #         print(self.plots['visit'][i].number_of_parent_folders)
    #         self.sliders[i].setMaximum(self.plots['visit'][i].number_of_parent_folders - 10)
    #         self.plots['family'][i].update_database(self.export_database_path,
    #                                                 os.path.join('resources', 'db', 'morphotypes.db'))
    #         self.plots['color'][i].update_database(self.export_database_path,
    #                                                 os.path.join('resources', 'db', 'morphotypes.db'))
    #         self.plots['shape'][i].update_database(self.export_database_path,
    #                                                 os.path.join('resources', 'db', 'morphotypes.db'))
    #         self.plots['folder'][i].update_database(self.export_database_path)
    #         self.plots['pie'][i].update_database(self.export_database_path)
    #         self.update_database(self.plots['database'][i], i + 1, self.export_database_path)

    def thread_finished(self):
        for thread in self.plotting_threads:
            if not thread.isRunning():
                thread.deleteLater()
        self.plotting_threads = [thread for thread in self.plotting_threads if thread.isRunning()]
        self.update_sliders()


    def update_sliders(self):
        for i, tab in enumerate(['Train', 'Val']):
            # Update sliders
            self.sliders[i].setMaximum(self.plots['visit'][i].number_of_parent_folders - 10)

    def update_inspection_graphs(self, database_path):
        self.export_database_path = database_path

        for i, tab in enumerate(['Train', 'Val']):
            for plot_type in ['time', 'visit', 'family', 'color', 'shape', 'folder', 'pie']:
                plot_instance = self.plots[plot_type][i]
                additional_db_path = None if plot_type not in ['family', 'color', 'shape'] else os.path.join(
                    'resources', 'db', 'morphotypes.db')

                thread = PlottingThread(plot_instance, self.export_database_path, additional_db_path)
                thread.finished.connect(self.thread_finished)
                self.plotting_threads.append(thread)
                thread.start()

            self.update_database(self.plots['database'][i], i + 1, self.export_database_path)

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
            db_folder = os.path.join("resources", "db")
            os.makedirs(db_folder, exist_ok=True)
            self.database_path = os.path.join(db_folder, f'{os.path.basename(folder_path)}.db')

            # Check if database exists
            if os.path.exists(self.database_path):
                self.update_gui_on_loaded_database()
            self.select_db_button.setEnabled(True)
            self.db_button.setEnabled(True)

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

    def update_gui_on_loaded_database(self):
        self.db_name_label.setText(f"Selected DB: {os.path.basename(self.database_path)}")
        self.export_button.setEnabled(True)

    def import_data(self, folder_path):

        db_folder = os.path.join("resources", "db")
        os.makedirs(db_folder, exist_ok=True)
        self.database_path = os.path.join(db_folder, f'{os.path.basename(folder_path)}.db')

        # Initiate
        self.database_populator = DatabasePopulator(root_path=folder_path, batch_size=100,
                                                    database_path=self.database_path)
        self.database_populator.progress_total_signal.connect(self.set_progress)
        self.database_populator.progress_signal.connect(self.update_progress)
        self.database_populator.finished.connect(self.update_gui_on_loaded_database)

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


if __name__ == '__main__':
    import sys

    app = QApplication(sys.argv)
    ex = ICDM()
    sys.exit(app.exec_())
