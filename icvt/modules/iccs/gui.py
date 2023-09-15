import math
import os
from datetime import datetime, timedelta
import sys
import sqlite3

import PyQt5
import PyQt5.QtCore as QtCore
import PyQt5.QtGui as QtGui
import cv2
from PIL import Image
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QIcon, QImage
from PyQt5.QtWidgets import (QToolBar, QSpacerItem, QGridLayout, QSizePolicy, QHBoxLayout, QScrollArea, QLabel,
                             QMainWindow, QPushButton)

# Get the directory containing the main script
main_script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))

# Add to the PATH environment variable
os.environ["PATH"] = main_script_dir + os.pathsep + os.environ["PATH"]
import mpv

# Import ICVT components
from ..iccs.mpv import MPVThread

pyqt = os.path.dirname(PyQt5.__file__)
os.environ['QT_PLUGIN_PATH'] = os.path.join(pyqt, "Qt5/plugins")

class ICCS_GUI:
    def __init__(self):
        pass

    def open_main_window(self):
        # Create the main window
        self.main_window = QMainWindow()
        self.main_window.setWindowTitle(
            f"{self.app_title} - Folder: {os.path.basename(os.path.normpath(self.video_folder_path))} - Table: {os.path.basename(os.path.normpath(self.annotation_file_path))}"
        )
        self.main_window.setWindowIcon(QIcon("resources/img/iccs.png"))

        # Central Widget
        central_widget = QWidget(self.main_window)
        self.main_window.setCentralWidget(central_widget)

        # Create a QVBoxLayout
        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        # Create and add components to layout
        top_toolbar = self.build_top_toolbar(central_widget)
        center_frame = self.build_center_frame(central_widget)
        bottom_toolbar = self.build_bottom_toolbar()

        # Add top and bottom toolbars using addToolBar
        self.main_window.addToolBar(Qt.TopToolBarArea, top_toolbar)
        self.main_window.addToolBar(Qt.BottomToolBarArea, bottom_toolbar)

        # Add other widgets to the central layout
        layout.addWidget(center_frame)

        # Update button images
        if self.loaded == True:
            for each in range(len(self.video_filepaths)):
                first = -1
                self.update_button_image(self.frames[each].copy(), (max(each, 0) // 6),
                                         (each - ((max(each, 0) // 6) * 6)), first)

        # Optional configurations for screen size, full-screen, etc.
        screen = self.main_window.screen()
        screen_geometry = screen.geometry()
        self.main_window.setGeometry(screen_geometry)

        # app = QApplication(sys.argv)
        self.main_window.showMaximized()
        #self.main_window.show()  # Assuming your QMainWindow object is called 'main_window'
        # sys.exit(app.exec_())

        #self.main_window.show()

    def build_top_toolbar(self, parent):
        def can_switch_folder(where):
            if not self.dir_hierarchy:  # If directory hierarchy is not set
                return False

            current_index = self.scanned_folders.index(os.path.basename(os.path.normpath(self.video_folder_path)))

            if where == "right":
                return current_index + 1 < len(self.scanned_folders)
            elif where == "left":
                return current_index > 0
            else:
                return False  # Default case

        def untoggle_buttons(buttons, selected_button):
            try:
                for i, btn in enumerate(buttons):
                    if btn is not selected_button:
                        btn.setChecked(False)
                    else:
                        self.update_crop_mode(i+1)
            except:
                raise

        def switch_autoprocessing():
            try:
                self.auto_processing = 1 - self.auto_processing
            except Exception as e:
                print(f"An error occurred: {e}")

        screen = self.main_window.screen()
        screen_geometry = screen.geometry()
        screen_width = screen_geometry.width()

        buttons = []

        # Create toolbar and set icon size
        toolbar = QToolBar("Top Toolbar")
        grid_layout = QGridLayout()

        # LEFT Button
        left_button = QPushButton("Previous folder")
        left_button.setIcon(QIcon(QPixmap("resources/img/la.png")))
        left_button.setEnabled(can_switch_folder("left"))
        left_button.clicked.connect(lambda: self.switch_folder("left"))
        grid_layout.addWidget(left_button, 0, 0)
        buttons.append(left_button)

        # MENU Button
        menu_button = QPushButton("Menu")
        menu_button.setIcon(QIcon(QPixmap("resources/img/mn.png")))
        menu_button.clicked.connect(self.open_menu)
        grid_layout.addWidget(menu_button, 0, 1)
        buttons.append(menu_button)

        toggle_buttons = []
        from functools import partial

        for i in range(1, 4):
            button = QPushButton(f' Mode')
            button.setIcon(QIcon(QPixmap(f"resources/img/{i}.png")))
            button.setCheckable(True)
            button.clicked.connect(partial(untoggle_buttons, toggle_buttons, button))
            buttons.append(button)
            toggle_buttons.append(button)
            grid_layout.addWidget(button, 0, 1+i)

            if i == self.crop_mode:
                button.setChecked(True)

        # AUTO Checkbox
        # auto_checkbox = QCheckBox()
        # auto_checkbox.setChecked(False)
        # #auto_checkbox.stateChanged.connect()
        # grid_layout.addWidget(auto_checkbox, 0, 5)

        # AUTO Button
        auto_button = QPushButton("Automatic evaluation")
        auto_button.setIcon(QIcon(QPixmap("resources/img/au.png")))
        auto_button.setCheckable(True)
        auto_button.clicked.connect(switch_autoprocessing)
        grid_layout.addWidget(auto_button, 0, 5)
        buttons.append(auto_button)

        if self.auto_processing == 1:
            auto_button.setChecked(True)

        # VIDEO FOLDER Button
        video_folder_button = QPushButton("Select video folder")
        video_folder_button.clicked.connect(self.change_video_folder)
        video_folder_button.setIcon(QIcon(QPixmap("resources/img/fl.png")))
        grid_layout.addWidget(video_folder_button, 0, 6)
        buttons.append(video_folder_button)

        # EXCEL PATH Button
        excel_path_button = QPushButton("Select Excel table")
        excel_path_button.clicked.connect(self.change_excel_path)
        excel_path_button.setIcon(QIcon(QPixmap("resources/img/et.png")))
        grid_layout.addWidget(excel_path_button, 0, 7)
        buttons.append(excel_path_button)

        # OCR Button
        ocr_button = QPushButton("OCR")
        ocr_button.clicked.connect(lambda: self.open_ocr_roi_gui(self.video_filepaths[0]))
        ocr_button.setIcon(QIcon(QPixmap("resources/img/ocr.png")))
        grid_layout.addWidget(ocr_button, 0, 8)
        buttons.append(ocr_button)

        # Generate ROIs Button
        genrois_button = QPushButton("ROIs")
        genrois_button.clicked.connect(self.generate_roi_entries)
        genrois_button.setIcon(QIcon(QPixmap("resources/img/od.png")))
        grid_layout.addWidget(genrois_button, 0, 9)
        buttons.append(genrois_button)

        # RIGHT Button
        right_button = QPushButton("Next folder")
        right_button.setIcon(QIcon(QPixmap("resources/img/ra.png")))
        right_button.setEnabled(can_switch_folder("right"))
        right_button.clicked.connect(lambda: self.switch_folder("right"))
        grid_layout.addWidget(right_button, 0, 10)
        buttons.append(right_button)

        for button in buttons:
            button_width = screen_width // (len(buttons) + 5)
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            button.setMinimumWidth(button_width)
            button.setIconSize(QtCore.QSize(50, 50))

        # Setting margins to 0
        grid_layout.setHorizontalSpacing(0)
        grid_layout.setContentsMargins(0, 0, 0, 0)

        # Set layout to a QWidget to act as the layout manager for the toolbar
        layout_widget = QWidget()
        layout_widget.setLayout(grid_layout)

        # Add this QWidget to the toolbar
        toolbar.addWidget(layout_widget)

        return toolbar

    def build_center_frame(self, parent):
        outer_frame = parent
        scroll_area = QScrollArea(outer_frame)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        target_frame = QWidget()
        vertical_layout = QVBoxLayout()

        row_length = 6

        screen = self.main_window.screen()
        screen_geometry = screen.geometry()
        screen_width = screen_geometry.width()
        button_width = int(screen_width // (row_length + 0.5))
        button_height = int(button_width / 1.76923)

        # button_width = 276
        # button_height = 156

        self.button_images = []
        self.buttons = []
        rows = math.ceil(len(self.video_filepaths) / 6)

        for i in range(rows):
            button_frame = QWidget()
            button_layout = QHBoxLayout()

            for j in range(6):
                vbox = QVBoxLayout()

                if j + i * 6 < len(self.video_filepaths):
                    index = j + i * 6
                    frame = self.frames[j + i * 6]
                    frame_height, frame_width, _ = frame.shape
                    button_height = int(button_width * (frame_height / frame_width))
                    pil_img = cv2.cvtColor(self.frames[j + i * 6], cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(pil_img)
                    #pil_img = pil_img.resize((button_width, button_height))
                    q_img = QImage(pil_img.tobytes(), pil_img.size[0], pil_img.size[1], QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(q_img)

                    button = RightClickableQPushButton(self.video_filepaths[index])
                    button.setIcon(QtGui.QIcon(pixmap))
                    button.setIconSize(QtCore.QSize(button_width, button_height))
                    button.clicked.connect(lambda i=i, j=j: self.open_roi_gui(i, j, self.button_images))

                    # Prepare name elements
                    filename = os.path.basename(self.video_filepaths[index])
                    locality, transect, plant_id, date, hour, minutes = filename[:-4].split("_")

                    label = QLabel(f"{date[-2:]}/{date[-4:-2]}/{date[:4]} - {hour}:{minutes}")
                    label.setFont(QtGui.QFont("Arial", 13))
                    label.setAlignment(Qt.AlignCenter)

                    vbox.setSpacing(0)
                    vbox.addWidget(button)
                    vbox.addWidget(label)

                    self.buttons.append(button)
                    self.button_images.append(pixmap)

                else:
                    spacer = QSpacerItem(button_width, button_height, QSizePolicy.Expanding, QSizePolicy.Fixed)
                    button_layout.addSpacerItem(spacer)

                button_layout.addLayout(vbox)

            button_layout.setSpacing(0)
            button_layout.setContentsMargins(0, 0, 0, 0)

            vertical_layout.setSpacing(0)
            vertical_layout.setContentsMargins(0, 0, 0, 0)

            button_frame.setLayout(button_layout)
            vertical_layout.addWidget(button_frame)

        target_frame.setLayout(vertical_layout)
        scroll_area.setWidget(target_frame)

        return scroll_area

    def build_bottom_toolbar(self):
        # Create QToolBar
        toolbar = QToolBar("Bottom Toolbar")

        # Create a grid layout
        grid_layout = QGridLayout()

        # INitiate list for buttons
        buttons = []

        # Get screen size
        screen = self.main_window.screen()
        screen_geometry = screen.geometry()
        screen_width = screen_geometry.width()

        # # Add button from build_bottom_left_panel function
        # fl_button = QPushButton()
        # fl_button.setIcon(QIcon("resources/img/fl.png"))
        # fl_button.clicked.connect(lambda: os.startfile(self.video_folder_path))
        # grid_layout.addWidget(fl_button, 0, 0)
        # buttons.append(fl_button)

        # Add buttons from original build_bottom_toolbar function
        button_data = [
            ("resources/img/fl.png", "Open video folder", lambda: os.startfile(self.video_folder_path)),
            ("resources/img/sv_1.png", "Save", self.save_progress),
            ("resources/img/cr_1.png", "Crop", self.crop_engine),
            ("resources/img/so.png", "Sort", self.sort_engine),
            ("resources/img/lo.png", "Load", self.load_progress),
            ("resources/img/et.png", "Open Excel table", lambda: os.startfile(self.annotation_file_path)),
            ("resources/img/ef.png", "Open Excel folder", lambda: os.startfile(os.path.dirname(self.annotation_file_path)))
        ]

        col_index = 0  # Column index to continue adding buttons
        for icon_path, text, callback in button_data:
            button = QPushButton(text)
            button.setIcon(QIcon(icon_path))
            button.clicked.connect(callback)
            grid_layout.addWidget(button, 0, col_index)
            button.setIconSize(QtCore.QSize(50, 50))
            col_index += 1
            buttons.append(button)

        for button in buttons:
            button_width = screen_width // (len(buttons) + 5)
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            button.setMinimumWidth(button_width)
            button.setIconSize(QtCore.QSize(50, 50))

        # Create a QWidget to act as a layout manager for the toolbar
        layout_widget = QWidget()
        layout_widget.setLayout(grid_layout)

        # Add the QWidget to the toolbar
        toolbar.addWidget(layout_widget)

        return toolbar

class RightClickableQPushButton(QPushButton):
    def __init__(self, video_filepath, *args, **kwargs):
        super(RightClickableQPushButton, self).__init__(*args, **kwargs)
        self.video_filepath = video_filepath
        self.player = None  # mpv player instance

    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton:
            #self.open_player()
            self.initialize_db()
            #self.open_database_entry_window()

            self.mpv_thread = MPVThread(self.video_filepath)
            self.mpv_thread.shift_click_signal.connect(self.open_database_entry_window)
            self.mpv_thread.start()

            # self.player_thread = Thread(target=self.run_mpv_player)
            # self.player_thread.start()
           # self.run_mpv_player()
        else:
            super().mousePressEvent(event)

    def run_mpv_player(self):
        self.player = mpv.MPV(player_operation_mode='pseudo-gui', input_default_bindings=True)
        #player.start = 0  # Adjust the value as needed
        self.player.register_key_binding('CLOSE_WIN', 'quit')
        self.player.play(self.video_filepath)
        self.player.wait_for_playback()
        self.player.terminate()

    def open_database_entry_window(self, current_time):
        if hasattr(self, "db_window") and self.db_window is not None:
            #print("Database window is already open.")
            return
        #print("Opening Database Entry Window")
        self.db_window = DatabaseEntryWindow(video_filepath=self.video_filepath, current_time=current_time)
        #print(f"DatabaseEntryWindow instantiated: {self.db_window}")
        self.db_window.setAttribute(Qt.WA_DeleteOnClose, True)
        self.db_window.show()
        #print("Show method called")
        self.db_window.destroyed.connect(self.reset_db_window)

    def reset_db_window(self):
        #print("Destroyed signal received.")
        self.db_window = None

    def initialize_db(self):

        # Get folder path
        folder_path = os.path.dirname(self.video_filepath)

        # Get filename
        filename = os.path.basename(self.video_filepath)

        # Prepare name elements
        locality, transect, plant_id, date, hour, minutes = filename[:-4].split("_")

        # Define compound info
        recording_identifier = "_".join([locality, transect, plant_id])

        conn = sqlite3.connect(os.path.join(folder_path, f'{recording_identifier}_flowering_minutes.db'))
        c = conn.cursor()

        c.execute('''CREATE TABLE IF NOT EXISTS video_data
                     (video_id TEXT, year INTEGER, month INTEGER, day INTEGER, 
                      hour INTEGER, minute INTEGER, no_of_flowers INTEGER)''')

        conn.commit()
        conn.close()

from PyQt5.QtWidgets import QWidget, QPushButton, QVBoxLayout, QTableWidget, QTableWidgetItem, QDesktopWidget

class DatabaseEntryWindow(QWidget):
    def __init__(self, video_filepath: str = None, current_time: int = None, *args, **kwargs):
        super(DatabaseEntryWindow, self).__init__(*args, **kwargs)

        self.time_data, self.recording_identifier, self.folder_path = self.calculate_data_to_enter(video_filepath, current_time)

        self.setWindowTitle('Database Entry')
        layout = QVBoxLayout()

        self.table_widget = QTableWidget()
        self.table_widget.setColumnCount(7)
        self.table_widget.setHorizontalHeaderLabels(
            ["video_id", "year", "month", "day", "hour", "minute", "no_of_flowers"])
        layout.addWidget(self.table_widget)

        # Create a QHBoxLayout for buttons
        button_layout = QHBoxLayout()

        # Save Button
        self.save_button = QPushButton('Save Data')
        self.save_button.setIcon(QIcon("resources/img/sv_1.png"))  # Replace with the path to your icon
        self.save_button.clicked.connect(self.save_data_from_table)
        self.save_button.setIconSize(QtCore.QSize(50, 50))
        button_layout.addWidget(self.save_button)

        # Add Entry Button
        self.add_entry_button = QPushButton("Add Entry")
        self.add_entry_button.setIcon(QIcon("resources/img/add.png"))  # Replace with the path to your icon
        self.add_entry_button.clicked.connect(lambda: self.add_new_entry(False))
        self.add_entry_button.setIconSize(QtCore.QSize(50, 50))
        button_layout.addWidget(self.add_entry_button)

        # Delete Button
        self.delete_button = QPushButton('Delete Entry')
        self.delete_button.setIcon(QIcon("resources/img/del.png"))  # Replace with the path to your icon
        self.delete_button.clicked.connect(self.delete_selected_entry)
        self.delete_button.setIconSize(QtCore.QSize(50, 50))
        button_layout.addWidget(self.delete_button)

        # Close Button
        self.close_button = QPushButton("Close")
        self.close_button.setIcon(QIcon("resources/img/cl.png"))  # Replace with the path to your icon
        self.close_button.clicked.connect(self.close_window)
        self.close_button.setIconSize(QtCore.QSize(50, 50))
        button_layout.addWidget(self.close_button)

        # Add the QHBoxLayout to the QVBoxLayout
        layout.addLayout(button_layout)

        self.setLayout(layout)

        self.load_data_to_table()
        self.add_new_entry()

        self.table_widget.cellChanged.connect(self.validate_cell)

        # Fit columns to their content
        #self.table_widget.resizeColumnsToContents()

        # Calculate width required for table
        width = sum([self.table_widget.columnWidth(i) for i in range(self.table_widget.columnCount())])

        # Add a small margin
        width += 100

        # Calculate height required for table
        row_heights = sum([self.table_widget.rowHeight(i) for i in range(self.table_widget.rowCount())])
        header_height = self.table_widget.horizontalHeader().height()
        height = row_heights + header_height + 40  # Adding some margin

        # Get screen dimensions
        screen = QDesktopWidget().screenGeometry()

        # Limit height to screen height
        height = min(height + 200, (screen.height() - 80))

        self.setGeometry(screen.width() - width, 40, width, height)

        # After you've populated your table_widget
        last_row = self.table_widget.rowCount() - 1  # 0-based index
        last_column = self.table_widget.columnCount() - 1  # 0-based index

        self.table_widget.setCurrentCell(last_row, last_column)

    def calculate_data_to_enter(self, video_filepath, current_time):

        # Get folder path
        folder_path = os.path.dirname(video_filepath)

        # Get filename
        filename = os.path.basename(video_filepath)

        # Prepare name elements
        locality, transect, plant_id, date, hour, minutes = filename[:-4].split("_")

        # Define compound info
        recording_identifier = "_".join([locality, transect, plant_id])
        timestamp = "_".join([date, hour, minutes])

        # Convert the timestamp string to a datetime object
        timestamp_dt = datetime.strptime(timestamp, "%Y%m%d_%H_%M")

        # Add the given number of seconds to the datetime object
        updated_dt = timestamp_dt + timedelta(seconds=current_time)

        return updated_dt, recording_identifier, folder_path


    def load_data_to_table(self):
        conn = sqlite3.connect(os.path.join(self.folder_path, f'{self.recording_identifier}_flowering_minutes.db'))
        c = conn.cursor()

        c.execute("SELECT * FROM video_data")
        rows = c.fetchall()

        self.table_widget.setRowCount(0)
        for row_number, row_data in enumerate(rows):
            self.table_widget.insertRow(row_number)
            for column_number, data in enumerate(row_data):
                self.table_widget.setItem(row_number, column_number, QTableWidgetItem(str(data)))

        conn.close()

    def add_new_entry(self, populate: bool = True):
        current_row = self.table_widget.currentRow()
        if current_row == -1:
            row_position = self.table_widget.rowCount()
            self.table_widget.insertRow(row_position)
            if populate:
                self.populate_row(row_position, self.time_data)
        else:
            self.table_widget.insertRow(current_row + 1)

    def populate_row(self, row_position, time_data):

        #print("populating")

        # Extract individual components
        year = time_data.year
        month = time_data.month
        day = time_data.day
        hour = time_data.hour
        minute = time_data.minute

        # Populate the first five columns with these values
        self.table_widget.setItem(row_position, 0, QTableWidgetItem(str(self.recording_identifier)))
        self.table_widget.setItem(row_position, 1, QTableWidgetItem(str(year)))
        self.table_widget.setItem(row_position, 2, QTableWidgetItem(str(month)))
        self.table_widget.setItem(row_position, 3, QTableWidgetItem(str(day)))
        self.table_widget.setItem(row_position, 4, QTableWidgetItem(str(hour)))
        self.table_widget.setItem(row_position, 5, QTableWidgetItem(str(minute)))

    def delete_selected_entry(self):
        selected_rows = set(index.row() for index in self.table_widget.selectedIndexes())
        for row in sorted(selected_rows, reverse=True):
            self.table_widget.removeRow(row)

    def close_window(self):
        self.close()

    def validate_cell(self, row, column):
        # Validation functions
        def is_num_in_range(text, min_val, max_val):
            return text.isdigit() and min_val <= int(text) <= max_val

        item = self.table_widget.item(row, column)
        if not item:
            return

        text = item.text()

        # Map columns to validation functions and their corresponding arguments
        validation_map = {
            0: (str, None),
            1: (is_num_in_range, (1000, 9999)),
            2: (is_num_in_range, (1, 12)),
            3: (is_num_in_range, (1, 31)),
            4: (is_num_in_range, (0, 24)),
            5: (is_num_in_range, (0, 60)),
            6: (int, None)
        }

        # Retrieve validation function and arguments
        validation_fn, args = validation_map.get(column, (None, None))

        # Run the validation
        if validation_fn == str:
            if not isinstance(text, str):
                item.setText('')
        elif validation_fn == int:
            pass
        elif validation_fn == is_num_in_range:
            if not is_num_in_range(text, *args):
                item.setText('')

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter or event.key() == Qt.Key_Space:
            self.save_data_from_table()
        else:
            super().keyPressEvent(event)

    def save_data_from_table(self):
        conn = sqlite3.connect(os.path.join(self.folder_path, f'{self.recording_identifier}_flowering_minutes.db'))
        c = conn.cursor()

        c.execute("DELETE FROM video_data")  # Clear the existing data

        row_count = self.table_widget.rowCount()
        column_count = self.table_widget.columnCount()

        for row in range(row_count):
            row_data = []
            for column in range(column_count):
                item = self.table_widget.item(row, column)
                if item is not None:
                    row_data.append(item.text())
                else:
                    row_data.append("NULL")  # Or a default value
            c.execute("INSERT INTO video_data VALUES (?, ?, ?, ?, ?, ?, ?)", tuple(row_data))

        conn.commit()
        conn.close()
        self.close()