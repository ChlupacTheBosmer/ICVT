# ICVT modules
from modules.utility.utils import yolobbox2bbox

# Extra modules
import cv2
import numpy as np
import pybboxes as pbx
import PyQt5 as pyqt
from PyQt5.QtWidgets import QApplication, QMainWindow, QGraphicsView, QGraphicsScene, QVBoxLayout, QHBoxLayout, QPushButton, QWidget, QLabel, QComboBox, QRubberBand, QFrame
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QGraphicsItem, QDesktopWidget
from PyQt5.QtCore import Qt, QRectF  # Import QRectF here
from PyQt5.QtGui import QFont, QFontMetrics
from PyQt5.QtGui import QPalette, QColor, QBrush, QCursor

# Default python modules
import shutil
import os
from functools import partial

class ThumbnailItem(QGraphicsItem):
    def __init__(self, pixmap, index, is_current, is_previous, status, viewer):
        super().__init__()
        self.viewer = viewer
        self.pixmap = pixmap
        self.index = index
        self.is_current = is_current
        self.is_previous = is_previous
        self.status_label = QLabel(status)  # Set the status_label based on the status provided

    def boundingRect(self):
        return QRectF(0, 0, self.pixmap.width(), self.pixmap.height() + 20)  # Increased height for the status label

    def paint(self, painter, option, widget):
        painter.drawPixmap(0, 0, self.pixmap)

        # Draw a border around the thumbnail if it represents the current image
        if self.is_current:
            pen = pyqt.QtGui.QPen(Qt.red, 3)  # You can adjust the color and width of the border here
            painter.setPen(pen)
            painter.drawRect(0, 0, self.pixmap.width(), self.pixmap.height())

        # Draw a gray border around the thumbnail if it represents the previous image
        if self.is_previous:
            pen = pyqt.QtGui.QPen(Qt.blue, 3)  # You can adjust the color and width of the border here
            painter.setPen(pen)
            painter.drawRect(0, 0, self.pixmap.width(), self.pixmap.height())

        # Create a QFont with the desired font size (e.g., 10)
        font = QFont()
        font.setPointSize(12)

        # Set the font for the painter
        painter.setFont(font)

        # Draw the status label below the thumbnail
        status_x = self.pixmap.width() // 2 - 17
        status_y = self.pixmap.height() + 15  # Adjust the vertical position of the status label

        # Calculate the bounding rectangle for the text
        font_metrics = QFontMetrics(font)
        text_width = font_metrics.width(self.status_label.text())
        text_height = font_metrics.height()

        # Calculate the position to center the text horizontally
        centered_x = (self.pixmap.width() - text_width) // 2

        # Calculate the vertical position to place the text below the thumbnail
        status_y = self.pixmap.height() + 2 + text_height  # Adjust the vertical position of the status label

        painter.drawText(centered_x, status_y, self.status_label.text())
        #painter.drawText(status_x, status_y, self.status_label.text())

    def setStatus(self, status):
        self.status_label.setText(status)
        self.update()  # Trigger a repaint of the item to update the displayed status

    def mousePressEvent(self, event):
        self.viewer.thumbnail_clicked(self.index)

class ImageViewer(QMainWindow):
    def __init__(self, folder_path, use_label_files: bool = False, positive_status_label = "Visitor", negative_status_label = "Empty", title = "ICVT"):
        super().__init__()
        self.folder_path = folder_path

        if use_label_files:
            self.label_folder_path = self.find_labels_directory(self.folder_path)
            if self.label_folder_path is None:
                print("Error: No label files found.")
                self.label_folder_path = self.folder_path

        self.current_index = 0
        self.previous_index = None  # Initialize the previous_index attribute
        self.thumbnail_padding = 10  # Adjust the padding value as needed
        self.start_index = 0
        self.positive_status_label = positive_status_label
        self.negative_status_label = negative_status_label
        self.image_files = []
        self.use_label_files = use_label_files
        self.title = title
        self.is_selecting_roi = False
        self.resize_in_progress = False
        self.resize_mode = None
        self.colors = [(0, 255, 0),  # Green
                       (0, 0, 255),  # Red
                       (255, 0, 0),  # Blue
                       (0, 255, 255),  # Yellow
                       (255, 0, 255)  # Magenta (Purple)
                       ]
        self.color_codes = [Qt.green, Qt.red, Qt.blue, Qt.yellow, Qt.magenta]
        self.color_names = ['lightgreen', 'red', 'blue', 'yellow', 'magenta']
        self.edit_buttons = []
        self.delete_buttons = []
        self.label_edited_index: int
        self.category_dict = {0: "0. Unspecified Visitor",
                              1: "1. Lepidoptera"
                              }
        self.orders = [value for key, value in sorted(self.category_dict.items())]
        self.device_pixel_ratio = QApplication.instance().devicePixelRatio()  # Get the device pixel ratio (DPR)

        # Get the desktop widget
        desktop = QDesktopWidget()

        # Get the primary screen's geometry (size and position)
        primary_screen = desktop.screenGeometry()

        # Get the width and height of the primary screen
        self.screen_width = primary_screen.width()
        self.screen_height = primary_screen.height()

        # Load the image file path
        self.image_files = self.get_images()

        # Dictionary to store the status for each image
        self.image_statuses = {i: "None" for i in range(len(self.image_files))}

        # Track the currently selected thumbnail index
        self.selected_thumbnail_index = 0

        # initialize the GUI
        self.init_ui()

    def find_labels_directory(self, folder_path):

        # Check if there are .txt files in the original folder
        txt_files_in_folder = [file for file in os.listdir(folder_path) if file.lower().endswith('.txt')]

        if txt_files_in_folder:
            print("There are .txt files in the original folder.")
            return folder_path
        else:
            labels_dir = os.path.join(folder_path, "labels")
            if os.path.exists(labels_dir) and os.path.isdir(labels_dir):
                txt_files = [file for file in os.listdir(labels_dir) if file.lower().endswith('.txt')]
                if txt_files:
                    print(f"Labels directory found: {labels_dir}")
                    return labels_dir
            return None


    def get_images(self):
        # Use os.listdir() to get all image files
        image_files = [os.path.join(self.folder_path, file.encode('utf-8').decode('utf-8')) for file in
                            os.listdir(self.folder_path) if
                            file.lower().endswith(('.jpg', '.png'))]
        return image_files

    def init_ui(self):
        # Set the title of the window
        self.setWindowTitle(self.title)

        # Set cursor appearance
        self.cursor_cross = QCursor(Qt.CrossCursor)
        self.cursor_default = QCursor(Qt.ArrowCursor)

        # Create a QGraphicsView to display the image
        self.scene = QGraphicsScene()
        view_layout = QHBoxLayout()
        self.view = QGraphicsView(self.scene)

        self.view.setMouseTracking(True)
        self.view.viewport().setCursor(self.cursor_default)

        # Create layout for listing labels
        self.labels_box = QVBoxLayout()

        # Create the main view for images and labels
        view_layout.addLayout(self.labels_box)
        view_layout.addWidget(self.view)

        # Create Next and Previous buttons
        self.next_button = QPushButton("Next")
        self.prev_button = QPushButton("Previous")

        # Create a label to display the current image name
        self.image_name_label = QLabel()
        self.progress_label = QLabel()

        # Create a QGraphicsView to display the thumbnails
        self.thumbnails_view = QGraphicsView()
        self.thumbnails_scene = QGraphicsScene()
        self.thumbnails_view.setScene(self.thumbnails_scene)

        # Add the buttons and thumbnails view to a layout
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.prev_button)
        button_layout.addWidget(self.next_button)

        main_layout = QVBoxLayout()
        main_layout.addLayout(view_layout)

        labels_layout = QHBoxLayout()
        labels_layout.addWidget(self.image_name_label)
        labels_layout.addStretch()
        labels_layout.addWidget(self.progress_label)
        main_layout.addLayout(labels_layout)

        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.thumbnails_view)

        # Create a central widget and set the layout
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # Set up the QGraphicsView to handle mouse events
        self.view.mousePressEvent = self.on_mouse_press
        self.view.mouseMoveEvent = self.on_mouse_move
        self.view.mouseReleaseEvent = self.on_mouse_release

        self.rubber_band = QRubberBand(QRubberBand.Rectangle, self.view)

        # Connect button signals to functions
        self.next_button.clicked.connect(self.next_image)
        self.prev_button.clicked.connect(self.prev_image)

        # Load and display the first image
        self.load_image(self.current_index)

        # Load and display the thumbnails
        self.load_thumbnails()

        # Set the window to be maximized
        self.showMaximized()

    def load_thumbnails(self):

        # Clear the existing thumbnails
        self.thumbnails_scene.clear()

        # Start dict
        self.thumbnails_dictionary = {}

        # Calculate the number of thumbnails that can fit in the window horizontally
        available_width = self.thumbnails_view.viewport().width()

        thumbnail_width_dip = 100  # Use DIP value for thumbnail width
        self.thumbnail_width = int(thumbnail_width_dip * (self.device_pixel_ratio*2))  # Convert DIP to physical pixels
        thumbnail_padding_dip = 10  # Use DIP value for thumbnail padding
        self.thumbnail_padding = int(thumbnail_padding_dip * self.device_pixel_ratio)  # Convert DIP to physical pixels

        #thumbnail_width = 100
        num_thumbnails = int(min(len(self.image_files), available_width // (self.thumbnail_width + self.thumbnail_padding)))

        # Calculate the total width for the thumbnails including padding
        total_width = num_thumbnails * self.thumbnail_width + (num_thumbnails - 1) * self.thumbnail_padding

        # Load and display the thumbnails of the previous and next images
        self.start_index = self.current_index - num_thumbnails // 2  # Update the class-level start_index
        self.end_index = self.start_index + num_thumbnails
        for i in range(self.start_index, self.end_index):
            if 0 <= i < len(self.image_files):

                # Resize image to save memory
                img = self.resize_thumbnail_image(self.image_files[i])

                # Create the QPixmap from the QImage
                pixmap = QPixmap.fromImage(img)
                pixmap = pixmap.scaledToWidth(self.thumbnail_width, Qt.SmoothTransformation)

                # Determine if this thumbnail represents the current image
                is_current = (i == self.current_index)

                # Determine if this thumbnail represents the previous image
                is_previous = (i == self.previous_index)

                # Get the status for the thumbnail based on the image index
                status = self.image_statuses.get(i, "None")

                item = ThumbnailItem(pixmap, i, is_current, is_previous, status, self)
                x_pos = (self.thumbnail_width + self.thumbnail_padding) * (i - self.start_index)
                item.setPos(x_pos, 0)
                self.thumbnails_scene.addItem(item)
                self.thumbnails_dictionary[i] = item

    def open_image_uni(self, image_path):
        # Open the image
        stream = open(image_path, "rb")
        bytes = bytearray(stream.read())
        numpyarray = np.asarray(bytes, dtype=np.uint8)
        img = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)
        return img

    def resize_thumbnail_image(self, image_path):
        # Open the image
        img = self.open_image_uni(image_path)

        # Resize the image to the desired dimensions (e.g., 80x80 pixels)
        desired_width, desired_height = self.thumbnail_width, self.thumbnail_width
        resized_img = cv2.resize(img, (desired_width, desired_height))

        # Convert the resized image to a QImage
        height, width, channel = resized_img.shape
        bytes_per_line = 3 * width
        q_img = QImage(resized_img.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        return q_img

    def add_label_category_comboboxes(self):

        if self.use_label_files:

            # Define variables
            widget_height = 40
            widget_width_large = 200
            self.edit_buttons.clear()
            self.delete_buttons.clear()
            self.comboboxes = []
            self.layout_frames = []
            self.current_highlight = None

            # Create the QLabel with the same dimensions as the QComboBox
            label_text = "Select Insect Order:"
            label = QLabel(label_text)
            label.setMinimumWidth(widget_width_large)  # Set the minimum width of the label to match the QComboBox
            label.setMinimumHeight(widget_height)  # Set the minimum height of the label to match the QComboBox
            self.labels_box.addWidget(label)

            # Read each line in the txt file
            for i, label in enumerate(self.label_parameters_list):
                category = int(label[0])

                # Create box per label
                label_layout_item = QHBoxLayout()

                # Create frame for the Layout
                frame = QFrame()  # Create a QFrame
                frame.setStyleSheet("border: 1px solid LightGray;")
                frame.setLayout(label_layout_item)  # Set the QHBoxLayout as the layout of the QFrame
                self.layout_frames.append(frame)

                # Create the QComboBox
                order_combobox = QComboBox()

                # Adjust the size of the QComboBox
                order_combobox.setMinimumWidth(widget_width_large)  # Set the minimum width of the QComboBox
                order_combobox.setMinimumHeight(widget_height)  # Set the minimum height of the QComboBox

                # Add some padding to the QComboBox using style sheet
                border_color = self.color_names[min(len(self.color_names)-1, i)]
                order_combobox.setStyleSheet("padding: 10px;")  # Adjust the padding value as needed
                order_combobox.setStyleSheet(f"QComboBox {{ border: 3px solid {border_color}; }}")

                # Add the options to the QComboBox
                order_combobox.addItems(self.orders)

                # Record the combo
                self.comboboxes.append(order_combobox)

                # Set the default selected option based on text (e.g., "Coleoptera")
                default_order = self.category_dict.get(category, self.category_dict[0])
                order_combobox.setCurrentText(default_order)
                # Connect the currentIndexChanged signal to your function
                order_combobox.currentIndexChanged.connect(self.record_comboboxes_values)

                # Add the QPushButton with the same dimensions as the QLabel and QComboBox
                button_text = "Edit"
                edit_button = QPushButton(button_text)
                edit_button.setMinimumWidth(widget_width_large // 4)  # Set the minimum width of the button to match the QLabel and QComboBox
                edit_button.setMinimumHeight(widget_height)  # Set the minimum height of the button to match the QLabel and QComboBox

                # Connect the function to the button
                edit_button.clicked.connect(partial(self.on_edit_button_clicked, i))

                # Add the QPushButton with the same dimensions as the QLabel and QComboBox
                button_text = "Delete"
                delete_button = QPushButton(button_text)
                delete_button.setMinimumWidth(widget_width_large // 4)  # Set the minimum width of the button to match the QLabel and QComboBox
                delete_button.setMinimumHeight(widget_height)  # Set the minimum height of the button to match the QLabel and QComboBox

                # Connect the function to the button
                delete_button.clicked.connect(partial(self.on_delete_button_clicked, i))

                # Add the QComboBox to the QVBoxLayout
                label_layout_item.addWidget(order_combobox)

                # Add the button
                label_layout_item.addWidget(edit_button)

                # Add the button
                label_layout_item.addWidget(delete_button)

                self.labels_box.addWidget(frame)

            # Add the QPushButton with the same dimensions as the QLabel and QComboBox
            button_text = "Add label..."
            add_button = QPushButton(button_text)
            add_button.setMinimumWidth(widget_width_large + 66 + (widget_width_large // 2))  # Set the minimum width of the button to match the QLabel and QComboBox
            add_button.setMinimumHeight(widget_height)  # Set the minimum height of the button to match the QLabel and QComboBox

            # Add the button
            self.labels_box.addWidget(add_button)

            # Connect the function to the button
            add_button.clicked.connect(self.on_add_button_clicked)

            # Add stretch
            self.labels_box.addStretch()

            # Higlight
            #print(f"len is:{len(self.layout_frames)}")
            self.current_highlight = max(len(self.layout_frames)-1, 0)
            self.highlight_box(self.current_highlight)

    def highlight_box(self, index):
        if not index > len(self.layout_frames) - 1:
            if self.current_highlight is not None:
                frame = self.layout_frames[self.current_highlight]
                frame.setStyleSheet("border: 1px solid LightGrey;")

            frame = self.layout_frames[index]
            frame.setStyleSheet("border: 1px solid red;")
            self.current_highlight = index

    def record_comboboxes_values(self):

        if self.use_label_files and hasattr(self, 'label_parameters_list'):
            # Reverse the dictionary to create a new dictionary with the values as keys and keys as values
            reversed_dict = {value: key for key, value in self.category_dict.items()}

            for i, label in enumerate(self.label_parameters_list):
                if len(self.comboboxes) >= (i + 1):
                    selected_text = self.comboboxes[i].currentText()

                    # Retrieve the key from the reversed dictionary using the selected_text
                    selected_key = reversed_dict.get(selected_text)

                    self.label_parameters_list[i][0] = selected_key


    def clear_layout(self, layout):
        if layout:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget:
                    widget.setParent(None)
                    widget.deleteLater()
                else:
                    self.clear_layout(item.layout())

        # Hide the QRubberBand to clear it
        self.rubber_band.hide()

    def keyPressEvent(self, event):
        # Handle keypress events to update the status of the selected thumbnail
        selected_thumbnail = self.thumbnails_dictionary[self.current_index]
        if event.key() == Qt.Key_D:
            selected_thumbnail.setStatus(self.positive_status_label)
            self.image_statuses[self.current_index] = self.positive_status_label
            self.next_image()
        elif event.key() == Qt.Key_A:
            selected_thumbnail.setStatus(self.negative_status_label)
            self.image_statuses[self.current_index] = self.negative_status_label
            self.next_image()
        elif event.key() == Qt.Key_Space:
            selected_thumbnail.setStatus("None")
            self.image_statuses[self.current_index] = "None"
            self.next_image()
        elif event.key() == Qt.Key_Q:
            self.on_add_button_clicked()
        elif event.key() == Qt.Key_E:
            self.on_edit_button_clicked(self.current_highlight)
        elif event.key() == Qt.Key_W or event.key() == Qt.Key_S:
            if self.current_highlight is None:
                self.highlight_box(0)
            elif len(self.layout_frames) > 0:
                if event.key() == Qt.Key_W:
                    self.highlight_box((self.current_highlight - 1) % len(self.layout_frames))
                elif event.key() == Qt.Key_S:
                    self.highlight_box((self.current_highlight + 1) % len(self.layout_frames))
        elif event.key() == Qt.Key_Backspace or event.key() == Qt.Key_R:
            self.on_delete_button_clicked(self.current_highlight)
        elif event.key() == Qt.Key_X:
            self.prev_image()
        elif event.key() == Qt.Key_C:
            self.next_image()
        elif event.key() == Qt.Key_Return or event.key() == Qt.Key_F:
            if self.resize_in_progress and len(self.label_parameters_list) > 0:

                # Add the label data
                # Calculate the width and height of the initial ROI
                width = self.rubber_band.width()
                height = self.rubber_band.height()
                view_width = self.view.viewport().width()
                coco_box = (
                self.rubber_band.x() - ((view_width // 2) - (min(view_width, self.image_width) // 2)), self.rubber_band.y(), width, height)
                yolo_box = pbx.convert_bbox(coco_box, from_type="coco", to_type="yolo", image_size=(self.image_width, self.image_width))
                yolo_box = list(yolo_box)
                yolo_box.insert(0, 0)
                self.label_parameters_list[self.label_edited_index] = yolo_box
                self.rubber_band.hide()

                # Clear the label category layout
                self.clear_layout(self.labels_box)

                # Reload image
                self.load_image(self.current_index, False)

                self.resize_in_progress = False
        else:
            super().keyPressEvent(event)

    def on_add_button_clicked(self):
        # Set the flag variable to True when the button is clicked
        self.view.viewport().setCursor(self.cursor_cross)
        self.start_x = 0
        self.start_y = 0
        self.is_selecting_roi = True

    def on_edit_button_clicked(self, label_index: int):

        if not label_index >= len(self.label_parameters_list):
            self.label_edited_index = label_index
            yolo_box = [float(parameter) for parameter in self.label_parameters_list[label_index][1:]]
            coco_box = pbx.convert_bbox(yolo_box, from_type="yolo", to_type="coco", image_size=(self.image_width, self.image_width))
            view_width = self.view.viewport().width()
            view_height = self.view.viewport().height()
            self.start_x = coco_box[0] + (view_width // 2) - (min(view_width, self.image_width) // 2)
            self.start_y = coco_box[1] + ((view_height // 2) - (min(view_height, self.image_width) // 2))
            width = coco_box[2]
            height = coco_box[3]
            self.rubber_band.setGeometry(self.start_x, self.start_y, width, height)
            self.set_rubberband_color(label_index)
            self.rubber_band.show()
            self.resize_mode = None
            self.resize_in_progress = True

    def on_delete_button_clicked(self, label_index: int):

        if not label_index >= len(self.label_parameters_list):
            del self.label_parameters_list[label_index]

            # Here you want to update the txt file

            # Clear the label category layout
            self.clear_layout(self.labels_box)

            self.load_image(self.current_index, False)

    def set_rubberband_color(self, label_index):

        # Create a new palette and set the Highlight role color to correct color
        palette = self.rubber_band.palette()
        highlight_color = QColor(self.color_codes[min(len(self.color_codes) - 1, label_index)])
        palette.setBrush(QPalette.Highlight, QBrush(highlight_color))
        self.rubber_band.setPalette(palette)

    def on_mouse_press(self, event):
        if self.is_selecting_roi:
            # Start selecting the ROI
            self.start_x = event.pos().x()
            self.start_y = event.pos().y()
            self.rubber_band.setGeometry(self.start_x, self.start_y, 0, 0)
            self.set_rubberband_color(len(self.label_parameters_list))
            self.rubber_band.show()
            self.resize_mode = None
        else:
            # Check if the mouse is near any of the corners
            corners = ["NW", "NE", "SW", "SE"]
            for corner in corners:
                if self.is_near_corner(event.pos(), corner):
                    self.resize_mode = corner
                    break
                else:
                    self.resize_mode = None

    def is_near_corner(self, mouse_pos, corner):
        if corner == "NW":
            return ((mouse_pos.x() < self.rubber_band.x() + 10) and (mouse_pos.x() > self.rubber_band.x() - 10)) and ((mouse_pos.y() < self.rubber_band.y() + 10) and (mouse_pos.y() > self.rubber_band.y() - 10))
        elif corner == "NE":
            return ((mouse_pos.x() > self.rubber_band.x() + self.rubber_band.width() - 10) and (mouse_pos.x() < self.rubber_band.x() + self.rubber_band.width() + 10)) and ((mouse_pos.y() < self.rubber_band.y() + 10) and (mouse_pos.y() > self.rubber_band.y() - 10))
        elif corner == "SW":
            return ((mouse_pos.x() < self.rubber_band.x() + 10) and (mouse_pos.x() > self.rubber_band.x() - 10)) and ((mouse_pos.y() > self.rubber_band.y() + self.rubber_band.height() - 10) and (mouse_pos.y() < self.rubber_band.y() + self.rubber_band.height() + 10))
        elif corner == "SE":
            return ((mouse_pos.x() > self.rubber_band.x() + self.rubber_band.width() - 10) and (mouse_pos.x() < self.rubber_band.x() + self.rubber_band.width() + 10)) and ((mouse_pos.y() > self.rubber_band.y() + self.rubber_band.height() - 10) and (mouse_pos.y() < self.rubber_band.y() + self.rubber_band.height() + 10))
        else:
            return False

    def on_mouse_move(self, event):

        # Update the ROI selection as the mouse moves
        if self.is_selecting_roi:
            self.rubber_band.setGeometry(
                self.start_x, self.start_y, event.pos().x() - self.start_x, event.pos().y() - self.start_y)
        elif self.resize_mode == "NW":
            self.rubber_band.setGeometry(event.pos().x(), event.pos().y(), self.rubber_band.width() + (self.rubber_band.x() - event.pos().x()), self.rubber_band.height() + (self.rubber_band.y() - event.pos().y()))
        elif self.resize_mode == "NE":
            self.rubber_band.setGeometry(self.rubber_band.x(), event.pos().y(), self.rubber_band.width() + (event.pos().x() - (self.rubber_band.x() + self.rubber_band.width())), self.rubber_band.height() + (self.rubber_band.y() - event.pos().y()))
        elif self.resize_mode == "SW":
            self.rubber_band.setGeometry(event.pos().x(), self.rubber_band.y(), self.rubber_band.width() + (self.rubber_band.x() - event.pos().x()), event.pos().y() - self.rubber_band.y())
        elif self.resize_mode == "SE":
            self.rubber_band.setGeometry(self.rubber_band.x(), self.rubber_band.y(), event.pos().x() - self.rubber_band.x(), event.pos().y() - self.rubber_band.y())

    def on_mouse_release(self, event):

        # Reset the cursor appearance
        self.view.viewport().setCursor(self.cursor_default)

        # Get the position of the mouse release
        end_x = event.pos().x()
        end_y = event.pos().y()

        # If you were selecting a roi
        if self.is_selecting_roi:
            if self.resize_mode is None:

                # Calculate the width and height of the initial ROI
                width = end_x - self.start_x
                height = end_y - self.start_y

                # If you only clicked then this will not get triggered
                if width > 0 and height > 0:
                    # Hide the initial QRubberBand
                    self.rubber_band.hide()

                    # Show the QRubberBand for interactive resizing
                    self.rubber_band.setGeometry(self.start_x, self.start_y, width, height)
                    self.rubber_band.show()
                    self.is_selecting_roi = False

                    # Add the label data
                    view_width = self.view.viewport().width()
                    view_height = self.view.viewport().height()
                    coco_box = (self.start_x - ((view_width // 2) - (min(view_width, self.image_width) // 2)), self.start_y - ((view_height // 2) - (min(view_height, self.image_width) // 2)), width, height)
                    yolo_box = pbx.convert_bbox(coco_box, from_type="coco", to_type="yolo", image_size=(self.image_width, self.image_width))
                    yolo_box = list(yolo_box)
                    yolo_box.insert(0, 0)
                    self.label_parameters_list.append(yolo_box)

                    # Clear the label category layout
                    self.clear_layout(self.labels_box)

                    # Reload image
                    self.load_image(self.current_index, False)
        # If you were resizing
        else:
            if self.resize_mode is not None:

                self.resize_mode = None

    def resize_roi(self, mode):
        # Enable the resize mode for the given corner
        self.resize_mode = mode

    def read_label_file(self, label_file_path):

        # Define the list for storage
        self.label_parameters_list = []

        if os.path.exists(label_file_path):
            # Open the txt file and read the coords
            with open(label_file_path, "r", encoding="utf-8") as f:  # Specify the encoding
                # Read each line in the txt file
                for line in f:
                    parameters = line.strip().split()
                    self.label_parameters_list.append(parameters)

    def write_label_file(self, label_file_path):

        if os.path.exists(label_file_path):
            os.remove(label_file_path)

        if self.use_label_files and hasattr(self, 'label_parameters_list') and not self.label_parameters_list == []:

            # Open the file in write mode
            with open(label_file_path, "w", encoding="utf-8") as file:
                # Iterate through the nested lists
                for sublist in self.label_parameters_list:
                    # Join the elements of the sublist with spaces and write it as a line in the file
                    line = " ".join([str(int(sublist[0]))] + ["{:.6f}".format(float(x)) for x in sublist[1:]])
                    file.write(line + "\n")

    def draw_the_bbox(self, image):

        if self.use_label_files and hasattr(self, 'label_parameters_list') and not self.label_parameters_list == []:
            for i, label in enumerate(self.label_parameters_list):
                coords = np.array([float(coord) for coord in label[1:]]).reshape(-1, 4)
                bbox_coords = yolobbox2bbox(coords)

                # Draw rectangles on the image using the bounding box coordinates
                for box in bbox_coords:
                    box_left, box_top, box_right, box_bottom, _, _ = box
                    cv2.rectangle(image, (int(box_left), int(box_top)), (int(box_right), int(box_bottom)), self.colors[min(len(self.colors)-1, i)],
                                  2)
            return image
        else:
            return image

    def load_image(self, index, read_txt_file: bool = True):

        if 0 <= index < len(self.image_files):
            file_path = self.image_files[index]

            # Open the image
            img = self.open_image_uni(file_path)

            if self.use_label_files:
                # Construct the full path to the corresponding txt file
                txt_path = os.path.join(self.label_folder_path, os.path.splitext(os.path.basename(file_path))[0] + ".txt")

                if read_txt_file:
                    # Load the data of labels from the txt file
                    self.read_label_file(txt_path)

                # Draw the bboxes
                img = self.draw_the_bbox(img)

                # Add label category dropdowns
                self.add_label_category_comboboxes()

            # Convert the image to QPixmap and display it in QGraphicsView

            image_width_dip = self.screen_height // 1.7  # Use DIP value for thumbnail width
            self.image_width = int(image_width_dip * self.device_pixel_ratio)  # Convert DIP to physical pixels

            height, width, _ = img.shape
            bytes_per_line = 3 * width
            q_img = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            pixmap = QPixmap.fromImage(q_img)
            pixmap = pixmap.scaledToWidth(self.image_width, Qt.SmoothTransformation)
            self.scene.clear()
            self.scene.addPixmap(pixmap)

            # Update the image name label
            self.image_name_label.setText(f"{os.path.basename(file_path)}")
            self.progress_label.setText(f"{self.current_index+1}/{len(self.image_files)}")

    def next_image(self):

        # Record combo values
        self.record_comboboxes_values()

        # Write the label file
        file_path = self.image_files[self.current_index]
        if self.use_label_files:
            txt_path = os.path.join(self.label_folder_path, os.path.splitext(os.path.basename(file_path))[0] + ".txt")
            self.write_label_file(txt_path)

        self.previous_index = self.current_index  # Record the index of the previously displayed image
        self.current_index += 1
        if self.current_index >= len(self.image_files):
            self.current_index = len(self.image_files) - 1

        # Clear the label category layout
        self.clear_layout(self.labels_box)

        self.load_image(self.current_index)

        # Load and display the thumbnails for the next image
        self.load_thumbnails()

    def prev_image(self):

        # Record combo values
        self.record_comboboxes_values()

        # Write the label file
        file_path = self.image_files[self.current_index]
        if self.use_label_files:
            txt_path = os.path.join(self.label_folder_path, os.path.splitext(os.path.basename(file_path))[0] + ".txt")
            self.write_label_file(txt_path)

        self.previous_index = self.current_index  # Record the index of the previously displayed image
        self.current_index -= 1
        if self.current_index < 0:
            self.current_index = 0

        # Clear the label category layout
        self.clear_layout(self.labels_box)

        self.load_image(self.current_index)

        # Load and display the thumbnails for the previous image
        self.load_thumbnails()

    def thumbnail_clicked(self, index):

        # Record combo values
        self.record_comboboxes_values()

        # Write the label file
        file_path = self.image_files[self.current_index]
        if self.use_label_files:
            txt_path = os.path.join(self.label_folder_path, os.path.splitext(os.path.basename(file_path))[0] + ".txt")
            self.write_label_file(txt_path)

        # Load and display the clicked thumbnail's corresponding image
        self.previous_index = self.current_index
        self.current_index = index

        # Clear the label category layout
        self.clear_layout(self.labels_box)

        self.load_image(self.current_index)

        # Load and display the thumbnails for the current image and its neighbors
        self.load_thumbnails()

        # Update the selected_thumbnail_index when a thumbnail is clicked
        self.selected_thumbnail_index = index

    def closeEvent(self, event):

        # Write the label file
        file_path = self.image_files[self.current_index]
        if self.use_label_files:
            txt_path = os.path.join(self.label_folder_path, os.path.splitext(os.path.basename(file_path))[0] + ".txt")
            self.write_label_file(txt_path)

        # When the program is about to be closed, move the images and txt files based on their status

        # Create 'ok' and 'wrong' directories if they don't exist
        ok_dir = os.path.join(self.folder_path, self.positive_status_label)
        wrong_dir = os.path.join(self.folder_path, self.negative_status_label)
        os.makedirs(ok_dir, exist_ok=True)
        os.makedirs(wrong_dir, exist_ok=True)

        for i, status in self.image_statuses.items():
            image_file = self.image_files[i]
            if self.use_label_files:
                txt_file = os.path.join(self.label_folder_path, os.path.splitext(os.path.basename(image_file))[0] + ".txt")

            # Move images and txt files based on their status
            if status == self.positive_status_label and os.path.exists(image_file):
                shutil.move(image_file, os.path.join(ok_dir, os.path.basename(image_file)))
                if self.use_label_files and os.path.exists(txt_file):
                    shutil.move(txt_file, os.path.join(ok_dir, os.path.basename(txt_file)))
            elif status == self.negative_status_label and os.path.exists(image_file):
                shutil.move(image_file, os.path.join(wrong_dir, os.path.basename(image_file)))
                if self.use_label_files and os.path.exists(txt_file):
                    shutil.move(txt_file, os.path.join(wrong_dir, os.path.basename(txt_file)))

        event.accept()

    def run(self):
        self.show()

