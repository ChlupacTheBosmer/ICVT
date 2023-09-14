# This file contains the ICCS app class that inherits from ICVT AppAncestor class

# Import ICVT components
from modules.sort import sorting_gui
from modules.utility import utils
from modules.utility import validator
from modules.database import anno_data
from modules.video import vid_data
from modules.video import video_inter
from modules.vision import vision_AI
from modules.base import icvt
from modules.crop import crop
from modules.iccs.gui import ICCS_GUI, RightClickableQPushButton, DatabaseEntryWindow
from modules.iccs.mpv import MPVThread

# Import other modules
import pandas as pd
import cv2
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO


from PyQt5.QtWidgets import (QSplashScreen, QToolBar, QFrame, QSpacerItem, QPushButton, QRadioButton, QCheckBox,
                             QGridLayout, QSizePolicy, QVBoxLayout, QHBoxLayout, QScrollArea, QLabel, QWidget,
                             QMainWindow, QApplication, QWidget, QLineEdit, QPushButton, QVBoxLayout, QTableWidget,
                             QTableWidgetItem, QDesktopWidget)
from PyQt5.QtGui import QPixmap, QIcon, QImage
import PyQt5.QtGui as QtGui
import PyQt5.QtCore as QtCore
from PyQt5.QtCore import QTimer, QCoreApplication, QThread, Qt, pyqtSignal, pyqtSlot
import PyQt5
import sqlite3

# Part of python
import json
import math
import configparser
import os
pyqt = os.path.dirname(PyQt5.__file__)
os.environ['QT_PLUGIN_PATH'] = os.path.join(pyqt, "Qt5/plugins")

class ICCS(ICCS_GUI, icvt.AppAncestor):
    def __init__(self):

        # Define logger
        self.logger = self.log_define()

        # First log
        self.logger.info("Initializing ICCS - Insect Communities Crop Suite application class...")

        # Init basic instance variables and get config
        self.app_title = "Insect Communities Crop Suite"

        # Create config and read file
        self.config = self.config_create()
        self.config_read()

        # Define variables
        self.scanned_folders = []
        self.dir_hierarchy = False
        self.loaded = False
        self.crop_mode: int
        self.auto = 0
        self.frames = []
        self.modified_frames = []
        self.video_filepaths = []
        self.points_of_interest_entry = []
        self.button_images = []
        self.buttons = []
        self.gui_imgs = []
        self.cap = None
        self.image_details_dict = {}
        self.auto_processing: int = 0

        # Initiation functions - get directories and files
        self.scan_default_folders()

        # If video folder path not supplied ask user to specify it
        while not self.check_path():
            self.get_video_folder(1)

        # Ask the user to specify the Excel path
        self.get_excel_path(1, self.crop_mode)

        # Load the videos
        self.load_videos()

        # Construct ROI data
        self.reload_roi_entries()

        # Load video frames for buttons
        self.load_video_frames()

        # Create output folders
        utils.create_dir(self.output_folder)
        utils.create_dir(os.path.join(".", self.output_folder, "whole frames"))

        # Open window
        self.open_main_window()

    def config_create(self):
        # Set default values
        config = configparser.ConfigParser()
        config['Resource Paths'] = {
            'OCR_tesseract_path': 'C:/Program Files/Tesseract-OCR/tesseract.exe',
            'video_folder_path': '',
            'annotation_file_path': '',
            'output_folder': 'output'
        }
        config['OCR settings'] = {
            'x_coordinate': '0',
            'y_coordinate': '0',
            'width': '500',
            'height': '40'
        }
        config['Crop settings'] = {
            'crop_mode': '1',
            'crop_interval_frames': '30',
            'frames_per_visit': '0',
            'filter_visitors': '0',
            'yolo_processing': '0',
            'default_label_category': '6',
            'yolo_conf': '0.25',
            'randomize_interval': '0',
            'export_whole_frame': '0',
            'export_crops': '1',
            'crop_size': '640',
            'random_offset_range': '600',
            'filename_prefix': ''
        }
        config['Workflow settings'] = {
            'Scan_default_folders': '1'
        }

        # Check if settings_crop.ini exists, and create it with default values if not
        if not os.path.exists('settings_crop.ini'):
            with open('settings_crop.ini', 'w', encoding='utf-8') as configfile:
                config.write(configfile)
        return config

    def config_read(self):

        # Define logger
        self.logger.debug('Running function config_read()')

        try:
            # Read settings from settings_crop.ini
            self.config.read('settings_crop.ini', encoding='utf-8')

            # Get values from the config file
            self.ocr_tesseract_path = self.config['Resource Paths'].get('OCR_tesseract_path',
                                                                        'C:/Program Files/Tesseract-OCR/tesseract.exe').strip()
            self.video_folder_path = self.config['Resource Paths'].get('video_folder_path', '').strip()
            self.annotation_file_path = self.config['Resource Paths'].get('annotation_file_path', '').strip()
            self.output_folder = self.config['Resource Paths'].get('output_folder', 'output').strip()
            self.scan_folders = self.config['Workflow settings'].get('Scan_default_folders', '0').strip()

            # Get crop values from config
            self.crop_mode = int(self.config['Crop settings'].get('crop_mode', '1').strip())
            self.frame_skip = int(self.config['Crop settings'].get('crop_interval_frames', '30').strip())
            self.frames_per_visit = int(self.config['Crop settings'].get('frames_per_visit', '0').strip())
            self.filter_visitors = int(self.config['Crop settings'].get('filter_visitors', '0').strip())
            self.yolo_processing = int(self.config['Crop settings'].get('yolo_processing', '0').strip())
            self.default_label_category = int(self.config['Crop settings'].get('default_label_category', '6').strip())
            self.yolo_conf = float(self.config['Crop settings'].get('yolo_conf', '0.25').strip())
            self.randomize = int(self.config['Crop settings'].get('randomize_interval', '0').strip())
            self.whole_frame = int(self.config['Crop settings'].get('export_whole_frame', '0').strip())
            self.cropped_frames = int(self.config['Crop settings'].get('export_crops', '1').strip())
            self.crop_size = int(self.config['Crop settings'].get('crop_size', '640').strip())
            self.offset_range = int(self.config['Crop settings'].get('random_offset_range', '600').strip())
            self.prefix = self.config['Crop settings'].get('filename_prefix', '').strip()

            self.x_coordinate = int(self.config['OCR settings'].get('x_coordinate', '0').strip())
            self.y_coordinate = int(self.config['OCR settings'].get('y_coordinate', '0').strip())
            self.width = int(self.config['OCR settings'].get('width', '500').strip())
            self.height = int(self.config['OCR settings'].get('height', '40').strip())

            # Create compound tuple
            self.ocr_roi = (self.x_coordinate, self.y_coordinate, self.width, self.height)

        except ValueError:
            self.logger.warning('Invalid folder/file path or crop settings found in settings_crop.ini')

    def config_write(self):

        # Define logger
        self.logger.debug("Running function config_write()")
        config = self.config
        config.read('settings_crop.ini')

        # Update values in the config file
        config.set('Resource Paths', 'output_folder', self.output_folder)
        config.set('Workflow settings', 'Scan_default_folders', self.scan_folders)
        config.set('Crop settings', 'crop_mode', str(self.crop_mode))
        config.set('Crop settings', 'crop_interval_frames', str(self.frame_skip))
        config.set('Crop settings', 'frames_per_visit', str(self.frames_per_visit))
        config.set('Crop settings', 'filter_visitors', str(self.filter_visitors))
        config.set('Crop settings', 'yolo_processing', str(self.yolo_processing))
        config.set('Crop settings', 'default_label_category', str(self.default_label_category))
        config.set('Crop settings', 'yolo_conf', str(self.yolo_conf))
        config.set('Crop settings', 'randomize_interval', str(self.randomize))
        config.set('Crop settings', 'export_whole_frame', str(self.whole_frame))
        config.set('Crop settings', 'export_crops', str(self.cropped_frames))
        config.set('Crop settings', 'crop_size', str(self.crop_size))
        config.set('Crop settings', 'random_offset_range', str(self.offset_range))
        config.set('Crop settings', 'filename_prefix', str(self.prefix))

        config.set('OCR settings', 'x_coordinate', str(self.x_coordinate))
        config.set('OCR settings', 'y_coordinate', str(self.y_coordinate))
        config.set('OCR settings', 'width', str(self.width))
        config.set('OCR settings', 'height', str(self.height))

        # Save changes to the config file
        with open('settings_crop.ini', 'w', encoding='utf-8') as configfile:
            config.write(configfile)

    def reload_roi_entries(self):
        self.logger.debug('Running function reload_points_of_interest()')

        # Clear the array of POIs and reconstruct it with empty lists.
        self.points_of_interest_entry.clear()
        self.points_of_interest_entry = [[[], filepath] for filepath in self.video_filepaths]

    def load_video_frames(self):
        # Define logger
        self.logger.debug(f"Running function load_video_frames()")

        # Define video formats
        video_formats = ['.mp4', '.avi', '.mkv']

        # Loop through each file in folder
        self.frames = []
        if utils.check_path(self.video_folder_path, 0):
            for filename in os.listdir(self.video_folder_path):
                if any(filename.endswith(format) for format in video_formats):
                    video_path = os.path.join(self.video_folder_path, filename)
                    try:
                        video_file_object = video_inter.VideoFileInteractive(filepath=video_path, initiate_start_and_end_times=False)
                        frame = video_file_object.read_video_frame(25, False)[0][3]
                        if frame is not None:
                            self.frames.append(frame)
                        else:
                            # If the read operation fails, add a default image
                            default_image = cv2.imread('resources/img/nf.png')
                            self.frames.append(default_image)
                    except (cv2.error, OSError) as e:
                        self.logger.error(f"Error: Failed to process video '{filename}': {e}")
                        default_image = cv2.imread('resources/img/nf.png')
                        self.frames.append(default_image)
        else:
            self.logger.error("Error: Invalid video folder path")
            messagebox.showerror("Error", "Invalid video folder path")

    ######################################### SECTION DEALS WITH THE BACKEND ###############################################
    ########################################################################################################################
    ########################################################################################################################
    ########################################################################################################################
    ########################################################################################################################

    ######################################### CROP FUNCTIONALITY ######################################################

    # async def capture_crop(self, frame, point):
    #
    #    # Define logger
    #     self.logger.debug(f"Running function capture_crop({point})")
    #
    #     # Prepare local variables
    #     x, y = point
    #
    #     # Add a random offset to the coordinates, but ensure they remain within the image bounds
    #     # DONE: Implement Milesight functionality
    #     frame_width, frame_height = self.video_file_object.get_frame_shape()
    #
    #     # Check if any of the dimensions is smaller than crop_size and if so upscale the image to prevent crops smaller than desired crop_size
    #     if frame_height < self.crop_size or frame_width < self.crop_size:
    #         # Calculate the scaling factor to upscale the image
    #         scaling_factor = self.crop_size / min(frame_height, frame_width)
    #
    #         # Calculate the new dimensions for the upscaled frame
    #         new_width = int(round(frame_width * scaling_factor))
    #         new_height = int(round(frame_height * scaling_factor))
    #
    #         # Upscale the frame using cv2.resize with Lanczos upscaling algorithm
    #         frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    #
    #     # Get the new frame size
    #     frame_height, frame_width = frame.shape[:2]
    #
    #     # Calculate the coordinates for the area that will be cropped
    #     x_offset = random.randint(-self.offset_range, self.offset_range)
    #     y_offset = random.randint(-self.offset_range, self.offset_range)
    #     x1 = max(0, min(((x - self.crop_size // 2) + x_offset), frame_width - self.crop_size))
    #     y1 = max(0, min(((y - self.crop_size // 2) + y_offset), frame_height - self.crop_size))
    #     x2 = max(self.crop_size, min(((x + self.crop_size // 2) + x_offset), frame_width))
    #     y2 = max(self.crop_size, min(((y + self.crop_size // 2) + y_offset), frame_height))
    #
    #     # Crop the image
    #     crop = frame[y1:y2, x1:x2]
    #
    #     # Convert to correct color space
    #     crop_img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    #     if crop_img.shape[2] == 3:
    #         crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
    #
    #     # Return the cropped image and the coordinates for future reference
    #     return crop_img, x1, y1, x2, y2
    #
    # async def generate_frames(self, frame, success, tag, index, frame_number_start):
    #
    #     # Define logger
    #     self.logger.debug(f"Running function generate_frames({index})")
    #
    #     # Prepare name elements
    #     filename_parts = tag[:-4].split("_")
    #     recording_identifier = "_".join(filename_parts[:-3])
    #     timestamp = "_".join(filename_parts[-3:])
    #
    #     # Define local variables
    #     crop_counter = 1
    #     frame_skip_loc = self.frame_skip
    #
    #     # Calculate the frame skip variable based on the limited number of frames per visit
    #     if self.frames_per_visit > 0:
    #         frame_skip_loc = int((self.visit_duration * self.fps) // self.frames_per_visit)
    #         if frame_skip_loc < 1:
    #             frame_skip_loc = 1
    #
    #     # Loop through the video and crop y images every n-th frame
    #     frame_count = 0
    #     image_paths = []
    #
    #     while success:
    #         # Crop images every n-th frame
    #         if int(frame_count % frame_skip_loc) == 0:
    #             for i, point in enumerate(self.points_of_interest_entry[index][0]):
    #                 if self.cropped_frames == 1:
    #                     crop_img, x1, y1, x2, y2 = await self.capture_crop(frame, point)
    #                     frame_number = frame_number_start + frame_count
    #                     roi_number = i + 1
    #                     visit_number = self.visit_index
    #                     image_name = f"{self.prefix}{recording_identifier}_{timestamp}_{roi_number}_{frame_number}_{visit_number}_{x1},{y1}_{x2},{y2}.jpg" #Now the output images will be ordered by the ROI therefore one will be able to delete whole segments of pictures.
    #                     image_path = os.path.join(self.output_folder, image_name)
    #                     #image_path = f"./{self.output_folder}/{self.prefix}{recording_identifier}_{timestamp}_{frame_number_start + frame_count}_{crop_counter}_{i + 1}_{x1},{y1}_{x2},{y2}.jpg"
    #                     cv2.imwrite(image_path, crop_img)
    #                     image_paths.append(image_path)
    #                     self.image_details_dict[image_name] = [image_path, frame_number, roi_number, visit_number, 0]
    #             if self.whole_frame == 1:
    #                 frame_number = frame_number_start + frame_count
    #                 visit_number = self.visit_index
    #                 image_name = f"{self.prefix}{recording_identifier}_{timestamp}_{frame_number}_{visit_number}_whole.jpg"
    #                 image_path = os.path.join(self.output_folder, "whole frames", image_name)
    #                 #image_path = f"./{self.output_folder}/whole frames/{self.prefix}{recording_identifier}_{timestamp}_{frame_number_start + frame_count}_{crop_counter}_whole.jpg"
    #                 cv2.imwrite(image_path, frame)
    #             crop_counter += 1
    #
    #         # If the random frame skip interval is activated add a random number to the counter or add the set frame skip interval
    #         if self.randomize == 1:
    #             if (frame_skip_loc - frame_count == 1):
    #                 frame_count += 1
    #             else:
    #                 frame_count += random.randint(1, max((frame_skip_loc - frame_count), 2))
    #         else:
    #             frame_count += frame_skip_loc
    #
    #         # Read the next frame
    #         # DONE: Implement Milesight functionality
    #         frame_to_read = frame_number_start + frame_count
    #         success, frame = self.video_file_object.read_video_frame(frame_to_read)
    #
    #         # If the frame count is equal or larger than the amount of frames that comprises the duration of the visit end the loop
    #         if not (frame_count < (self.visit_duration * self.fps)-1):
    #             # Release the video capture object and close all windows
    #             # DONE: Implement Milesight functionality
    #             if not self.video_file_object.video_origin == "MS":
    #                 self.video_file_object.cap.release()
    #             cv2.destroyAllWindows()
    #             break
    #
    #     # Return the resulting list of image paths for future reference
    #     return image_paths

    def yolo_preprocessing(self, frames, valid_annotations_array, index):

        # Define logger
        self.logger.debug(f"Running function yolo_preprocessing({frames[0].name})")

        # Define list
        list = [frame.frame for frame in frames]

        # Define and run the model
        model = YOLO('resources/yolo/best.pt')
        results = model(list, save=False, imgsz=self.crop_size, conf=self.yolo_conf, save_txt=False, max_det=2,
                        stream=True)
        for i, (result, frame) in enumerate(zip(results, frames)):
            boxes = result.boxes.data
            image_name = frame.name
            #original_path = os.path.join(img_paths[i])
            utils.create_dir(f"{self.output_folder}/empty")
            utils.create_dir(f"{self.output_folder}/visitor")
            utils.create_dir(f"{self.output_folder}/visitor/labels")
            empty_path = os.path.join(f"{self.output_folder}", "empty", image_name)
            visitor_path = os.path.join(f"{self.output_folder}", "visitor", image_name)
            label_path = os.path.join(f"{self.output_folder}", "visitor", "labels", image_name[:-4])

            if len(result.boxes.data) > 0:
                #shutil.move(original_path, visitor_path)
                frame.save(os.path.join(f"{self.output_folder}", "visitor"))
                self.image_details_dict[image_name][0] = visitor_path
                self.image_details_dict[image_name][4] = 1
                frame.visitor_detected = True
                with open(f"{label_path}.txt", 'w') as file:
                    # Write the box_data to the file
                    txt = []
                    lst = result.boxes.xywhn[0].tolist()
                    for item in lst:
                        txt_item = round(item, 6)
                        txt.append(txt_item)
                    txt = str(txt)
                    if any(len(row) < 7 for row in valid_annotations_array):
                        visitor_category = self.default_label_category
                    else:
                        visitor_category = valid_annotations_array[index][6]
                    file.write(f"{visitor_category} {txt.replace('[', '').replace(']', '').replace(',', '')}")
            else:
                #shutil.move(original_path, empty_path)
                frame.save(os.path.join(f"{self.output_folder}", "empty"))
                self.image_details_dict[image_name][0] = empty_path
                self.image_details_dict[image_name][4] = 0
                frame.visitor_detected = False

    def filter_array_by_visitors(self, valid_annotations_array):

        # Define logger
        self.logger.debug(f"Running function filter_array_by_visitors()")

        # Create filter window
        filter_window = tk.Tk()
        filter_window.title("Filter Visitors")
        filter_window.wm_attributes("-topmost", 1)

        def apply_filter():
            results = []
            for i, var in enumerate(checkbox_vars):
                checkbox_value = var.get()
                checkbox_text = checkboxes[i].cget("text")
                dropdown_value = selected_items[i].get()
                if not dropdown_value[0].isdigit():
                    dropdown_value = "0. Default"
                results.append([int(checkbox_value), checkbox_text, int(dropdown_value[0])])
            results = [row for row in results if row[0] != 0]
            if self.yolo_processing == 1:
                for row in results:
                    if row[2] > 1 or row[2] < 0:
                        filter_window.quit()
                        filter_window.destroy()
                        messagebox.showinfo("Warning",
                                            f"One of the selected groups of visitors was not assigned a correct category for labelling. Please try again.")
                        self.filtered_array = []
                        return

            selected_values = [value.get() for value in checkbox_vars]
            self.filtered_array = []
            for row in valid_annotations_array:
                if row[5] in [result[1] for result in results]:
                    matching_result = next(result for result in results if result[1] == row[5])
                    filtered_row = row + [matching_result[2]]
                    self.filtered_array.append(filtered_row)
            filter_window.quit()
            filter_window.destroy()

        def handle_selection(index, selection):
            if selection in allowed_items:
                selected_items[index].set(selection)
                label_values[index].config(text=selection)
            else:
                selected_items[index].set("")
                label_values[index].config(text="")

        # Get unique values from column index 5
        column_5_values = [row[5] for row in valid_annotations_array]
        unique_values = set(column_5_values)
        allowed_items = ["0. Default", "1. Lepidoptera"]
        if len(unique_values) > 0:
            checkbox_vars = []
            checkboxes = []
            selected_items = []
            label_values = []
            for i, value in enumerate(unique_values):
                var = tk.StringVar(filter_window, value=0)  # Pass 'filter_window' as the 'master' argument
                checkbox_vars.append(var)

                checkbox = tk.Checkbutton(filter_window, text=value, variable=var)
                checkbox.grid(row=i, column=0, sticky='w')
                checkboxes.append(checkbox)

                selected_item = tk.StringVar(filter_window)
                selected_item.set("0. Default")
                selected_items.append(selected_item)

                option_menu = tk.OptionMenu(filter_window, selected_item, *allowed_items,
                                            command=lambda selection, index=i: handle_selection(index, selection))
                option_menu.config(width=35)
                option_menu.grid(row=i, column=1, sticky='nsew')

                label_value = tk.Label(filter_window, text="", width=40)
                label_value.grid(row=i, column=2, sticky='nsew')
                label_values.append(label_value)

            apply_button = tk.Button(filter_window, text="Apply Filter", command=apply_filter)
            apply_button.grid(row=len(unique_values), column=0, columnspan=3, sticky='nsew', pady=(30, 0))

            # Set properties and start window
            filter_window.update()
            screen_width = filter_window.winfo_screenwidth()
            screen_height = filter_window.winfo_screenheight()
            window_width = filter_window.winfo_reqwidth()
            window_height = filter_window.winfo_reqheight()
            x_pos = int((screen_width - window_width) / 2)
            y_pos = int((screen_height - window_height) / 2)
            filter_window.geometry(f"{window_width}x{window_height}+{x_pos}+{y_pos}")
            filter_window.mainloop()
        else:
            self.logger.info("No visitor items to filter by. Processing all visits.")
            self.filtered_array = []
            for row in valid_annotations_array:
                filtered_row = row + [0]
                self.filtered_array.append(filtered_row)
            # print(self.filtered_array)  # Do something with the filtered array
            filter_window.quit()
            filter_window.destroy()

    def generate_frames_for_visits(self, valid_annotations_array: list, video_files: list,
                                   frames_per_visit: int = 20, frame_skip: int = 15) -> dict:
        frames_per_video_dict = {}
        visitor_category_dict = {}
        for iter_num, (_, _, video_filepath, *_) in enumerate(valid_annotations_array):
            matching_annotations = [
                (iter_num, duration, time_of_visit, fp, video_start_time, *_)
                for duration, time_of_visit, fp, video_start_time, *_
                in valid_annotations_array if fp == video_filepath
            ]
            if len(valid_annotations_array[iter_num]) > 0:
                visitor_category_dict[iter_num] = valid_annotations_array[iter_num][5]



            list_of_visit_frames = []
            list_of_visit_number = []
            for visit_number, duration, time_of_visit, video_filepath, video_start_time, *_ in matching_annotations:
                try:
                    fps, total_frames = next((video_file.fps, video_file.total_frames) for video_file in video_files if
                                             video_file.filename == os.path.basename(video_filepath))
                except StopIteration:
                    self.logger.warning("No video file found for visit.")
                    continue
                time_from_start = int(
                    (pd.to_datetime(time_of_visit, format='%Y%m%d_%H_%M_%S') - video_start_time).total_seconds())
                first_frame = time_from_start * fps
                adjusted_visit_duration = (min(
                    time_from_start * fps + int(duration) * fps,
                    total_frames
                ) - time_from_start * fps) // fps
                last_frame = first_frame + (adjusted_visit_duration * fps)
                list_of_visit_frames += list(range(first_frame, last_frame, frame_skip if frames_per_visit < 1 else (
                            (adjusted_visit_duration * fps) // frames_per_visit)))
                list_of_visit_number += [visit_number for _ in list_of_visit_frames]

            frames_per_video_dict[os.path.basename(video_filepath)] = (tuple(list_of_visit_frames),
                                                                       tuple(list_of_visit_number),
                                                                       visitor_category_dict)

        return frames_per_video_dict

    def crop_engine(self):

        # Define logger
        self.logger.debug("Running function crop_engine()")

        # Define variables
        root = self.main_window
        self.image_details_dict = {}

        # Ask to confirm whether the process should begin
        result = utils.ask_yes_no("Do you want to start the cropping process?")
        if result:
            self.logger.info("Initializing the cropping engine...")

            # Check if some ROIs were selected
            if len(self.points_of_interest_entry[0][0]) == 0:
                messagebox.showinfo("Warning", "No regions of interest selected. Please select at least one ROI.")
                self.reload(True, False)
                return

            # Define arrays
            valid_annotations_array = []
            valid_annotation_data_entry = []

            # Close window and start cropping
            self.logger.debug(f"Start cropping on the following videos: {self.video_filepaths}")
            root.hide()

            # This code is for cases when cropping runs according to either watchers, or croplog Excel files
            if not self.crop_mode == 3:

                # Check validity of paths
                video_ok = utils.check_path(self.video_folder_path, 0)
                excel_ok = utils.check_path(self.annotation_file_path, 1)
                if not video_ok or not excel_ok:
                    messagebox.showinfo("Warning",
                                        f"Unspecified path to a video folder or a valid Excel file.")
                    self.reload(True, False)
                    return

                # Gather video and excel data - Watcher file
                if self.crop_mode == 1:
                    # Load data from excel
                    excel = anno_data.Annotation_watcher_file(self.annotation_file_path, True, True, True, True)
                    annotation_data_array = excel.dataframe.loc[:, ['duration', 'ts']].values.tolist()
                    visitor_id = excel.dataframe.loc[:, ['vis_id']].values.tolist()

                # Gather video and excel data - Croplog file
                if self.crop_mode == 2:
                    # Load data from excel
                    excel = anno_data.Annotation_custom_file(self.annotation_file_path)
                    annotation_data_array = excel.dataframe.loc[:, ['duration', 'ts']].values.tolist()

                # If no annotations were extracted end cropping process
                if annotation_data_array is None:
                    messagebox.showinfo("Warning",
                                        f"Attempted to fix errors in the selected Excel file. Attempt failed. Please fix"
                                        f" the errors manually and try again.")
                    self.reload(True, False)
                    return

                # Filter video filepaths to only those relevant for the annotations that are to be processed
                sorted_video_filepaths = self.get_relevant_video_paths(self.video_filepaths, annotation_data_array)

                # If no relevant videos are found inform the user and end the cropping process.
                if sorted_video_filepaths is None:
                    messagebox.showinfo("Warning",
                                        f"Attempted to locate relevant video files but none were found. Please check "
                                        f"whether you are using the right recordings for your selected annotations file "
                                        f"and vice versa. If the problem persists check the Excel table for mismatch in "
                                        f"the recorded date and the names of the video files.")
                    self.reload(True, False)
                    return

                # Get video data
                video_data, video_files = self.get_video_data(sorted_video_filepaths, True)

                # Log the information that are fed into the rest of the engine
                self.logger.debug(f"Start cropping according to the following annotations: {annotation_data_array}")
                self.logger.debug(f"Start cropping with the following video data: {video_data}")

                # Construct the valid annotation data array which contains visit data coupled with the path to the
                # video containing the visit
                # valid annotation entries in array have the following format:
                # [duration, time_of_visit, video_filepath, video_start_time, video_end_time]
                valid_annotations_array = self.construct_valid_annotation_array(annotation_data_array, video_data)

                # If following Watchers file append also visitor id
                if self.crop_mode == 1:
                    for annotation, vis_id in zip(valid_annotations_array, visitor_id):
                        annotation += vis_id

                # If relevant allow user to choose to filter the visits by visitor type.
                if self.filter_visitors == 1 and self.crop_mode == 1:
                    self.filter_array_by_visitors(valid_annotations_array)
                    if len(self.filtered_array) > 0:
                        valid_annotations_array = self.filtered_array
                    else:
                        self.logger.info("No visitors of the selected type found.")
                        self.reload(True, False)
                        return

                # Generate which frames should be exported for each video
                self.generate_frames_for_visits(valid_annotations_array, video_files, self.frame_skip,
                                                self.frames_per_visit)

                # Process the visits and generate cropped images
                for index, visit_entry in enumerate(valid_annotations_array):
                    self.selective_mode_crop_generation(valid_annotations_array, index, video_files)

            # This is for when the crop_mode is 3 and no annotation file is supplied
            else:

                # Check validity of paths
                video_ok = utils.check_path(self.video_folder_path, 0)
                if not video_ok:
                    messagebox.showinfo("Warning",
                                        f"Unspecified path to a video folder.")
                    self.reload(True, False)
                    return

                # The whole frame settings is artificially altered to allow for whole frame generation - messy
                orig_wf = self.whole_frame
                self.whole_frame = 1

                # Starting from the second frame of every video frames are generated.
                for i, video_filepath in enumerate(self.video_filepaths):

                    # Define relevant video object
                    self.video_file_object = video_inter.VideoFileInteractive(video_filepath, self.main_window, self.ocr_roi, False)
                    total_frames = self.video_file_object.total_frames
                    visit_duration = total_frames // self.video_file_object.fps
                    frame_number_start = 2

                    # Iterate over the list of lists to find which points of interest entry index to summon for each visit-video combo
                    roi_index = 0
                    for ix, sublist in enumerate(self.points_of_interest_entry):
                        if sublist[1] == video_filepath:
                            # Found the specific filepath
                            roi_index = ix
                            break
                    else:
                        # Filepath not found in any of the nested lists
                        self.logger.warning("No ROI entry found for a video file. Defaults to index 0")

                    frame_generator = crop.generate_frames(self, self.video_file_object, self.points_of_interest_entry[roi_index][0], frame_number_start, visit_duration, name_prefix=self.prefix)

                    for frame in frame_generator:
                        success, image_path = frame.save(self.output_folder)
                        self.image_details_dict[frame.name] = [image_path, frame.frame_number, frame.roi_number, frame.visit_number, int(frame.visitor_detected)]
                self.whole_frame = orig_wf

        # When done, reload the application
        self.reload(True, False)

    def selective_mode_crop_generation(self, valid_annotations_array, index, video_files):
        self.logger.info(f"Processing visit {index + 1}")

        # Define index which will be shared to sync the stage of the process
        self.visit_index = index

        # Defien entry
        visit_entry = valid_annotations_array[index]

        # Define the variables
        visit_duration, visit_timestamp, video_filepath, video_start_time, *_ = visit_entry

        # Pick the correct video file object from the list - Filter the list to find the object with matching filepath
        self.video_file_object = next((video for video in video_files if video.filepath == video_filepath),
                                      None)

        # Turn timestamp into datetime and calculate how many seconds from the start_time of the video recording does the visit take place
        visit_time = pd.to_datetime(visit_timestamp, format='%Y%m%d_%H_%M_%S')
        visit_time_from_start = (visit_time - video_start_time).total_seconds()

        # The actual duration is taken as limited by the end of the video, therefore cropping wont carry on for longer than one entire video file.
        visit_duration = (min(((visit_time_from_start * self.video_file_object.fps) + (
                    int(visit_duration) * self.video_file_object.fps)),
                              self.video_file_object.total_frames) - (
                                      visit_time_from_start * self.video_file_object.fps)) // self.video_file_object.fps

        # First frame to capture - start of the visit
        frame_number_start = int(visit_time_from_start * self.video_file_object.fps)

        # Find the index of the specific filepath in the list of lists
        try:
            roi_index = next(ix for ix, sublist in enumerate(self.points_of_interest_entry) if
                             sublist[1] == video_filepath)
        except StopIteration:
            roi_index = 0
            self.logger.warning("No ROI entry found for a video file. Defaults to index 0")

        # Generate frames and get back their paths
        frames_list = []
        frame_generator = crop.generate_frames(self, self.video_file_object,
                                               self.points_of_interest_entry[roi_index][0], frame_number_start,
                                               visit_duration, index, self.frame_skip, self.frames_per_visit,
                                               bool(self.cropped_frames), bool(self.whole_frame))

        # If relevant preprocess the images using yolo
        for frame in frame_generator:
            self.image_details_dict[frame.name] = ["", frame.frame_number, frame.roi_number, frame.visit_number,
                                                   int(frame.visitor_detected)]
            frames_list.append(frame)

        # Wait until all frames are generated and run YOLO OD on them
        if self.yolo_processing == 1:
            self.yolo_preprocessing(frames_list, valid_annotations_array, index)

    def sort_engine(self):

        # Define logger
        self.logger.debug("Running function sort_engine()")

        # Ask user if they want to run sorting script
        run_sorting = utils.ask_yes_no("Do you want to Running function the sorting script on the generated images?")
        if run_sorting:
            self.logger.info("Initializing the sorting engine...")
            # sort_script_path = "sort.py"
            # if os.path.exists(sort_script_path):
            #     # subprocess.run(['python', f'{sort_script_path}'])
            #     subprocess.call([sys.executable, f'{sort_script_path}', "--subprocess"])
            # else:
            #     self.logger.warning("Sorting script not found.")

            if self.image_details_dict == {}:
                self.image_details_dict = sorting_gui.gather_image_details(self.output_folder)
            sorting_gui.survey_visits_for_sorting(self.image_details_dict)

    ######################################### SECTION DEALS WITH THE GUI ###################################################
    ########################################################################################################################
    ########################################################################################################################
    ########################################################################################################################
    ########################################################################################################################

    def reload(self, is_window_already_loaded: bool, reload_POIs: bool):

        # Define logger
        logger = self.logger
        logger.debug(f"Running function reload({is_window_already_loaded}, {reload_POIs})")

        # Define variables
        root = self.main_window

        # Reload videos, clear POIs, reload video frames
        self.load_videos()
        if reload_POIs:
            self.reload_roi_entries()
        self.load_video_frames()
        self.close_main_window()
        self.loaded = is_window_already_loaded
        self.open_main_window()

    ######################################### SECTION DEALS WITH GUI BACKEND ###########################################
    ####################################################################################################################
    ####################################################################################################################
    ####################################################################################################################
    ####################################################################################################################

    def change_excel_path(self):

        # Define logger
        self.logger.debug(f'Running function change_video_folder()')

        # Get new Excel path
        self.annotation_file_path = utils.get_excel_path(self.annotation_file_path, 0, self.video_folder_path,
                                                         self.crop_mode)

        # Reload the window with the new input
        self.reload(False, True)

    def change_video_folder(self):
        self.logger.debug(f'Running function change_video_folder()')

        # Set loaded to false as when video folder is changed, the GUI must be reloaded.
        self.loaded = False

        # Get new video folder and reload the GUI
        self.video_folder_path, self.scanned_folders, self.dir_hierarchy = utils.get_video_folder(
            self.video_folder_path, 0)
        self.reload(False, True)

    def switch_folder(self, which):

        # Define logger
        self.logger.debug(f'Running function switch_folder({which})')

        # Set loaded to false as when video folder is changed, the GUI must be reloaded.
        self.loaded = False

        # Check if there is another folder either left or right in the directory hierarchy and if yes, then switch and reload.
        index = self.scanned_folders.index(os.path.basename(os.path.normpath(self.video_folder_path)))
        if index > 0 and which == "left":
            self.video_folder_path = os.path.join(os.path.dirname(self.video_folder_path),
                                                  self.scanned_folders[index - 1])
            self.reload(False, True)
        if (index + 1) < len(self.scanned_folders) and which == "right":
            self.video_folder_path = os.path.join(os.path.dirname(self.video_folder_path),
                                                  self.scanned_folders[index + 1])
            self.reload(False, True)

    def open_menu(self):

        def save_fields():

            end_values = []
            variable_mappings = [
                ('self.output_folder', str),
                ('self.scan_folders', str),
                ('self.prefix', str),
                ('self.crop_mode', int),
                ('self.frame_skip', int),
                ('self.frames_per_visit', int),
                ('self.filter_visitors', int),
                ('self.yolo_processing', int),
                ('self.default_label_category', int),
                ('self.yolo_conf', float),
                ('self.randomize', int),
                ('self.whole_frame', int),
                ('self.cropped_frames', int),
                ('self.crop_size', int),
                ('self.offset_range', int)
            ]

            # Get end values from the entry fields
            end_values = [var_type(fields[i].get()) for i, (var_name, var_type) in enumerate(variable_mappings)]

            # Update the variables with the new values from the fields
            for i, (var_name, var_type) in enumerate(variable_mappings):
                exec(f"{var_name} = end_values[i]", globals(), locals())

            # Write the new values into the config
            self.config_write()

            # Create new output dirs
            utils.create_dir(self.output_folder)
            utils.create_dir(os.path.join(".", self.output_folder, "whole frames"))

            # Destroy menu window
            window.destroy()

        # Define logger
        self.logger.debug("Running function open_menu()")

        # Create the Tkinter window
        window = tk.Tk()
        window.title("Menu")
        window.wm_attributes("-topmost", 1)

        # Create the labels and input fields
        label_text = ["Output folder path:", "Scan default folders:", "Filename prefix:", "Default crop mode:",
                      "Frames to skip:", "Frames per visit:", "Filter visitors:", "Yolo processing",
                      "Default label category", "Yolo conf. tresh.", "Randomize interval:", "Export whole frames:",
                      "Export cropped frames:",
                      "Crop size:", "Offset size:"]
        labels = []
        fields = []
        outer_frame = tk.Frame(window, pady=20)
        outer_frame.pack(side=tk.TOP, fill=tk.BOTH)
        for i in range(len(label_text)):
            label = tk.Label(outer_frame, text=f"{label_text[i]}")
            label.grid(row=i, column=0, padx=10)
            labels.append(label)

            field = tk.Entry(outer_frame, width=120)
            field.grid(row=i, column=1, padx=10)
            fields.append(field)

        # Create save button
        save_button = tk.Button(outer_frame, text="Save", command=save_fields)
        save_button.grid(row=16, column=0, columnspan=2)

        # Set initial values for the input fields
        self.initial_values = [self.output_folder, self.scan_folders, self.prefix, self.crop_mode, self.frame_skip,
                               self.frames_per_visit, self.filter_visitors,
                               self.yolo_processing, self.default_label_category, self.yolo_conf, self.randomize,
                               self.whole_frame,
                               self.cropped_frames, self.crop_size, self.offset_range]
        for i in range(len(self.initial_values)):
            fields[i].insert(0, str(self.initial_values[i]))

        # Start the Tkinter event
        window.update()
        screen_width = window.winfo_screenwidth()
        screen_height = window.winfo_screenheight()
        window_width = window.winfo_reqwidth()
        window_height = window.winfo_reqheight()
        x_pos = int((screen_width - window_width) / 2)
        y_pos = int((screen_height - window_height) / 2)
        window.geometry(f"{window_width}x{window_height}+{x_pos}+{y_pos}")
        window.mainloop()

    def open_ocr_roi_gui(self, video_filepath):

        # function that will open a frame with an image and prompt the user to drag a rectangle around the text and the
        # top left and bottom right coordinates will be saved in the settings_crop.ini file
        self.logger.debug(f'Running function open_ocr_roi_gui({video_filepath})')

        def draw_rectangle(event, x, y, flags, param):
            # DONE: Implement Milesight functionality
            frame = self.video_file_object.read_video_frame(1, False)[0][3]
            if event == cv2.EVENT_LBUTTONDOWN:
                self.x_coordinate = x
                self.y_coordinate = y
            elif event == cv2.EVENT_LBUTTONUP:
                self.width = x - self.x_coordinate
                self.height = y - self.y_coordinate
                self.text_roi = (self.x_coordinate, self.y_coordinate, self.width, self.height)
                cv2.rectangle(frame, (self.x_coordinate, self.y_coordinate), (x, y), (0, 255, 0), 2)
                cv2.imshow('image', frame)
                # cv2.waitKey(0)

        # Read settings from settings_crop.ini
        self.config.read('settings_crop.ini', encoding='utf-8')
        # Create video object
        self.video_file_object = video_inter.VideoFileInteractive(video_filepath, self.main_window, self.ocr_roi, False)
        try:
            self.x_coordinate = int(self.config['OCR settings'].get('x_coordinate', '0').strip())
            self.y_coordinate = int(self.config['OCR settings'].get('y_coordinate', '0').strip())
            self.width = int(self.config['OCR settings'].get('width', '500').strip())
            self.height = int(self.config['OCR settings'].get('height', '40').strip())
        except ValueError:
            # Handle cases where conversion to integer fails
            self.logger.warning('Invalid integer value found in settings_crop.ini')
        # check if video_filepath is valid path to a video file
        if not os.path.isfile(video_filepath) or not (
                video_filepath.endswith(".mp4") or video_filepath.endswith(".avi")):
            self.logger.warning('Invalid video file path')
            return

        # Set up video cap
        # DONE: MS
        # self.cap = cv2.VideoCapture(video_filepath)

        # Create a window and pass it to the mouse callback function
        cv2.namedWindow('image')
        cv2.setWindowProperty('image', cv2.WND_PROP_TOPMOST, 1)

        # display rectangle on image from the text_roi coordinates
        cv2.setMouseCallback('image', draw_rectangle)
        # DONE: MS
        while True:
            # ret, frame = self.cap.read()
            frame = self.video_file_object.read_video_frame(1, False)[0][3]
            if frame is not None:
                cv2.rectangle(frame, (self.x_coordinate, self.y_coordinate),
                              (self.x_coordinate + self.width, self.y_coordinate + self.height),
                              (0, 255, 0),
                              2)
                cv2.imshow('image', frame)
                k = cv2.waitKey(1) & 0xFF
                if k == 27:
                    cv2.destroyAllWindows()
                    break
            else:
                break
        cv2.destroyAllWindows()
        del self.video_file_object

        self.ocr_roi = (self.x_coordinate, self.y_coordinate, self.width, self.height)

        # Save settings to settings_crop.ini
        self.config['OCR settings']['x_coordinate'] = str(self.x_coordinate)
        self.config['OCR settings']['y_coordinate'] = str(self.y_coordinate)
        self.config['OCR settings']['width'] = str(self.width)
        self.config['OCR settings']['height'] = str(self.height)
        with open('settings_crop.ini', 'w', encoding='utf-8') as configfile:
            self.config.write(configfile)

    def generate_roi_entries(self):

        original_points = self.points_of_interest_entry[0][0].copy()

        self.reload_roi_entries()

        result = utils.ask_yes_no(
            "Would you like to perform single ROI detection? Only the first frame will be used for the detection. "
            "If you click 'No' all frames will be querried. Please use the advanced option only in the case of often changing flower location. "
            "\n\nThe number of frames querried will be reflected in the quota usage and you might be billed. ")
        if result:
            limit = 1
        else:
            limit = len(self.frames) + 1

        for i, (frame, filepath) in enumerate(zip(self.frames, self.video_filepaths)):
            if result and i >= limit:
                break
            height, width, _ = frame.shape
            unique_rois = vision_AI.get_unique_rois_from_frame(frame)
            overlaps = []
            for roi in unique_rois:
                rectangles = self.get_roi_extreme_offset_dimensions(roi, (width, height), 0, self.crop_size,
                                                                    self.offset_range)
                top_left_corner, bottom_right_corner = self.get_roi_offset_overlap(rectangles)

                # Calculate width and height
                width = bottom_right_corner[1] - top_left_corner[0]
                height = bottom_right_corner[1] - top_left_corner[1]

                overlaps.append((width, height))

            grouped_rois = vision_AI.get_grouped_rois_from_frame(self.frames, unique_rois, overlaps)
            self.points_of_interest_entry[i] = [grouped_rois, filepath]
            if not result:
                for each, _ in enumerate(self.points_of_interest_entry):
                    self.update_button_image(frame.copy(), (max(each, 0) // 6), (each - ((max(each, 0) // 6) * 6)), 0)

        if result:
            self.update_roi_entries(0, original_points)

    def open_roi_gui(self, i, j, button_images):

        def get_mouse_position(event, x, y, flags, mode, i, j):

            # Function that is triggered when user clicks the image in the interface. Coordinates of the click are recorded,
            # and the identification of the button clicked is accepted. Array of the selected POIs is appended accordingly.
            if event == cv2.EVENT_LBUTTONUP:
                index = j + ((i) * 6)
                if flags & cv2.EVENT_FLAG_SHIFTKEY:
                    closest_point = None
                    closest_distance = float('inf')
                    for point in self.points_of_interest_entry[index][0]:
                        distance = math.sqrt((x - point[0]) ** 2 + (y - point[1]) ** 2)
                        if distance < closest_distance:
                            closest_point = point
                            closest_distance = distance
                    if closest_distance < 30:
                        self.points_of_interest_entry[index][0].remove(closest_point)
                    self.logger.debug(f"Retrieved POIs: {self.points_of_interest_entry}")
                    cv2.destroyAllWindows()
                elif flags & cv2.EVENT_FLAG_CTRLKEY and flags & cv2.EVENT_FLAG_ALTKEY:
                    self.points_of_interest_entry[index][0] = []
                    if index in self.modified_frames:
                        self.modified_frames.remove(index)
                    cv2.destroyAllWindows()
                else:
                    if not mode == 0:
                        self.points_of_interest_entry[index][0].append((x, y))
                    else:
                        self.points_of_interest_entry.append((x, y))
                    self.logger.debug(f"Retrieved POIs: {self.points_of_interest_entry}")
                    cv2.destroyAllWindows()

        # Define logger
        self.logger.debug(f"Running function open_roi_gui({i}, {j})")

        # Define variables
        index = j + ((i) * 6)
        mode = 1
        frame_tmp = self.frames[index].copy()
        dict_of_extremes_to_draw = {
            'u_l': -1,
            'u_r': -1,
            'b_l': -1,
            'b_r': -1
        }

        # Ask the user if they want to select additional points of interest
        while True:
            original_points = self.points_of_interest_entry[index][0].copy()
            frame = frame_tmp.copy()

            # Draw a rectangle around the already selected points of interest
            frame = self.draw_roi_offset_boundaries(frame, self.points_of_interest_entry[index][0],
                                                    dict_of_extremes_to_draw, True,
                                                    True, True)

            # Display the image with the rectangles marking the already selected points of interest
            cv2.imshow("Frame", frame)
            cv2.setWindowProperty("Frame", cv2.WND_PROP_TOPMOST, 1)
            screen_width, screen_height = cv2.getWindowImageRect("Frame")[2:]
            window_width, window_height = int(1 * screen_width), int(1 * screen_height)
            cv2.resizeWindow("Frame", window_width, window_height)
            cv2.moveWindow("Frame", int((screen_width // 2) - (window_width // 2)), 0)

            # Prompt the user to click on the next point of interest
            cv2.setMouseCallback("Frame",
                                 lambda event, x, y, flags, mode: get_mouse_position(event, x, y, flags, mode, i, j),
                                 mode)

            # Wait for key press
            key = cv2.waitKey(0)

            # Define which keys will result in multiplying which variable by which value
            key_mappings = {
                27: ('esc', None),
                13: ('enter', None),
                ord('q'): ('u_l', -1),
                ord('w'): ('u_r', -1),
                ord('e'): ('b_l', -1),
                ord('r'): ('b_r', -1)
            }

            # If key is in the dictionary then check whether the value for this key is none - if so then it is esc or enter
            # and the window should be closed. If not then do the multiplication.
            if key in key_mappings:
                action, value = key_mappings[key]
                if value is None:
                    cv2.destroyAllWindows()
                    return

                if action in dict_of_extremes_to_draw:
                    dict_of_extremes_to_draw[action] *= value

            # Update other ROI entries to inherit the changes
            self.update_roi_entries(index, original_points)

    def update_roi_entries(self, index, original_points):

        # Define logger
        self.logger.debug(f"Running function update_roi_entries({index}, {original_points})")

        # For each video POIs list after the edited one check if there are no POIs set. If so then make it inherit the POIs from
        # the newly edited list. If there are some POIs for that entry and they are same as the original set of POIs before
        # the edit, then make it also inherit the newly edited ones. Otherwise that is when there is a different set of POIs
        # do not propagate the changes.
        for each in range(index + 1, len(self.points_of_interest_entry)):
            if len(self.points_of_interest_entry[each][0]) == 0:
                self.points_of_interest_entry[each][0] = self.points_of_interest_entry[index][0].copy()
            else:
                if self.points_of_interest_entry[each][0] == original_points:
                    self.points_of_interest_entry[each][0] = self.points_of_interest_entry[index][0].copy()
                else:
                    break
        first = 1
        for each in range(index, len(self.video_filepaths)):
            first = first - 1
            if first >= 0 and (
                    index == 0 or not self.points_of_interest_entry[max(index - 1, 0)][0] ==
                                      self.points_of_interest_entry[index][0]):
                self.modified_frames.append(each)
            self.update_button_image(self.frames[each].copy(), (max(each, 0) // 6), (each - ((max(each, 0) // 6) * 6)),
                                     first)

    def update_button_image(self, frame, i, j, first):

        # Define variables
        index = j + ((i) * 6)
        frame = frame.copy()

        # Draw roi area guides and offset overlap
        frame = self.draw_roi_offset_boundaries(frame, self.points_of_interest_entry[index][0], {
            'u_l': -1,
            'u_r': -1,
            'b_l': -1,
            'b_r': -1
        }, True, False, True)

        if (first >= 0 or index in self.modified_frames) and (
                (index == 0 and not len(self.points_of_interest_entry[0][0]) == 0) or not self.points_of_interest_entry[
                                                                                              max(index - 1, 0)][0] ==
                                                                                          self.points_of_interest_entry[
                                                                                              index][0]):
            # Define the ROI
            #frame = cv2.resize(frame, (276, 156), interpolation=cv2.INTER_AREA)
            height, width, channels = frame.shape
            x, y, w, h = 0, 0, height // 5, height // 5
            roi = frame[y:y + h, x:x + w]

            # Load the overlay image
            overlay = cv2.imread('resources/img/sp.png', cv2.IMREAD_UNCHANGED)

            # Resize the overlay image to fit the ROI
            overlay = cv2.resize(overlay, (w, h))

            # Merge the overlay image and the ROI
            alpha = overlay[:, :, 3] / 255.0
            alpha = cv2.merge([alpha, alpha, alpha])
            background = cv2.multiply(1.0 - alpha, roi, dtype=cv2.CV_8UC3)
            foreground = cv2.multiply(alpha, overlay[:, :, 0:3], dtype=cv2.CV_8UC3)
            result = cv2.add(background, foreground)

            # Copy the result back into the original image
            frame[y:y + h, x:x + w] = result

        # Convert colorspace
        pil_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(pil_img, mode='RGB')

        # Resize image to fit button
        #pil_img = pil_img.resize((276, 156))

        # Convert PIL image to tkinter image
        #img = ImageTk.PhotoImage(pil_img)
        self.button_images[index] = pil_img
        q_img = QImage(pil_img.tobytes(), pil_img.size[0], pil_img.size[1], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        self.buttons[index].setIcon(QtGui.QIcon(pixmap))

    def draw_roi_offset_boundaries(self, frame, list_of_ROIs, dict_of_extremes_to_draw, draw_roi: bool,
                                   draw_extremes: bool,
                                   draw_overlap: bool):

        # Define variables
        labels = ['BR', 'UL', 'UR', 'BL']
        offsets = [(1, 1), (-1, -1), (1, -1), (-1, 1)]
        conditions = [dict_of_extremes_to_draw["b_r"] > 0, dict_of_extremes_to_draw["u_l"] > 0,
                      dict_of_extremes_to_draw["u_r"] > 0, dict_of_extremes_to_draw["b_l"] > 0]
        pos_off = 0  # Offset if you want the lines to artificially draw next to each other in case you do not want an overlap of the lines.
        height, width, _ = frame.shape

        # For each point draw desired shapes
        for point in list_of_ROIs:

            # Define variables
            rectangles = []

            # Draw basic roi area
            if draw_roi:
                top_left_corner, bottom_right_corner = self.get_roi_dimensions_from_center(point, (width, height),
                                                                                           self.crop_size)
                top_right_corner = (bottom_right_corner[0], top_left_corner[1])
                bottom_left_corner = (top_left_corner[0], bottom_right_corner[1])
                center_of_roi = (point[0], point[1])
                cv2.rectangle(frame, (point[0] - 30, point[1] - 30), (point[0] + 30, point[1] + 30), (0, 255, 0), 2)
                cv2.rectangle(frame, top_left_corner, bottom_right_corner, (0, 0, 255), 3)
                cv2.line(frame, center_of_roi, top_left_corner, (0, 255, 255), thickness=2, lineType=cv2.LINE_AA)
                cv2.line(frame, center_of_roi, bottom_left_corner, (0, 255, 255), thickness=2, lineType=cv2.LINE_AA)
                cv2.line(frame, center_of_roi, top_right_corner, (0, 255, 255), thickness=2, lineType=cv2.LINE_AA)
                cv2.line(frame, center_of_roi, bottom_right_corner, (0, 255, 255), thickness=2, lineType=cv2.LINE_AA)

            # Draw the offset extremes rectangles
            if draw_extremes or draw_overlap:
                rectangles = self.get_roi_extreme_offset_dimensions(point, (width, height), 0, self.crop_size,
                                                                    self.offset_range)
                for rectangle, condition in zip(rectangles, conditions):
                    if condition:
                        cv2.rectangle(frame, rectangle[0], rectangle[1], (255, 229, 0), 2)

                # Draw overlap area
                if draw_overlap:
                    top_left_corner, bottom_right_corner = self.get_roi_offset_overlap(rectangles)

                    # Draw the rectangle of the overlap
                    cv2.rectangle(frame, top_left_corner, bottom_right_corner, (0, 255, 0), 2)
        return frame

    def get_roi_dimensions_from_center(self, center_point, frame_dimensions: tuple, square_crop_size: int = 640):

        # Define variables
        width, height = frame_dimensions

        # Calculater top-left and bottom-right corner coords
        x1, y1 = max(0, center_point[0] - (square_crop_size // 2)), max(0, center_point[1] - (square_crop_size // 2))
        x2, y2 = min(width, center_point[0] + (square_crop_size // 2)), min(height,
                                                                            center_point[1] + (square_crop_size // 2))

        return (x1, y1), (x2, y2)

    def get_roi_extreme_offset_dimensions(self, center_point, frame_dimensions: tuple, pos_off: int = 0,
                                          square_crop_size: int = 640, offset_range: int = 100):

        # Define variables
        width, height = frame_dimensions
        labels = ['BR', 'UL', 'UR', 'BL']  # This will be the mapping of the output
        offsets = [(1, 1), (-1, -1), (1, -1), (-1, 1)]
        rectangles = []

        for i, (label, (offset_x, offset_y)) in enumerate(zip(labels, offsets)):
            o_x1 = max(pos_off, min(((center_point[0] - square_crop_size // 2) + offset_x * offset_range) - pos_off,
                                    width - square_crop_size - pos_off))
            o_y1 = max(pos_off, min(((center_point[1] - square_crop_size // 2) + offset_y * offset_range) - pos_off,
                                    height - square_crop_size - pos_off))
            o_x2 = max(square_crop_size - pos_off,
                       min(((center_point[0] + square_crop_size // 2) + offset_x * offset_range) - pos_off,
                           width - pos_off))
            o_y2 = max(square_crop_size - pos_off,
                       min(((center_point[1] + square_crop_size // 2) + offset_y * offset_range) - pos_off,
                           height - pos_off))
            rectangles.append([(o_x1, o_y1), (o_x2, o_y2)])

        return rectangles

    def get_roi_offset_overlap(self, rectangles):

        x_min = max(rect[0][0] for rect in rectangles)
        y_min = max(rect[0][1] for rect in rectangles)
        x_max = min(rect[1][0] for rect in rectangles)
        y_max = min(rect[1][1] for rect in rectangles)

        return (x_min, y_min), (x_max, y_max)

    def update_crop_mode(self, var):

        # Define logger
        self.logger.debug(f"Running function update_crop_mode({var})")

        # Set variable to the new value
        self.crop_mode = var

    def save_progress(self):
        # Define logger
        self.logger.debug(f"Running function save_progress()")

        # Confirm the users decision
        result = utils.ask_yes_no("Do you want to save the settings? This will overwrite any previous saves.")
        if result:
            if utils.check_path(self.video_folder_path, 0):
                # Create an in-memory file object
                filepath = os.path.join(self.video_folder_path, 'crop_information.json')
                data_matched = {i: self.points_of_interest_entry[i] for i in
                                range(len(self.points_of_interest_entry))}

                data_combined = {
                    "auto_processing": self.auto_processing,
                    "video_data": data_matched
                }

                with open(filepath, "w") as json_file:
                    json.dump(data_combined, json_file)
            else:
                # display message box with error message
                messagebox.showerror("Error", "Invalid video folder path")

    def load_progress(self):

        # Define logger
        self.logger.debug(f"Running function load_progress()")

        # Confirm with the user if they want to load the settings
        result = utils.ask_yes_no("Do you want to load settings? This will overwrite any unsaved progress.")

        if result:
            if utils.check_path(self.video_folder_path, 0):

                # Create an in-memory file object
                filepath = os.path.join(self.video_folder_path, 'crop_information.json')
                if os.path.isfile(filepath):
                    try:
                        if self.main_window.winfo_exists():
                            self.main_window.destroy()
                    except:
                        self.logger.debug("Error: Unexpected, window destroyed before reference.")
                    self.points_of_interest_entry = []
                    with open(filepath, "r") as json_file:
                        data_combined = json.load(json_file)

                        # Restore data from dictionary
                        self.auto_processing = (data_combined["auto_processing"])
                        data_matched = data_combined["video_data"]
                        self.points_of_interest_entry = [item for item in data_matched.values()]
                        video_filepaths_new = [item[1] for item in data_matched.values()]

                    # Compare the loaded and the currently found video filepaths.
                    set1 = set({os.path.basename(filepath) for filepath in video_filepaths_new})
                    set2 = set({os.path.basename(filepath) for filepath in self.video_filepaths})
                    if not set1 == set2:
                        messagebox.showinfo("Discrepancy detected",
                                            "The contents of the video folder have changed since the save has been made. Cannot load the progress. Please start over.")
                        self.reload_roi_entries()
                    else:
                        self.video_filepaths = []
                        self.video_filepaths = video_filepaths_new.copy()
                    self.reload(True, False)
                else:
                    messagebox.showinfo("No save detected",
                                        "There are no save files in the current directory.")
            else:
                messagebox.showerror("Error", "Invalid video folder path")


if __name__ == '__main__':
    # iccs = ICCS()
    def show_main_window():
        global iccs  # Declare global to access the instance elsewhere
        iccs = ICCS()


    import sys
    import traceback
    from PyQt5.QtWidgets import QMessageBox


    def except_hook(exc_type, exc_value, exc_tb):
        tb_string = ''.join(traceback.format_exception(exc_type, exc_value, exc_tb))
        QMessageBox.critical(
            None,
            "An exception was raised",
            f"An unhandled Exception occurred:\n{tb_string}",
            QMessageBox.Ok
        )


    sys.excepthook = except_hook


    app = QApplication([])

    # Create a hidden main window
    hidden_main_window = QMainWindow()
    hidden_main_window.setWindowFlags(Qt.Tool)

    # Splash screen
    splash_pix = QPixmap("resources/img/iccs.png")
    splash = QSplashScreen(splash_pix, Qt.WindowStaysOnTopHint)
    splash.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
    splash.setEnabled(False)

    # Optionally set the window icon for the splash screen
    splash.setWindowIcon(QIcon("resources/img/iccs.png"))

    splash.show()

    # Close splash screen after 2 seconds
    QTimer.singleShot(2000, splash.close)
    QTimer.singleShot(2000, show_main_window)

    app.exec_()


