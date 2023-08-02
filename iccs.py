# This file contains the ICCS app class that inherits from ICVT AppAncestor class
import utils
import anno_data
import icvt
import pandas as pd
import os
import subprocess
import threading
import cv2
import configparser
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import pickle
import sys
import random
import math
import time
from ultralytics import YOLO
import shutil
import asyncio
import numpy as np

class ICCS(icvt.AppAncestor):
    def __init__(self):

        # Define logger
        self.logger = self.log_define()

        # First log
        self.logger.info("Initializing ICCS - Insect Communities Crop Suite application class...")

        # Start the loading bar in a separate thread
        time.sleep(0.2)
        loading_thread = threading.Thread(target=self.loading_bar)
        loading_thread.start()
        self.loading_progress = 1

        # Init basic instance variables and get config
        self.app_title = "Insect Communities Crop Suite"

        # Create config and read file
        self.config = self.config_create()
        self.config_read()
        self.loading_progress = 20

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
        self.loading_progress = 25

        # Initiation functions - get directories and files
        self.scan_default_folders()
        self.loading_progress = 35

        # If video folder path not supplied ask user to specify it
        while not self.check_path():
            self.get_video_folder(1)
        self.loading_progress = 40

        # Ask the user to specify the Excel path
        self.get_excel_path(1, self.crop_mode)
        self.loading_progress = 45

        # Load the videos
        self.load_videos()
        self.loading_progress = 55

        # Construct ROI data
        self.reload_points_of_interest()
        self.loading_progress = 65

        # Load video frames for buttons
        self.load_video_frames()
        self.loading_progress = 85

        # Create output folders
        utils.create_dir(self.output_folder)
        utils.create_dir(os.path.join(".", self.output_folder, "whole frames"))

        # Open window
        self.loading_progress = 100
        time.sleep(0.5)
        self.stop_loading = True
        loading_thread.join()
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

        # Save changes to the config file
        with open('settings_crop.ini', 'w', encoding='utf-8') as configfile:
            config.write(configfile)

    def reload_points_of_interest(self):
        logger = self.logger
        logger.debug('Running function reload_points_of_interest()')

        # Clear the array of POIs and reconstruct it with empty lists.
        self.points_of_interest_entry.clear()
        self.points_of_interest_entry = [[[], filepath] for filepath in self.video_filepaths]

    def load_video_frames(self):
        # Define logger
        logger = self.logger
        logger.debug(f"Running function load_video_frames()")

        # Loop through each file in folder
        self.frames = []
        if utils.check_path(self.video_folder_path, 0):
            for filename in os.listdir(self.video_folder_path):
                if filename.endswith(".mp4"):  # Modify file extension as needed
                    video_path = os.path.join(self.video_folder_path, filename)
                    try:
                        # Use OpenCV or other library to extract first frame of video
                        # and add it to the frames list
                        cap = cv2.VideoCapture(video_path)
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 25)
                        ret, frame = cap.read()
                        if ret:
                            self.frames.append(frame)
                        else:
                            # If the read operation fails, add a default image
                            default_image = cv2.imread('resources/img/nf.png')
                            self.frames.append(default_image)
                    except (cv2.error, OSError) as e:
                        logger.error(f"Error: Failed to process video '{filename}': {e}")
                        default_image = cv2.imread('resources/img/nf.png')
                        self.frames.append(default_image)
        else:
            logger.error("Error: Invalid video folder path")
            messagebox.showerror("Error", "Invalid video folder path")

######################################### SECTION DEALS WITH THE BACKEND ###############################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################

    ######################################### CROP FUNCTIONALITY ######################################################

    async def capture_crop(self, frame, point):

       # Define logger
        self.logger.debug(f"Running function capture_crop({point})")

        # Prepare local variables
        x, y = point

        # Add a random offset to the coordinates, but ensure they remain within the image bounds
        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Check if any of the dimensions is smaller than crop_size and if so upscale the image to prevent crops smaller than desired crop_size
        if frame_height < self.crop_size or frame_width < self.crop_size:
            # Calculate the scaling factor to upscale the image
            scaling_factor = self.crop_size / min(frame_height, frame_width)

            # Calculate the new dimensions for the upscaled frame
            new_width = int(round(frame_width * scaling_factor))
            new_height = int(round(frame_height * scaling_factor))

            # Upscale the frame using cv2.resize with Lanczos upscaling algorithm
            frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)

        # Get the new frame size
        frame_height, frame_width = frame.shape[:2]

        # Calculate the coordinates for the area that will be cropped
        x_offset = random.randint(-self.offset_range, self.offset_range)
        y_offset = random.randint(-self.offset_range, self.offset_range)
        x1 = max(0, min(((x - self.crop_size // 2) + x_offset), frame_width - self.crop_size))
        y1 = max(0, min(((y - self.crop_size // 2) + y_offset), frame_height - self.crop_size))
        x2 = max(self.crop_size, min(((x + self.crop_size // 2) + x_offset), frame_width))
        y2 = max(self.crop_size, min(((y + self.crop_size // 2) + y_offset), frame_height))

        # Crop the image
        crop = frame[y1:y2, x1:x2]

        # Convert to correct color space
        crop_img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        if crop_img.shape[2] == 3:
            crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)

        # Return the cropped image and the coordinates for future reference
        return crop_img, x1, y1, x2, y2

    async def generate_frames(self, frame, success, tag, index, frame_number_start):

        # Define logger
        self.logger.debug(f"Running function generate_frames({index})")

        # Prepare name elements
        filename_parts = tag[:-4].split("_")
        recording_identifier = "_".join(filename_parts[:-3])
        timestamp = "_".join(filename_parts[-3:])

        # Define local variables
        crop_counter = 1
        frame_skip_loc = self.frame_skip

        # Calculate the frame skip variable based on the limited number of frames per visit
        if self.frames_per_visit > 0:
            frame_skip_loc = int((self.visit_duration * self.fps) // self.frames_per_visit)
            if frame_skip_loc < 1:
                frame_skip_loc = 1

        # Loop through the video and crop y images every n-th frame
        frame_count = 0
        image_paths = []
        image_details_dict = {}

        while success:
            # Crop images every n-th frame
            if int(frame_count % frame_skip_loc) == 0:
                for i, point in enumerate(self.points_of_interest_entry[index][0]):
                    if self.cropped_frames == 1:
                        crop_img, x1, y1, x2, y2 = await self.capture_crop(frame, point)
                        frame_number = frame_number_start + frame_count
                        roi_number = i + 1
                        visit_number = crop_counter
                        image_name = f"{self.prefix}{recording_identifier}_{timestamp}_{roi_number}_{frame_number}_{visit_number}_{x1},{y1}_{x2},{y2}.jpg" #Now the output images will be ordered by the ROI therefore one will be able to delete whole segments of pictures.
                        image_path = os.path.join(self.output_folder, image_name)
                        #image_path = f"./{self.output_folder}/{self.prefix}{recording_identifier}_{timestamp}_{frame_number_start + frame_count}_{crop_counter}_{i + 1}_{x1},{y1}_{x2},{y2}.jpg"
                        cv2.imwrite(image_path, crop_img)
                        image_paths.append(image_path)
                        image_details_dict[image_name] = (image_path, frame_number, roi_number, visit_number)
                if self.whole_frame == 1:
                    frame_number = frame_number_start + frame_count
                    visit_number = crop_counter
                    image_name = f"{self.prefix}{recording_identifier}_{timestamp}_{frame_number}_{visit_number}_whole.jpg"
                    image_path = os.path.join(self.output_folder, "whole frames", image_name)
                    #image_path = f"./{self.output_folder}/whole frames/{self.prefix}{recording_identifier}_{timestamp}_{frame_number_start + frame_count}_{crop_counter}_whole.jpg"
                    cv2.imwrite(image_path, frame)
                crop_counter += 1

            # If the random frame skip interval is activated add a random number to the counter or add the set frame skip interval
            if self.randomize == 1:
                if (frame_skip_loc - frame_count == 1):
                    frame_count += 1
                else:
                    frame_count += random.randint(1, max((frame_skip_loc - frame_count), 2))
            else:
                frame_count += frame_skip_loc

            # Read the next frame
            frame_to_read = frame_number_start + frame_count
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_to_read)
            success, frame = self.cap.read()

            # If the frame count is equal or larger than the amount of frames that comprises the duration of the visit end the loop
            if not (frame_count <= (self.visit_duration * self.fps)):
                # Release the video capture object and close all windows
                self.cap.release()
                cv2.destroyAllWindows()
                break

        # Return the resulting list of image paths for future reference
        return image_paths

    async def yolo_preprocessing(self, img_paths, valid_annotations_array, index):

        # Define logger
        self.logger.debug(f"Running function yolo_preprocessing({img_paths})")

        # Define and run the model
        model = YOLO('resources/yolo/best.pt')
        results = model(img_paths, save=False, imgsz=self.crop_size, conf=self.yolo_conf, save_txt=False, max_det=1, stream=True)
        for i, result in enumerate(results):
            boxes = result.boxes.data
            original_path = os.path.join(img_paths[i])
            utils.create_dir(f"{self.output_folder}/empty")
            utils.create_dir(f"{self.output_folder}/visitor")
            utils.create_dir(f"{self.output_folder}/visitor/labels")
            empty_path = os.path.join(f"{self.output_folder}/empty", os.path.basename(img_paths[i]))
            visitor_path = os.path.join(f"{self.output_folder}/visitor", os.path.basename(img_paths[i]))
            label_path = os.path.join(f"{self.output_folder}/visitor/labels", os.path.basename(img_paths[i])[:-4])
            if len(result.boxes.data) > 0:
                shutil.move(original_path, visitor_path)
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
                shutil.move(original_path, empty_path)

    def filter_array_by_visitors(self, valid_annotations_array):

        # Define logger
        self.logger.debug(f"Running function filter_array_by_visitors()")

        # Create filter window
        filter_window = tk.Tk()
        filter_window.title("Filter Visitors")
        filter_window.wm_attributes("-topmost", 1)

        def apply_filter():
            results = []
            #print(checkbox_vars)
            for i, var in enumerate(checkbox_vars):
                checkbox_value = var.get()
                #print(checkbox_value)
                checkbox_text = checkboxes[i].cget("text")
                #print(checkbox_text)
                dropdown_value = selected_items[i].get()
                #print(dropdown_value)
                if not dropdown_value[0].isdigit():
                    dropdown_value = "6. empty"
                results.append([int(checkbox_value), checkbox_text, int(dropdown_value[0])])
            results = [row for row in results if row[0] != 0]
            if self.yolo_processing == 1:
                for row in results:
                    if row[2] == 6:
                        filter_window.quit()
                        filter_window.destroy()
                        messagebox.showinfo("Warning",
                                            f"One of the selected groups of visitors was not assigned a correct category for labelling. Please try again.")
                        self.filtered_array = []
                        return
            #print(results)
            selected_values = [value.get() for value in checkbox_vars]
            self.filtered_array = []
            for row in valid_annotations_array:
                if row[5] in [result[1] for result in results]:
                    matching_result = next(result for result in results if result[1] == row[5])
                    filtered_row = row + [matching_result[2]]
                    self.filtered_array.append(filtered_row)
            #print(self.filtered_array)  # Do something with the filtered array
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
        allowed_items = ["0. Hymenoptera", "1. Diptera", "2. Lepidoptera", "3. Coleoptera", "4. Other"]
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
                selected_item.set("------------")
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
            #print(self.filtered_array)  # Do something with the filtered array
            filter_window.quit()
            filter_window.destroy()

    def crop_engine(self):

        # Define logger
        self.logger.debug("Running function crop_engine()")

        # Define variables
        root = self.main_window

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
            root.withdraw()

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
                video_data = self.get_video_data(sorted_video_filepaths)

                # Log the information that are fed into the rest of the engine
                self.logger.debug(f"Start cropping according to the following annotations: {annotation_data_array}")
                self.logger.debug(f"Start cropping with the following video data: {video_data}")

                # Construct the valid annotation data array which contains visit data coupled with the path to the
                # video containing the visit
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

                # Process the visits and generate cropped images
                for index in range(len(valid_annotations_array)):
                    self.logger.info(f"Processing visit {index+1}")

                    # Define index which will be shared to sync the stage of the process
                    self.visit_index = index

                    # Define the variables
                    visit_duration, visit_timestamp, video_filepath, video_start_time, *_ = valid_annotations_array[index]

                    # Turn timestamp into datetime and calculate how many seconds from the start_time of the video recording does the visit take place
                    visit_time = pd.to_datetime(visit_timestamp, format='%Y%m%d_%H_%M_%S')
                    visit_time_from_start = (visit_time - video_start_time).total_seconds()

                    # Define shared cap to open the video filepath
                    self.cap = cv2.VideoCapture(video_filepath)

                    # Calculat the total number of video frames, fps ... and figure out where t oset the cap and read a frame
                    total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    self.fps = self.cap.get(cv2.CAP_PROP_FPS)

                    # The actual duration is taken as limited by the end of the video, therefore cropping wont carry on for longer than one entire video file.
                    self.visit_duration = (min(((visit_time_from_start * self.fps) + (int(visit_duration) * self.fps)),
                                          total_frames) - (visit_time_from_start * self.fps)) // self.fps

                    # First frame to capture - start of the visit
                    frame_number_start = int(visit_time_from_start * self.fps)

                    # Read the frame
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number_start)
                    success, frame = self.cap.read()

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

                    # Generate frames and get back their paths
                    img_paths = asyncio.run(
                        self.generate_frames(frame, success, os.path.basename(video_filepath),
                                        roi_index, frame_number_start))

                    #print(img_paths)

                    # If relevant preprocess the images using yolo
                    if self.yolo_processing == 1:
                        asyncio.run(self.yolo_preprocessing(img_paths, valid_annotations_array, index))

            # This is for when the crop_mode is 3 and no annotation file is supplied
            else:

                # Check validy of paths
                video_ok = utils.check_path(self.video_folder_path, 0)
                if not video_ok:
                    messagebox.showinfo("Warning",
                                        f"Unspecified path to a video folder.")
                    self.reload(True, False)
                    return

                # The whole frame setings is artificially altered to allow for whole frame generation - messy
                orig_wf = self.whole_frame
                self.whole_frame = 1

                # Starting from the second frame of every video frames are generated.
                for i, filepath in enumerate(self.video_filepaths):
                    self.cap = cv2.VideoCapture(self.video_filepaths[i])
                    self.fps = self.cap.get(cv2.CAP_PROP_FPS)
                    self.visit_duration = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) // self.fps
                    frame_number_start = 2
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number_start)
                    success, frame = self.cap.read()
                    img_paths = self.generate_frames(frame, success, os.path.basename(self.video_filepaths[i]), i, frame_number_start)
                self.whole_frame = orig_wf

        # When done, reload the application
        self.reload(True, False)

    def sort_engine(self):

        # Define logger
        self.logger.debug("Running function sort_engine()")

        # Ask user if they want to run sorting script
        run_sorting = utils.ask_yes_no("Do you want to Running function the sorting script on the generated images?")
        if run_sorting:
            self.logger.info("Initializing the sorting engine...")
            sort_script_path = "sort.py"
            if os.path.exists(sort_script_path):
                # subprocess.run(['python', f'{sort_script_path}'])
                subprocess.call([sys.executable, f'{sort_script_path}', "--subprocess"])
            else:
                self.logger.warning("Sorting script not found.")

######################################### SECTION DEALS WITH THE GUI ###################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################

    def open_main_window(self):

        # Define logger
        self.logger.debug(f"Running function open_main_window()")

        # Create tkinter window
        root = tk.Tk()
        self.main_window = root
        root.focus()
        root.title(f"{self.app_title} - Folder: {os.path.basename(os.path.normpath(self.video_folder_path))} - Table: {os.path.basename(os.path.normpath(self.annotation_file_path))}")

        # Arbitrary variable for lambdas
        j = 0

        # Create frame for the rest
        outer_frame = tk.Frame(root)
        outer_frame.pack(side=tk.TOP, expand=True, fill=tk.BOTH)

        top_toolbar = self.build_top_toolbar(outer_frame)
        center_frame = self.build_center_frame(outer_frame)
        top_toolbar.lift(center_frame)
        bottom_left_panel = self.build_bottom_left_panel(root)
        bottom_right_panel = self.build_bottom_right_panel(root)
        bottom_toolbar = self.build_bottom_toolbar(root)

        # Update button images
        if self.loaded == True:
            for each in range(len(self.video_filepaths)):
                first = -1
                self.update_button_image(self.frames[each].copy(), (max(each, 0) // 6), (each - ((max(each, 0) // 6) * 6)), first)
            # if auto is defined, set the auto processing to that value otherwise set it to 0
            if self.auto is None:
                self.auto_processing.set(0)
            else:
                self.auto_processing.set(self.auto)

        # Get the screen width and height
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()

        # Set the window size to fit the entire screen
        root.geometry(f"{screen_width}x{screen_height - 70}+-10+0")
        root.mainloop()

    def build_top_toolbar(self, parent):

        def can_switch_folder(where):
            btn_state = "disabled"
            if where == "right":
                if self.dir_hierarchy and (self.scanned_folders.index(os.path.basename(os.path.normpath(self.video_folder_path))) + 1) != len(self.scanned_folders):
                    btn_state = "normal"
                else:
                    btn_state = "disabled"
            elif where == "left":
                if self.dir_hierarchy and self.scanned_folders.index(os.path.basename(os.path.normpath(self.video_folder_path))) != 0:
                    btn_state = "normal"
                else:
                    btn_state = "disabled"
            return btn_state

        # Define variables
        outer_frame = parent
        j = 0

        # Create top toolbar
        toolbar = tk.Frame(outer_frame)
        toolbar.pack(side=tk.TOP, fill=tk.BOTH)

        # LEFT button
        left_button = tk.Button(toolbar, image=self.load_icon("resources/img/la.png"), compound=tk.LEFT, text="Previous folder", padx=10, pady=5,
                                height=48, width=200, state=can_switch_folder("left"), command=lambda j=j: self.switch_folder("left"))
        left_button.grid(row=0, column=0, padx=0, pady=5, sticky="ew")

        # MENU button
        menu_button = tk.Button(toolbar, image=self.load_icon("resources/img/mn.png"), compound=tk.LEFT, text="Menu", padx=10, pady=5, height=48,
                                command=lambda j=j: self.open_menu())
        menu_button.grid(row=0, column=1, padx=0, pady=5, sticky="ew")

        # frame for radio buttons
        radio_frame = tk.Frame(toolbar)
        radio_frame.grid(row=0, column=2, padx=0, pady=5, sticky="ew")

        # create a tkinter variable to hold the selected value
        self.selected_option = tk.StringVar(value=self.crop_mode)

        # CROP_MODE buttons
        # create the radio buttons and group them together
        rb1 = tk.Radiobutton(radio_frame, text="", image=self.load_icon("resources/img/1.png"), variable=self.selected_option, value=1, indicatoron=False,
                             height=56, width=116, font=("Arial", 17), command=lambda j_=j: self.update_crop_mode(1))
        rb2 = tk.Radiobutton(radio_frame, text="", image=self.load_icon("resources/img/2.png"), variable=self.selected_option, value=2, indicatoron=False,
                             height=56, width=116, font=("Arial", 17), command=lambda j=j: self.update_crop_mode(2))
        rb3 = tk.Radiobutton(radio_frame, text="", image=self.load_icon("resources/img/3.png"), variable=self.selected_option, value=3,
                             indicatoron=False,
                             height=56, width=116, font=("Arial", 17), command=lambda j=j: self.update_crop_mode(3))

        # arrange the radio buttons in a horizontal layout using the grid geometry manager
        rb1.grid(row=0, column=0, sticky="ew")
        rb2.grid(row=0, column=1, sticky="ew")
        rb3.grid(row=0, column=2, sticky="ew")
        radio_frame.grid_columnconfigure(0, weight=1, minsize=50)
        radio_frame.grid_columnconfigure(1, weight=1, minsize=50)
        radio_frame.grid_columnconfigure(2, weight=1, minsize=50)

        # AUTO checkbox
        self.on_image = tk.PhotoImage(width=116, height=57)
        self.off_image = tk.PhotoImage(width=116, height=57)
        self.on_image.put(("green",), to=(0, 0, 56, 56))
        self.off_image.put(("red",), to=(57, 0, 115, 56))
        self.auto_processing = tk.IntVar(value=0)
        self.auto_processing.set(0)
        cb1 = tk.Checkbutton(toolbar, image=self.off_image, selectimage=self.on_image, indicatoron=False, onvalue=1, offvalue=0,
                             variable=self.auto_processing)
        cb1.grid(row=0, column=3, padx=0, pady=5, sticky="ew")

        # AUTO button
        auto_button = tk.Button(toolbar, image=self.load_icon("resources/img/au.png"), compound=tk.LEFT, text="Automatic evaluation", padx=10,
                                pady=5, height=48,
                                command=lambda j=j: self.auto_processing.set(1 - self.auto_processing.get()))
        auto_button.grid(row=0, column=4, padx=0, pady=5, sticky="ew")

        # VIDEO FOLDER button
        fl_button = tk.Button(toolbar, image=self.load_icon("resources/img/fl.png"), compound=tk.LEFT, text="Select video folder", padx=10, pady=5,
                              height=48, command=lambda j=j: self.change_video_folder())
        fl_button.grid(row=0, column=5, padx=0, pady=5, sticky="ew")

        # EXCEL PATH button
        et_button = tk.Button(toolbar, image=self.load_icon("resources/img/et.png"), compound=tk.LEFT, text="Select Excel table", padx=10, pady=5,
                              height=48, command=lambda j=j: self.change_excel_path())
        et_button.grid(row=0, column=6, padx=0, pady=5, sticky="ew")

        # OCR button
        ocr_button = tk.Button(toolbar, image=self.load_icon("resources/img/ocr.png"), compound=tk.LEFT, text="OCR", padx=10, pady=5, height=48,
                               width=100, command=lambda j=j: self.open_ocr_roi_gui(self.video_filepaths[0]))
        ocr_button.grid(row=0, column=7, padx=0, pady=5, sticky="ew")

        # RIGHT button
        right_button = tk.Button(toolbar, image=self.load_icon("resources/img/ra.png"), compound=tk.RIGHT, text="Next folder", padx=10, pady=5,
                                 height=48, width=200, state=can_switch_folder("right"), command=lambda j=j: self.switch_folder("right"))
        right_button.grid(row=0, column=8, padx=0, pady=5, sticky="ew")

        # configure columns of toolbox
        toolbar.grid_columnconfigure(0, weight=2, minsize=50)
        toolbar.grid_columnconfigure(1, weight=3, minsize=50)
        toolbar.grid_columnconfigure(2, weight=4, minsize=150)
        toolbar.grid_columnconfigure(3, weight=4, minsize=50)
        toolbar.grid_columnconfigure(4, weight=1, minsize=50)
        toolbar.grid_columnconfigure(5, weight=4, minsize=50)
        toolbar.grid_columnconfigure(6, weight=4, minsize=50)
        toolbar.grid_columnconfigure(7, weight=4, minsize=50)
        toolbar.grid_columnconfigure(8, weight=2, minsize=50)

        return toolbar

    def build_center_frame(self, parent):

        # Define variables
        outer_frame = parent

        # Create a canvas to hold the tile buttons
        canvas = tk.Canvas(outer_frame)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Create a scrollbar for the canvas
        scrollbar = tk.Scrollbar(outer_frame, orient=tk.VERTICAL, command=canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Configure the canvas to use the scrollbar
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        # Create target frame for the rest
        target_frame = tk.Frame(outer_frame)
        target_frame.pack(side=tk.TOP)

        self.button_images = []
        self.buttons = []
        rows = math.ceil(len(self.video_filepaths) / 6)
        # per_row = math.ceil(len(self.video_filepaths) // rows)
        label_frame = tk.Frame(target_frame, width=50)
        label_frame.pack(side=tk.LEFT)
        for i in range(rows):
            hour_1st = ((i) * 6)
            if (((i) * 6) + 6) + 1 <= len(self.video_filepaths):
                hour_2nd = max(((i) * 6) + 6, len(self.video_filepaths) - 1)
            else:
                hour_2nd = (len(self.video_filepaths) % 6) - 1
            text_label = tk.Label(label_frame,
                                  text=f"{os.path.basename(self.video_filepaths[hour_1st])[-9:-7]}-{os.path.basename(self.video_filepaths[hour_2nd])[-9:-7]}",
                                  font=("Arial", 15), background=outer_frame.cget('bg'))
            text_label.pack(side=tk.TOP, padx=30, pady=67)
        for i in range(rows):
            # Loop through first 24 frames and create buttons with images
            button_frame = tk.Frame(target_frame)
            button_frame.pack(side=tk.TOP)
            row = []
            for j in range(6):
                if (j + ((i) * 6)) < len(self.video_filepaths):
                    # Convert frame to PIL image
                    pil_img = cv2.cvtColor(self.frames[j + ((i) * 6)], cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(pil_img, mode='RGB')
                    # Resize image to fit button
                    pil_img = pil_img.resize((276, 156))
                    # Convert PIL image to tkinter image
                    tk_img = ImageTk.PhotoImage(pil_img)
                    self.button_images.append(tk_img)
                    # Create button with image and add to button frame
                    button = tk.Button(button_frame, image=self.button_images[j + ((i) * 6)],
                                       command=lambda i=i, j=j: self.open_roi_gui(i, j, self.button_images))
                    button.grid(row=i, column=j, sticky="w")
                    row.append(button)
                else:
                    # Create a dummy frame to fill in the grid
                    new_img = Image.new("RGBA", (276, 156), (0, 0, 0, 0))
                    new_img = ImageTk.PhotoImage(new_img)
                    dummy_frame = tk.Button(button_frame, image=new_img, foreground="white", state="disabled")
                    dummy_frame.grid(row=i, column=j, sticky='w')
                    row.append(dummy_frame)
            self.buttons.append(row)

        # Update the canvas to show the buttons
        canvas.create_window((0, 0), window=target_frame, anchor=tk.NW)
        target_frame.update_idletasks()

        # Configure the canvas to show the entire frame
        canvas.configure(scrollregion=canvas.bbox("all"))

        # Bind mouse wheel event to canvas
        canvas.bind_all("<MouseWheel>", lambda event: canvas.yview_scroll(-1 * (event.delta // 120), "units"))

        return target_frame

    def build_bottom_left_panel(self, parent):

        # Define variables
        root = parent
        j = 0

        # Create panel frame
        bottom_left_panel = tk.Frame(root, pady=0)
        bottom_left_panel.pack(side=tk.LEFT)

        # Create buttons
        fl_button = tk.Button(bottom_left_panel, image=self.load_icon("resources/img/fl.png"), compound=tk.LEFT, text="", padx=10, pady=5, width=60,
                              height=58,
                              command=lambda j=j: os.startfile(self.video_folder_path))
        fl_button.pack(side=tk.LEFT)

        return bottom_left_panel

    def build_bottom_right_panel(self, parent):

        # Define variables
        root = parent
        j = 0

        # Create panel frame
        bottom_right_panel = tk.Frame(root, pady=0)
        bottom_right_panel.pack(side=tk.RIGHT)

        # Create buttons
        et_button = tk.Button(bottom_right_panel, image=self.load_icon("resources/img/et.png"), compound=tk.LEFT, text="", padx=10, pady=5, width=60,
                              height=58,
                              command=lambda j=j: os.startfile(self.annotation_file_path))
        et_button.pack(side=tk.RIGHT)

        ef_button = tk.Button(bottom_right_panel, image=self.load_icon("resources/img/ef.png"), compound=tk.LEFT, text="", padx=0, pady=5, width=60,
                              height=58,
                              command=lambda j=j: os.startfile(os.path.dirname(self.annotation_file_path)))
        ef_button.pack(side=tk.RIGHT)

        return bottom_right_panel

    def build_bottom_toolbar(self, parent):

        # Define variables
        root = parent
        j = 0

        # Create panel frame
        bottom_toolbar = tk.Frame(root, pady=5)
        bottom_toolbar.pack(side=tk.TOP, fill=tk.BOTH)

        # Create buttons
        save_button = tk.Button(bottom_toolbar, text="Save", image=self.load_icon("resources/img/sv_1.png"), compound=tk.LEFT, padx=10, pady=5,
                                width=300,
                                height=48, command=lambda j=j: self.save_progress())

        crop_button = tk.Button(bottom_toolbar, text="Crop", image=self.load_icon("resources/img/cr_1.png"), compound=tk.LEFT, padx=10, pady=5,
                                width=300,
                                height=48, command=lambda j=j: self.crop_engine())

        sort_button = tk.Button(bottom_toolbar, text="Sort", image=self.load_icon("resources/img/so.png"), compound=tk.LEFT, padx=10, pady=5,
                                width=300,
                                height=48, command=lambda j=j: self.sort_engine())

        load_button = tk.Button(bottom_toolbar, text="Load", image=self.load_icon("resources/img/lo.png"), compound=tk.LEFT, padx=10, pady=5,
                                width=300,
                                height=48, command=lambda j=j: self.load_progress())

        # Specify the column for each button
        save_button.grid(row=0, column=1, sticky="ew")
        crop_button.grid(row=0, column=2, sticky="ew")
        sort_button.grid(row=0, column=3, sticky="ew")
        load_button.grid(row=0, column=4, sticky="ew")

        # Add padding between the buttons
        bottom_toolbar.grid_columnconfigure(0, weight=1)
        bottom_toolbar.grid_columnconfigure(1, weight=2)
        bottom_toolbar.grid_columnconfigure(2, weight=2)
        bottom_toolbar.grid_columnconfigure(3, weight=2)
        bottom_toolbar.grid_columnconfigure(4, weight=2)
        bottom_toolbar.grid_columnconfigure(5, weight=1)

        return bottom_toolbar

    def reload(self, is_window_already_loaded: bool, reload_POIs: bool):

        # Define logger
        logger = self.logger
        logger.debug(f"Running function reload({is_window_already_loaded}, {reload_POIs})")

        # Define variables
        root = self.main_window

        # Reload videos, clear POIs, reload video frames
        self.load_videos()
        if reload_POIs:
            self.reload_points_of_interest()
        self.load_video_frames()
        self.close_main_window()
        self.loaded = is_window_already_loaded
        self.open_main_window()

######################################### SECTION DEALS WITH GUI BACKEND #########################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################

    def change_excel_path(self):

        # Define logger
        self.logger.debug(f'Running function change_video_folder()')

        # Get new Excel path
        self.annotation_file_path = utils.get_excel_path(self.annotation_file_path, 0, self.video_folder_path, self.crop_mode)

        # Reload the window with the new input
        self.reload(False, True)

    def change_video_folder(self):
        self.logger.debug(f'Running function change_video_folder()')

        # Set loaded to false as when video folder is changed, the GUI must be reloaded.
        self.loaded = False

        # Get new video folder and reload the GUI
        self.video_folder_path, self.scanned_folders, self.dir_hierarchy = utils.get_video_folder(self.video_folder_path, 0)
        self.reload(False, True)

    def switch_folder(self, which):

        # Define logger
        self.logger.debug(f'Running function switch_folder({which})')

        # Set loaded to false as when video folder is changed, the GUI must be reloaded.
        self.loaded = False

        # Check if there is another folder either left or right in the directory hierarchy and if yes, then switch and reload.
        index = self.scanned_folders.index(os.path.basename(os.path.normpath(self.video_folder_path)))
        if index > 0 and which == "left":
            self.video_folder_path = os.path.join(os.path.dirname(self.video_folder_path), self.scanned_folders[index - 1])
            self.reload(False, True)
        if (index + 1) < len(self.scanned_folders) and which == "right":
            self.video_folder_path = os.path.join(os.path.dirname(self.video_folder_path), self.scanned_folders[index + 1])
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
        self.initial_values = [self.output_folder, self.scan_folders, self.prefix, self.crop_mode, self.frame_skip, self.frames_per_visit, self.filter_visitors,
                          self.yolo_processing, self.default_label_category, self.yolo_conf, self.randomize, self.whole_frame,
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
            frame = self.cap.read()[1]
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
        try:
            self.x_coordinate = int(self.config['OCR settings'].get('x_coordinate', '0').strip())
            self.y_coordinate = int(self.config['OCR settings'].get('y_coordinate', '0').strip())
            self.width = int(self.config['OCR settings'].get('width', '500').strip())
            self.height = int(self.config['OCR settings'].get('height', '40').strip())
        except ValueError:
            # Handle cases where conversion to integer fails
            self.logger.warning('Invalid integer value found in settings_crop.ini')
        # check if video_filepath is valid path to a video file
        if not os.path.isfile(video_filepath) or not video_filepath.endswith(".mp4"):
            self.logger.warning('Invalid video file path')
            return

        # Set up video cap
        self.cap = cv2.VideoCapture(video_filepath)

        # Create a window and pass it to the mouse callback function
        cv2.namedWindow('image')
        cv2.setWindowProperty('image', cv2.WND_PROP_TOPMOST, 1)

        # display rectangle on image from the text_roi coordinates
        cv2.setMouseCallback('image', draw_rectangle)
        while (self.cap.isOpened()):
            ret, frame = self.cap.read()
            if ret:
                cv2.rectangle(frame, (self.x_coordinate, self.y_coordinate), (self.x_coordinate + self.width, self.y_coordinate + self.height),
                              (0, 255, 0),
                              2)
                cv2.imshow('image', frame)
                k = cv2.waitKey(1) & 0xFF
                if k == 27:
                    cv2.destroyAllWindows()
                    break
            else:
                break
        self.cap.release()
        cv2.destroyAllWindows()

        # Save settings to settings_crop.ini
        self.config['OCR settings']['x_coordinate'] = str(self.x_coordinate)
        self.config['OCR settings']['y_coordinate'] = str(self.y_coordinate)
        self.config['OCR settings']['width'] = str(self.width)
        self.config['OCR settings']['height'] = str(self.height)
        with open('settings_crop.ini', 'w', encoding='utf-8') as configfile:
            self.config.write(configfile)

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
            frame = self.draw_roi_offset_boundaries(frame, self.points_of_interest_entry[index][0], dict_of_extremes_to_draw, True,
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
                    index == 0 or not self.points_of_interest_entry[max(index - 1, 0)][0] == self.points_of_interest_entry[index][0]):
                self.modified_frames.append(each)
            self.update_button_image(self.frames[each].copy(), (max(each, 0) // 6), (each - ((max(each, 0) // 6) * 6)), first)

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
                                                                                  self.points_of_interest_entry[index][0]):
            # Define the ROI
            frame = cv2.resize(frame, (276, 156), interpolation=cv2.INTER_AREA)
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
        pil_img = pil_img.resize((276, 156))

        # Convert PIL image to tkinter image
        img = ImageTk.PhotoImage(pil_img)
        self.button_images[index] = img
        self.buttons[i][j].configure(image=self.button_images[index])

    def draw_roi_offset_boundaries(self, frame, list_of_ROIs, dict_of_extremes_to_draw, draw_roi: bool, draw_extremes: bool,
                                   draw_overlap: bool):

        # Define variables
        labels = ['BR', 'UL', 'UR', 'BL']
        offsets = [(1, 1), (-1, -1), (1, -1), (-1, 1)]
        conditions = [dict_of_extremes_to_draw["b_r"] > 0, dict_of_extremes_to_draw["u_l"] > 0, dict_of_extremes_to_draw["u_r"] > 0, dict_of_extremes_to_draw["b_l"] > 0]
        pos_off = 0
        height, width, channels = frame.shape

        # For each point draw desired shapes
        for point in list_of_ROIs:

            # Define variables
            rectangles = []

            # Draw basic roi area
            if draw_roi:
                x1, y1 = max(0, point[0] - (self.crop_size // 2)), max(0, point[1] - (self.crop_size // 2))
                x2, y2 = min(width, point[0] + (self.crop_size // 2)), min(height, point[1] + (self.crop_size // 2))
                cv2.rectangle(frame, (point[0] - 30, point[1] - 30), (point[0] + 30, point[1] + 30), (0, 255, 0), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.line(frame, (point[0], point[1]), (x1, y1), (0, 255, 255), thickness=2, lineType=cv2.LINE_AA)
                cv2.line(frame, (point[0], point[1]), (x1, y2), (0, 255, 255), thickness=2, lineType=cv2.LINE_AA)
                cv2.line(frame, (point[0], point[1]), (x2, y1), (0, 255, 255), thickness=2, lineType=cv2.LINE_AA)
                cv2.line(frame, (point[0], point[1]), (x2, y2), (0, 255, 255), thickness=2, lineType=cv2.LINE_AA)

            # Draw the offset extremes rectangles
            if draw_extremes or draw_overlap:
                for i, (label, (offset_x, offset_y), condition) in enumerate(zip(labels, offsets, conditions)):
                    o_x1 = max(pos_off, min(((point[0] - self.crop_size // 2) + offset_x * self.offset_range) - pos_off,
                                            width - self.crop_size - pos_off))
                    o_y1 = max(pos_off, min(((point[1] - self.crop_size // 2) + offset_y * self.offset_range) - pos_off,
                                            height - self.crop_size - pos_off))
                    o_x2 = max(self.crop_size - pos_off,
                               min(((point[0] + self.crop_size // 2) + offset_x * self.offset_range) - pos_off,
                                   width - pos_off))
                    o_y2 = max(self.crop_size - pos_off,
                               min(((point[1] + self.crop_size // 2) + offset_y * self.offset_range) - pos_off,
                                   height - pos_off))
                    rectangles.append([(o_x1, o_y1), (o_x2, o_y2)])
                    if condition:
                        cv2.rectangle(frame, (o_x1, o_y1), (o_x2, o_y2), (255, 229, 0), 2)

                # Draw overlap area
                if draw_overlap:
                    # Find the overlapping area
                    x_min = max(rect[0][0] for rect in rectangles)
                    y_min = max(rect[0][1] for rect in rectangles)
                    x_max = min(rect[1][0] for rect in rectangles)
                    y_max = min(rect[1][1] for rect in rectangles)

                    # Draw the rectangle of the overlap
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        return frame

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
                filepath = os.path.join(self.video_folder_path, 'crop_information.pkl')
                with open(filepath, 'wb') as f:
                    # Use the pickle module to write the data to the file
                    pickle.dump([self.auto_processing.get(), self.points_of_interest_entry, self.video_filepaths], f)
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
                filepath = os.path.join(self.video_folder_path, 'crop_information.pkl')
                if os.path.isfile(filepath):
                    try:
                        if self.main_window.winfo_exists():
                            self.main_window.destroy()
                    except:
                        self.logger.debug("Error: Unexpected, window destroyed before reference.")
                    self.points_of_interest_entry = []
                    with open(filepath, 'rb') as f:
                        data = pickle.load(f)
                    self.auto = data[0]
                    self.points_of_interest_entry = data[1].copy()
                    video_filepaths_new = data[2].copy()
                    set1 = set({os.path.basename(filepath) for filepath in video_filepaths_new})
                    set2 = set({os.path.basename(filepath) for filepath in self.video_filepaths})
                    if not set1 == set2:
                        messagebox.showinfo("Discrepancy detected",
                                            "The contents of the video folder have changed since the save has been made. Cannot load the progress. Please start over.")
                        self.reload_points_of_interest()
                    else:
                        self.video_filepaths = []
                        self.video_filepaths = video_filepaths_new.copy()
                    self.reload(True, False)
                else:
                    messagebox.showinfo("No save detected",
                                        "There are no save files in the current directory.")
            else:
                messagebox.showerror("Error", "Invalid video folder path")

iccs = ICCS()