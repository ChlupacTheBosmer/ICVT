# This file contains the ancestral class of application used in ICVT. Any other tool can inherit from this.
import utils
import vid_data
import os
import tkinter as tk
from tkinter import messagebox
import logging
from datetime import datetime, timedelta
from datetime import timedelta
import pandas as pd
import sys
import time
import random
from PIL import Image, ImageTk
import numpy as np
import cv2

class AppAncestor:
    def __init__(self):
        # Define logger
        self.logger = self.log_define()

        # Set basic instance variables
        self.app_title = "Insect Communities Video Tools"
        self.video_folder_path = "videos"
        self.annotation_file_path = "excel"
        self.scan_folders = "1"
        self.scanned_folders = []
        self.dir_hierarchy = False
        self.gui_imgs = []
        self.ocr_roi = ()

        # Load up functions to get the video and excel folders
        # self.scan_default_folders()
        # while not self.check_path():
        #     self.get_video_folder(1)
        # self.get_excel_path(1, 1)

    def scan_default_folders(self):
        self.video_folder_path, self.annotation_file_path = utils.scan_default_folders(self.scan_folders)

    def check_path(self):
        return utils.check_path(self.video_folder_path, 0)

    def get_video_folder(self, check):
        self.video_folder_path, self.scanned_folders, self.dir_hierarchy = utils.get_video_folder(self.video_folder_path, check)

    def get_excel_path(self, check, excel_type):
        self.annotation_file_path = utils.get_excel_path(self.annotation_file_path, check, self.video_folder_path, excel_type)

    def load_videos(self):

        # Define logger
        self.logger.debug('Running function load_videos()')

        self.video_filepaths = []

        # Check if the video folder path is valid
        if utils.check_path(self.video_folder_path, 0):

            # Look for corrupted videos
            utils.delete_corrupted_videos(self.video_folder_path)

            # Load videos
            try:
                self.video_filepaths = [
                    os.path.join(self.video_folder_path, f)
                    for f in os.listdir(self.video_folder_path)
                    if (f.endswith('.mp4') or f.endswith('.avi'))
                ]
            except OSError as e:
                self.logger.error(f"Failed to load videos: {e}")
                self.video_filepaths = []
        else:
            self.logger.error("Error", "Invalid video folder path")
            self.video_filepaths = []

    def get_video_data(self, video_filepaths, return_video_file_objects: bool = False):

        # Define logger
        self.logger.debug(f"Running function get_video_data({video_filepaths})")

        # loop through time annotations and open corresponding video file
        # extract video data beforehand to save processing time
        video_data = []
        video_files = []
        i: int
        for i, filepath in enumerate(video_filepaths):
            if filepath.endswith('.mp4') or filepath.endswith('.avi'):
                video = vid_data.Video_file(video_filepaths[i], self.main_window, self.ocr_roi)
                video_files.append(video)
                video_data_entry = [video.filepath, video.start_time, video.end_time]
                video_data.append(video_data_entry)
        if return_video_file_objects:
            return video_data, video_files
        else:
            return video_data

    def get_relevant_video_paths(self, video_filepaths, annotation_data_array):
        new_video_filepaths = set()
        for visit_data in annotation_data_array:
            for filepath in video_filepaths:
                if visit_data[1][:-9] in os.path.basename(filepath):
                    time_difference = datetime.strptime(visit_data[1][-8:-3], '%H_%M') - datetime.strptime(
                        filepath[-9:-4], '%H_%M')
                    if timedelta() <= time_difference <= timedelta(minutes=15):
                        new_video_filepaths.add(filepath)
                        break  # Exit the inner loop after finding a match

        # Convert the set back to a list
        new_video_filepaths = list(new_video_filepaths)
        return new_video_filepaths

    def construct_valid_annotation_array(self, annotation_data_array, video_data):
        # Iterate over annotation_data_array along with the index and construct the valid annotation data array which
        # contains visit data coupled with the path to the video containing the visit
        valid_annotations_array = []
        for index, annotation_data in enumerate(annotation_data_array):
            # Get the annotation time
            annotation_time = pd.to_datetime(annotation_data[1], format='%Y%m%d_%H_%M_%S')

            # Find the corresponding video data entry
            relevant_video_data = next(
                (video_data_entry for video_data_entry in video_data if video_data_entry[1] <= annotation_time <= video_data_entry[2]),
                None)

            if relevant_video_data:
                valid_annotation_data_entry = annotation_data + relevant_video_data

                self.logger.debug(f"Relevant annotations: {valid_annotation_data_entry}")
                valid_annotations_array.append(valid_annotation_data_entry)

            # Clear the list if no longer needed
            valid_annotation_data_entry = []
        # valid annotation entry: [duration, time_of_visit, video_filepath, video_start_time, video_end_time]
        return valid_annotations_array

    def open_main_window(self):

        # Load logger
        logger = self.logger
        logger.debug(f"Running function open_window()")

        # Create tkinter window
        root = tk.Tk()
        self.main_window = root
        root.focus()
        root.title(
            f"{self.app_title} - Folder: {os.path.basename(os.path.normpath(self.video_folder_path))} - Table: {os.path.basename(os.path.normpath(self.annotation_file_path))}")

        # Get the screen width and height
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()

        # Set the window size to fit the entire screen
        root.geometry(f"{screen_width}x{screen_height - 70}+-10+0")
        root.mainloop()

    def close_main_window(self):

        # Load logger
        logger = self.logger

        # Attempts to close the main app window
        root = self.main_window
        try:
            if root.winfo_exists():
                root.destroy()
        except:
            logger.debug("Unexpected, window destroyed before reference.")

    def loading_bar(self):
        self.loading_progress = 0
        self.stop_loading = False
        while not self.stop_loading:
            index = int(self.loading_progress//1)
            bar = "█" * (index) + "▒" * (100 - index)
            sys.stdout.write("\r" + "Loading:  " + bar)
            if not index == 100:
                self.loading_progress += random.uniform(0.1, 1)
                time.sleep(0.1)
            else:
                time.sleep(0.1)
        sys.stdout.flush()

    def log_define(self):

        # Create a logger instance
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)

        # Create a file handler that logs all messages, and set its formatter
        file_handler = logging.FileHandler('runtime.log', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(filename)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Create a console handler that logs only messages with level INFO or higher, and set its formatter
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        return logger

    def load_icon(self, path, size: tuple = (50, 50), master = None):

        stream = open(path, "rb")
        bytes = bytearray(stream.read())
        numpyarray = np.asarray(bytes, dtype=np.uint8)
        img1 = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)

        # Convert the BGR image to RGB (Tkinter uses RGB format)
        # img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

        # Convert the NumPy array to a PIL image
        pil_img = Image.fromarray(img1)

        img = pil_img.resize(size)

        if not master == None:
            img = ImageTk.PhotoImage(master=master, image=pil_img)
        else:
            img = ImageTk.PhotoImage(pil_img)
        self.gui_imgs.append(img)
        return img

