# This file contains the ancestral class of application used in ICVT. Any other tool can inherit from this.
import utils
import os
import tkinter as tk
import logging

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

        # Load up functions to get the video and excel folders
        self.scan_default_folders()
        while not self.check_path():
            self.get_video_folder(1)
        self.get_excel_path(1, 1)

    def scan_default_folders(self):
        self.video_folder_path, self.annotation_file_path = utils.scan_default_folders(self.scan_folders)

    def check_path(self):
        return utils.check_path(self.video_folder_path, 0)

    def get_video_folder(self, check):
        self.video_folder_path, self.scanned_folders, self.dir_hierarchy = utils.get_video_folder(self.video_folder_path, check)

    def get_excel_path(self, check, excel_type):
        self.annotation_file_path = utils.get_excel_path(self.annotation_file_path, check, self.video_folder_path, excel_type)

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
            logger.info("Error: Unexpected, window destroyed before reference.")

    def log_define(self):

        # Create a logger instance
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        # Create a file handler that logs all messages, and set its formatter
        file_handler = logging.FileHandler('runtime.log', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # Create a console handler that logs only messages with level INFO or higher, and set its formatter
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

app = AppAncestor()
