# This file contains all shared functions and classes

import pandas as pd
import os
import subprocess
import re
import cv2
import pytesseract
import configparser
import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog
from PIL import Image, ImageTk
import pickle
import datetime
import sys
import random
import openpyxl
import math
import time
from hachoir.metadata import extractMetadata
from hachoir.parser import createParser
import logging
import xlwings as xw
import keyboard
from ultralytics import YOLO
import torch
import shutil

def ask_yes_no(text):
    global logger
    logger.debug(f'Running function ask_yes_no({text})')
    result: bool = messagebox.askyesno("Confirmation", text)
    return result

def create_dir(path):
    global logger
    logger.debug(f'Running function create_dir({path})')
    if not os.path.exists(path):
        os.makedirs(path)

def scan_default_folders(scan_folders, file_type_index: int = 1):
    # The scan_default_folders function scans folders for annotation files and videos. The function takes an optional
    # parameter file_type_index, which represents the type of Excel file expected. Typically, it will reflect the crop
    # mode settings in the crop.py script. By default, the index is set to 1, indicating that Excel annotation
    # files produced by the watchers will be expected. Setting the index to any value other than 1 or 2, such as 0,
    # will cause the function to skip scanning the Excel folder for annotation files since it is not necessary.

    # The nested function select_file deals with the processing of the user made selection of the Excel to load.

    def select_file(selected_file_index, index, window):
        global logger
        logger.debug(f'Running function select_file({selected_file_index}, {index}, {window})')
        selected_file_index.set(index + 1)
        window.destroy()

    global logger
    logger.debug('Running function scan_default_folders()')

    # set which file type will be expected
    file_types = ["excel (watchers)", "excel (manual)"]

    # Check if scan folder feature is on
    if scan_folders == "1":

        # Create directories if they do not exist
        create_dir("videos/")
        create_dir("excel/")

        # Detect video files
        video_folder_path: str = ""
        scan_video_files = [f for f in os.listdir('videos') if f.endswith('.mp4')]
        if scan_video_files:
            response = ask_yes_no(f"Video files detected in the default folder. Do you want to continue?")
            if response:
                video_folder_path = 'videos'

        # Check if the current default crop mode requires an annotation file.
        if file_type_index == 1 or file_type_index == 2:

            # Detect Excel files
            annotation_file_path: str = ""
            scan_excel_files = [f for f in os.listdir('excel') if f.endswith('.xlsx') or f.endswith('.xls')]
            if scan_excel_files:
                response = ask_yes_no(f"Excel files detected in the default folder. Do you want to continue?")
                if response:

                    # Create the window for selecting the Excel file
                    excel_files_win = tk.Tk()
                    excel_files_win.title("Select file")
                    excel_files_win.wm_attributes("-topmost", 1)

                    # Create window contents
                    prompt_label = tk.Label(excel_files_win,
                                            text=f"Please select the {file_types[(file_type_index - 1)]} file you\nwant to use as the source of visit times.")
                    prompt_label.pack()
                    label = tk.Label(excel_files_win, text="Excel files in the folder:")
                    label.pack()
                    outer_frame = tk.Frame(excel_files_win)
                    outer_frame.pack(side=tk.TOP, expand=True, fill=tk.BOTH, padx=20, pady=20)
                    for i, f in enumerate(scan_excel_files):
                        button = tk.Button(outer_frame, text=f"{i + 1}. {f}", width=30,
                                           command=lambda i=i: select_file(selected_file_index, i, excel_files_win))
                        button.pack(pady=0)
                    selected_file_index = tk.IntVar()

                    # Set the window position to the center of the screen
                    excel_files_win.update()
                    screen_width = excel_files_win.winfo_screenwidth()
                    screen_height = excel_files_win.winfo_screenheight()
                    window_width = excel_files_win.winfo_reqwidth()
                    window_height = excel_files_win.winfo_reqheight()
                    x_pos = int((screen_width - window_width) / 2)
                    y_pos = int((screen_height - window_height) / 2)
                    excel_files_win.geometry(f"{window_width}x{window_height}+{x_pos}+{y_pos}")
                    excel_files_win.mainloop()
                    selection = selected_file_index.get()

                    # Assign the path of the selected file to a variable
                    if selection > 0 and selection <= len(scan_excel_files):
                        annotation_file_path = os.path.join(("excel/"), scan_excel_files[selection - 1])
                        logger.debug(f'Selected file: {annotation_file_path}')
                    else:
                        logger.warning('Invalid selection')
    return video_folder_path, annotation_file_path

def check_path(path, path_type: int = 0):
    # The check_paths function validates a given path based on the specified path_type. If path_type is 0, it checks if the
    # provided path is a valid path to a folder. If path_type is 1, it checks if the provided path is a valid path to an
    # Excel file. The function returns True if the path is valid according to the specified type, and False otherwise.
    # The function helps ensure that the provided paths are correct for further processing or usage.

    path_ok = False
    if path_type == 0:
        video_folder_path = path
        # Check if the video folder path is valid path to a folder
        if not os.path.isdir(video_folder_path):
            logger.error(f"Video folder path is not valid: {video_folder_path}")
            path_ok = False
        else:
            path_ok = True
    if path_type == 1:
        annotation_file_path = path
        # Check if the annotation file path is valid path to an Excel file
        if not os.path.isfile(annotation_file_path) or not annotation_file_path.endswith(".xlsx"):
            logger.error(f"Annotation file path is not valid: {annotation_file_path}")
            path_ok = False
        else:
            path_ok = True
    return path_ok

def get_video_folder(video_folder_path, check):

    # Define logger
    global logger
    logger.debug(f'Running function get_video_folder({check})')

    # Define variables
    tree_allow = False

    # set path to folder containing mp4 files
    original_video_folder_path = video_folder_path

    # If there is invalid folder path in the video_folder_path variable or when the control is surpased by setting check
    # to zero then the file dialog is opened. If no path is selected and originally there was a valid path in the
    # video_folder_path variable then the original path is preserved.
    scanned_folders = []
    if not check_path(video_folder_path, 0) or check == 0:
        video_folder_path = filedialog.askdirectory(title="Select the video folder",
                                                    initialdir=os.path.dirname(os.path.abspath(__file__)))
        if video_folder_path == "" and original_video_folder_path != "":
            video_folder_path = original_video_folder_path

        # If validity control was bypassed and new path was selected or original validity check failed - and also - there is a valid
        # video folder path now. Then get all video files in the folder. Also get all folders in the folder.
        if ((check == 0 and video_folder_path != original_video_folder_path) or check == 1) and check_path(video_folder_path, 0):
            scan_video_files = [f for f in os.listdir(video_folder_path) if f.endswith('.mp4')]
            scanned_folders = [f for f in os.listdir(video_folder_path) if
                              os.path.isdir(os.path.join(video_folder_path, f))]
            # If there are no video files in the folder but there are folders. Then get all video files in the first
            # child folder. If there are any video files then tree_allow will be set to 1 meaning that user will be
            # allowed to navigate in between the folders which are expected to contain video files.
            if not scan_video_files and scanned_folders:
                scan_child_video_files = [f for f in os.listdir(os.path.join(video_folder_path, scanned_folders[0])) if
                                          f.endswith('.mp4')]
                if scan_child_video_files:
                    video_folder_path = os.path.join(video_folder_path, scanned_folders[0])
                    tree_allow = True
            else:
                tree_allow = False
            # If check was bypassed - which is a situation when user is calling this function from the GUI. then reload the window.
            # if check == 0:
            #     reload(0, True)
    else:
        logger.debug(f"Obtained video folder path: {video_folder_path}")
    return video_folder_path, scanned_folders, tree_allow

def get_excel_path(annotation_file_path, check, ini_dir, excel_type):
    global logger
    logger.debug(f'Running function get_excel_path({check}, {ini_dir})')
    # Set path to Excel file manually
    file_type = ["excel (watchers)", "excel (manual)"]
    if excel_type == 1 or excel_type == 2:
        if not check_path(annotation_file_path, 1) or check == 0:
            annotation_file_path = filedialog.askopenfilename(
                title=f"Select the path to the {file_type[(excel_type - 1)]} file",
                initialdir=ini_dir,
                filetypes=[("Excel Files", "*.xlsx"), ("Excel Files", "*.xls")])
        return annotation_file_path
    else:
        #display message box that the crop mode does not require an annotation file
        messagebox.showinfo("Info", "The current mode does not require an annotation file.")


def log_define():
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

global logger
logger = log_define()