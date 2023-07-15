# This file contains the ICID app class that inherits from ICVT AppAncestor class
import utils
import anno_data
import vid_data
import inat_id
import icvt
import pandas as pd
import os
import subprocess
import threading
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
import asyncio
import cProfile
import tracemalloc
from typing import Dict, Callable
from datetime import datetime
from datetime import timedelta
os.environ["PATH"] = os.path.dirname(__file__) + os.pathsep + os.environ["PATH"]
import mpv
import requests

class ICID(icvt.AppAncestor):
    def __init__(self):
        # Define logger
        self.logger = self.log_define()

        # First log
        self.logger.info("Initializing ICID - Insect Communities ID application class...")

        # Start the loading bar in a separate thread
        time.sleep(0.2)
        loading_thread = threading.Thread(target=self.loading_bar)
        loading_thread.start()
        self.loading_progress = 1
        self.stop_loading = True

        # Init basic instance variables and get config
        self.app_title = "Insect Communities ID"
        self.config = self.config_create()
        self.config_read()
        self.loading_progress = 20

        # Define variables
        self.video_filepaths = []
        self.player = None
        self.dir_hierarchy = False
        self.loading_progress = 25

        # Initiation functions - get directories and files
        self.scan_default_folders()
        self.loading_progress = 35

        # If video folder path not supplied ask user to specify it
        while not self.check_path():
            self.get_video_folder(1)
        self.loading_progress = 40

        # Ask the user to specify the Excel path
        self.get_excel_path(1, 1)
        self.loading_progress = 45

        # Load the videos
        self.main_window = tk.Tk()
        self.load_videos()
        self.close_main_window()

        # Process visits
        self.visit_processing_engine()

        # Open the main window
        # self.loading_progress = 100
        # time.sleep(0.5)
        # self.stop_loading = True
        # loading_thread.join()
        # self.open_main_window()

    def config_create(self):
        global video_folder_path
        global annotation_file_path
        global scan_folders
        global config
        global logger

        # Define logger
        self.logger.debug('Running function config_create()')

        # Create the file
        # Set default values
        config = configparser.ConfigParser()
        config['Resource Paths'] = {
            'video_folder_path': '',
            'annotation_file_path': ''
        }
        config['ICID settings'] = {
            'Scan_default_folders': '1'
        }

        # Check if settings_crop.ini exists, and create it with default values if not
        if not os.path.exists('settings_ICID.ini'):
            with open('settings_ICID.ini', 'w', encoding='utf-8') as configfile:
                config.write(configfile)

        return config

    def config_read(self):

        # Define logger
        self.logger.debug('Running function config_read()')

        # Read settings from settings_crop.ini
        config.read('settings_ICID.ini', encoding='utf-8')

        # Get values from the config file
        try:
            self.video_folder_path = config['Resource Paths'].get('video_folder_path', '').strip()
            self.annotation_file_path = config['Resource Paths'].get('annotation_file_path', '').strip()
            self.scan_folders = config['ICID settings'].get('Scan_default_folders', '0').strip()
        except ValueError:
            self.logger.warning('Invalid folder/file path found in settings_ICID.ini')

    def open_main_window(self, time_of_visit, video_filepath):

        def set_window_geometry(window, width, height, x, y):
            screen_width = window.winfo_screenwidth()
            screen_height = window.winfo_screenheight()
            window.geometry(f"{width}x{height}+{x}+{y}")

        self.close_main_window()

        if self.player is not None:
            self.player.terminate()


        # Define logger
        self.logger.debug(f"Running function open_main_window()")

        # Define variables
        self.zoom_level = 0.0
        self.video_pan_x = 0.0
        self.video_pan_y = 0.0
        self.speed = 1.0
        j = 0

        # Create the main tkinter window (root)
        root = tk.Tk()
        self.main_window = root
        set_window_geometry(root, root.winfo_screenwidth() * 2 // 3, root.winfo_screenheight(), -100, 0)
        root.title("MPV Window")

        # Create the MPV player and play the video
        self.player = mpv.MPV(player_operation_mode='pseudo-gui',
                         script_opts='osc-deadzonesize=0,osc-minmousemove=1',
                         input_default_bindings=True,
                         input_vo_keyboard=True,
                         osc=True)
        self.player.geometry = f'{root.winfo_screenwidth() * 2 // 3}x{root.winfo_screenheight()}+-50+-20'
        self.player.start = time_of_visit  # Adjust the value as needed
        self.player.play(video_filepath)


        # Create the second tkinter window (control_panel)
        control_panel = tk.Toplevel(root)
        set_window_geometry(control_panel, root.winfo_screenwidth() // 4, root.winfo_screenheight(),
                            root.winfo_screenwidth() * 2 // 3 - 100, 0)
        control_panel.title("Control Panel")

        # Create the input fields and labels in control_panel
        label_texts = ["Label 1", "Label 2", "Label 3", "Label 4", "Label 5", "Label 6", "Label 7",
                       "Label 8", "Label 9", "Label 10", "Label 11", "Label 12", "Label 13"]
        initial_values = ["Value 1", "Value 2", "Value 3", "Value 4", "Value 5", "Value 6", "Value 7",
                          "Value 8", "Value 9", "Value 10", "Value 11", "Value 12", "Value 13"]

        # Create the input fields and labels
        for i in range(len(label_texts)):
            label = tk.Label(control_panel, text=label_texts[i])
            label.grid(row=i, column=1, padx=10, pady=5)

            entry = tk.Entry(control_panel)
            entry.insert(tk.END, initial_values[i])
            entry.grid(row=i, column=2, padx=10, pady=5)

        # Create the additional labels and input fields
        visitor_species_label = tk.Label(control_panel, text="Visitor Species")
        visitor_species_label.grid(row=len(label_texts), column=1, padx=10, pady=5)

        visitor_species_entry = tk.Entry(control_panel)
        visitor_species_entry.grid(row=len(label_texts), column=2, padx=10, pady=5)

        visitor_order_label = tk.Label(control_panel, text="Visitor Order")
        visitor_order_label.grid(row=len(label_texts) + 1, column=1, padx=10, pady=5)

        visitor_order_entry = tk.Entry(control_panel)
        visitor_order_entry.grid(row=len(label_texts) + 1, column=2, padx=10, pady=5)

        # Create the buttons for video playback control
        button_frame = tk.Frame(control_panel)
        button_frame.grid(row=len(label_texts) + 2, column=1, columnspan=2, pady=10)

        play_button = tk.Button(button_frame, text="Play",
                                command=lambda j=j: self.player.play(video_filepath))
        play_button.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        pause_button = tk.Button(button_frame, text="Pause", command=lambda j=j: self.pause_vid(self.player))
        pause_button.grid(row=0, column=1, padx=5, pady=5, sticky="nsew", columnspan=2)

        stop_button = tk.Button(button_frame, text="Stop", command=lambda j=j: self.player.stop(True))
        stop_button.grid(row=0, column=3, padx=5, pady=5, sticky="nsew")

        frame_back_button = tk.Button(button_frame, text="<- Frame", command=lambda j=j: self.player.frame_back_step())
        frame_back_button.grid(row=1, column=0, padx=5, pady=5, sticky="nsew", columnspan=2)

        frame_forw_button = tk.Button(button_frame, text="Frame ->", command=lambda j=j: self.player.frame_step())
        frame_forw_button.grid(row=1, column=2, padx=5, pady=5, sticky="nsew", columnspan=2)

        zoom_in_button = tk.Button(button_frame, text="Zoom In", command=lambda j=j: self.zoom_vid(self.player, 1))
        zoom_in_button.grid(row=2, column=0, padx=5, pady=5, sticky="nsew", columnspan=2)

        zoom_out_button = tk.Button(button_frame, text="Zoom Out", command=lambda j=j: self.zoom_vid(self.player, -1))
        zoom_out_button.grid(row=2, column=2, padx=5, pady=5, sticky="nsew", columnspan=2)

        sshot_button = tk.Button(button_frame, text="Screenshot",
                                 command=lambda j=j: self.player.screenshot_to_file("1.png"))
        sshot_button.grid(row=3, column=0, padx=5, pady=5, sticky="nsew", columnspan=2)

        id_button = tk.Button(button_frame, text="Identify", command=lambda j=j: self.id_insect())
        id_button.grid(row=3, column=2, padx=5, pady=5, sticky="nsew", columnspan=2)

        pan_up_button = tk.Button(button_frame, text="Pan Up", command=lambda j=j: self.pan_vid(self.player, "down"))
        pan_up_button.grid(row=4, column=2, padx=5, pady=5, sticky="nsew")

        pan_left_button = tk.Button(button_frame, text="Pan Left", command=lambda j=j: self.pan_vid(self.player, "right"))
        pan_left_button.grid(row=5, column=1, padx=5, pady=5, sticky="nsew")

        center_button = tk.Button(button_frame, text="Center", command=lambda j=j: self.center_vid(self.player))
        center_button.grid(row=5, column=2, padx=5, pady=5, sticky="nsew")

        pan_right_button = tk.Button(button_frame, text="Pan right", command=lambda j=j: self.pan_vid(self.player, "left"))
        pan_right_button.grid(row=5, column=3, padx=5, pady=5, sticky="nsew")

        pan_down_button = tk.Button(button_frame, text="Pan Down", command=lambda j=j: self.pan_vid(self.player, "up"))
        pan_down_button.grid(row=6, column=2, padx=5, pady=5, sticky="nsew")

        speed_up1_button = tk.Button(button_frame, text="+0.5×", command=lambda j=j: self.speed_vid(self.player, 0.5))
        speed_up1_button.grid(row=8, column=0, padx=5, pady=5, sticky="nsew", columnspan=1)

        speed_up2_button = tk.Button(button_frame, text="+0.25×", command=lambda j=j: self.speed_vid(self.player, 0.25))
        speed_up2_button.grid(row=8, column=1, padx=5, pady=5, sticky="nsew", columnspan=1)

        speed_norm_button = tk.Button(button_frame, text="1×", command=lambda j=j: self.speed_vid(self.player, 0))
        speed_norm_button.grid(row=8, column=2, padx=5, pady=5, sticky="nsew", columnspan=1)

        speed_down_button = tk.Button(button_frame, text="-0.25×", command=lambda j=j: self.speed_vid(self.player, -0.25))
        speed_down_button.grid(row=8, column=3, padx=5, pady=5, sticky="nsew", columnspan=1)

        speed_down2_button = tk.Button(button_frame, text="-0.5×", command=lambda j=j: self.speed_vid(self.player, -0.5))
        speed_down2_button.grid(row=8, column=4, padx=5, pady=5, sticky="nsew", columnspan=1)

        root.mainloop()

        print('calling player.terminate.')
        self.player.terminate()
        print('terminate player.returned.')

    def pause_vid(self, player):
        if player.pause == True:
            player.pause = False
        else:
            player.pause = True

    def zoom_vid(self, player, where):
        global zoom_level
        if where > 0:
            zoom_level = zoom_level + 0.1  # Adjust the value as needed
            player.video_zoom = zoom_level
        else:
            zoom_level = zoom_level - 0.1  # Adjust the value as needed
            player.video_zoom = zoom_level

    def pan_vid(self, player, where):
        global video_pan_x
        global video_pan_y
        if where == "left":
            video_pan_x = video_pan_x - 0.05  # Adjust the value as needed
            player.video_pan_x = video_pan_x
        elif where == "right":
            video_pan_x = video_pan_x + 0.05  # Adjust the value as needed
            player.video_pan_x = video_pan_x
        elif where == "up":
            video_pan_y = video_pan_y - 0.05  # Adjust the value as needed
            player.video_pan_y = video_pan_y
        elif where == "down":
            video_pan_y = video_pan_y + 0.05  # Adjust the value as needed
            player.video_pan_y = video_pan_y

    def center_vid(self, player):
        global video_pan_x
        global video_pan_y
        global zoom_level
        player.video_pan_x = 0
        player.video_pan_y = 0
        player.video_zoom = 0.0

    def speed_vid(self, player, how):
        global speed
        if how == 0:
            speed = 1.0
            player.speed = 1.0
        else:
            speed = speed + how  # Adjust the value as needed
            player.speed = speed

    def id_insect(self):
        client_id = "LWjPdw733dsLPmr_05T9GWYX-9_qZ40TcGNtUq4ZPvk"
        client_secret = 'HhXxWAsXVm-5SQHth_TEKZc2JAf4mUkKi9oFS6DuSzM'
        username = 'petaschlup@seznam.cz'
        password = '8*u;n2H5-gLQYV]'
        identificator = inat_id.InatIdentificator(client_id, client_secret, username, password)

        observation_data = {
            "species_guess": "Amegilla quadrifasciata",
            "taxon_id": 47158,
            "observed_on_string": "2022-05-24",
            "time_zone": "Prague",
            "description": "large bee",
            "tag_list": "GR2",
            "place_guess": "Lesvos, GR",
            "latitude": 39.166666,
            "longitude": 26.333332,
            "map_scale": 11,
            "location_is_exact": False,
            "positional_accuracy": 1000,
            "geoprivacy": "obscured",
        }

        obs = identificator.create_observation(observation_data)
        obs.upload_image("img.jpg")

    def visit_processing_engine(self):

        # Define logger
        self.logger.debug("Running function visit_processing_engine()")

        # Check validity of paths
        video_ok = utils.check_path(self.video_folder_path, 0)
        excel_ok = utils.check_path(self.annotation_file_path, 1)
        if not video_ok or not excel_ok:
            messagebox.showinfo("Warning",
                                f"Unspecified path to a video folder or a valid Excel file.")
            return

        # Load data from excel
        excel = anno_data.Annotation_watcher_file(self.annotation_file_path, True, True, True, True)
        annotation_data_array = excel.dataframe.loc[:, ['duration', 'ts']].values.tolist()
        visitor_id = excel.dataframe.loc[:, ['vis_id']].values.tolist()

        # If no annotations were extracted end cropping process
        if annotation_data_array is None:
            messagebox.showinfo("Warning",
                                f"Attempted to fix errors in the selected Excel file. Attempt failed. Please fix the errors manually and try again.")
            return

        # Filter video filepaths to only those relevant for the annotations that are to be processed
        sorted_video_filepaths = self.get_relevant_video_paths(self.video_filepaths, annotation_data_array)

        # Get video data
        video_data = self.get_video_data(sorted_video_filepaths)

        # Log the information that are fed into the rest of the engine
        self.logger.debug(f"Processing visits according to the following annotations: {annotation_data_array}")
        self.logger.debug(f"Processing visits with the following video data: {video_data}")

        # Construct the valid annotation data array which contains visit data coupled with the path to the
        # video containing the visit
        valid_annotations_array = self.construct_valid_annotation_array(annotation_data_array, video_data)

        # Append also visitor id - here I could add anything else I want to append from
        # excel.dataframe
        for annotation, vis_id in zip(valid_annotations_array, visitor_id):
            annotation += vis_id

        def create_button_text(index):
            return f"{index + 1}        10:24:05 and 10:24:05"

        def create_buttons(parent):
            for i in range(len(valid_annotations_array)):
                button_text = create_button_text(i)

                # Define the variables
                visit_duration, visit_timestamp, video_filepath, video_start_time, *_ = valid_annotations_array[i]

                # Turn timestamp into datetime and calculate how many seconds from the start_time of the video recording does the visit take place
                visit_time = pd.to_datetime(visit_timestamp, format='%Y%m%d_%H_%M_%S')
                visit_time_from_start = (visit_time - video_start_time).total_seconds()
                self.visit_times.append(visit_time_from_start)
                print(visit_time)
                print(visit_time_from_start)
                print(video_filepath)
                print(video_start_time)
                button = tk.Button(parent, text=button_text, height=2, width=30, command=lambda i=i: self.open_main_window(self.visit_times[i], valid_annotations_array[i][2]))
                button.grid(row=i, column=0, sticky="ew")
                self.buttons.append(button)

        # Create the Tkinter window
        window = tk.Tk()

        # Get screen height and calculate window height
        screen_height = window.winfo_screenheight()
        window_height = screen_height

        # Get screen width and calculate window width
        screen_width = window.winfo_screenwidth()
        window_width = screen_width // 8

        # Set window dimensions and position
        window.geometry(f"{window_width}x{window_height}+{screen_width - window_width}+0")

        # Create buttons dynamically based on the array length
        outer_frame = tk.Frame(window)
        outer_frame.pack(side=tk.TOP, expand=True, fill=tk.BOTH, padx=10, pady=10)
        self.buttons = []
        self.visit_times = []
        create_buttons(outer_frame)
        outer_frame.grid_columnconfigure(0, weight=2, minsize=20)

        # Start the Tkinter event loop
        window.mainloop()

        # Process the visits and generate cropped images
        for index in range(len(valid_annotations_array)):
            self.logger.info(f"Processing visit {index + 1}")

            # Define index which will be shared to sync the stage of the process
            self.visit_index = index

            # Define the variables
            visit_duration, visit_timestamp, video_filepath, video_start_time, *_ = valid_annotations_array[index]

            # Turn timestamp into datetime and calculate how many seconds from the start_time of the video recording does the visit take place
            visit_time = pd.to_datetime(visit_timestamp, format='%Y%m%d_%H_%M_%S')
            visit_time_from_start = (visit_time - video_start_time).total_seconds()



icid = ICID()