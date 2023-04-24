import pandas as pd
import os
import subprocess
import re
import cv2
import pytesseract
import configparser
import tkinter as tk
from tkinter import simpledialog
from tkinter import messagebox
from tkinter import ttk
from tkinter import filedialog
import time
from PIL import Image, ImageTk
import pickle
import shutil
import datetime
import sys
import openpyxl
import random
import numpy as np
import openpyxl
import math

#Define all required functions
def config_read():
    global ocr_tesseract_path
    global video_folder_path
    global annotation_file_path
    global output_folder
    global scan_folders
    global crop_mode
    global frame_skip
    global randomize
    global whole_frame
    global cropped_frames
    global crop_size
    global offset_range
    global config
    global prefix
    # load config or create the file
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

    # Read settings from settings_crop.ini
    config.read('settings_crop.ini', encoding='utf-8')

    # Get values from the config file
    try:
        ocr_tesseract_path = config['Resource Paths'].get('OCR_tesseract_path', 'C:/Program Files/Tesseract-OCR/tesseract.exe').strip()
        video_folder_path = config['Resource Paths'].get('video_folder_path', '').strip()
        annotation_file_path = config['Resource Paths'].get('annotation_file_path', '').strip()
        output_folder = config['Resource Paths'].get('output_folder', 'output').strip()
        scan_folders = config['Workflow settings'].get('Scan_default_folders', '0').strip()
    except ValueError:
        print('Error: Invalid folder/file path found in settings_crop.ini')
    #Get crop values from config
    try:
        crop_mode = int(config['Crop settings'].get('crop_mode', '1').strip())
        frame_skip = int(config['Crop settings'].get('crop_interval_frames', '30').strip())
        randomize = int(config['Crop settings'].get('randomize_interval', '0').strip())
        whole_frame = int(config['Crop settings'].get('export_whole_frame', '0').strip())
        cropped_frames = int(config['Crop settings'].get('export_crops', '1').strip())
        crop_size = int(config['Crop settings'].get('crop_size', '640').strip())
        offset_range = int(config['Crop settings'].get('random_offset_range', '600').strip())
        prefix = config['Crop settings'].get('filename_prefix', '').strip()
    except ValueError:
        print('Error: Invalid crop settings specified in settings_crop.ini')
def ask_yes_no(text):
    result = messagebox.askyesno("Confirmation", text)
    return result
def select_file(selected_file_index, index, root):
    selected_file_index.set(index + 1)
    root.destroy()
def scan_default_folders():
    # scan default folders
    file_type = ["excel (watchers)", "excel (manual)"]
    video_folder_path = ""
    annotation_file_path = ""
    if scan_folders == "1":
        if not os.path.exists("videos/"):
            os.makedirs("videos/")
        if not os.path.exists("excel/"):
            os.makedirs("excel/")
        # Detect video files
        scan_video_files = [f for f in os.listdir('videos') if f.endswith('.mp4')]
        if scan_video_files:
            response = ask_yes_no(f"Video files detected in the default folder. Do you want to continue?")
            # response = input(f"Video files detected in the default folder. Do you want to continue? (y/n): ")
            # if response.lower() == 'y':
            if response:
                video_folder_path = 'videos'
        if crop_mode == 1 or crop_mode == 2:
            # Detect Excel files
            scan_excel_files = [f for f in os.listdir('excel') if f.endswith('.xlsx') or f.endswith('.xls')]
            if scan_excel_files:
                response = ask_yes_no(f"Excel files detected in the default folder. Do you want to continue?")
                # response = input(f"Excel files detected in the default folder. Do you want to continue? (y/n): ")
                # if response.lower() == 'y':
                if response:

                    # create the tkinter window
                    root = tk.Tk()
                    root.title("Select file")
                    root.wm_attributes("-topmost", 1)
                    label_frame = tk.Frame(root)
                    label_frame.pack(side=tk.TOP, expand=True, fill=tk.BOTH, padx=20)
                    prompt_label = tk.Label(root,
                                            text=f"Please select the {file_type[(crop_mode - 1)]} file you\nwant to use as the source of visit times.")
                    prompt_label.pack()
                    label = tk.Label(root, text="Excel files in the folder:")
                    label.pack()
                    outer_frame = tk.Frame(root)
                    outer_frame.pack(side=tk.TOP, expand=True, fill=tk.BOTH, padx=20, pady=20)
                    for i, f in enumerate(scan_excel_files):
                        button = tk.Button(outer_frame, text=f"{i + 1}. {f}", width=30,
                                           command=lambda i=i: select_file(selected_file_index, i, root))
                        button.pack(pady=0)
                    selected_file_index = tk.IntVar()

                    # set the window position to the center of the screen
                    root.update()
                    screen_width = root.winfo_screenwidth()
                    screen_height = root.winfo_screenheight()
                    window_width = root.winfo_reqwidth()
                    window_height = root.winfo_reqheight()
                    x_pos = int((screen_width - window_width) / 2)
                    y_pos = int((screen_height - window_height) / 2)
                    root.geometry(f"{window_width}x{window_height}+{x_pos}+{y_pos}")
                    root.mainloop()
                    selection = selected_file_index.get()

                    # print('Excel files in the folder:')
                    # for i, f in enumerate(scan_excel_files):
                    # print(f'{i + 1}. {f}')
                    # selection = int(input(f'Enter the number of the {file_type[(crop_mode-1)]} file you want to use: '))

                    # Assign the path of the selected file to a variable
                    if selection > 0 and selection <= len(scan_excel_files):
                        annotation_file_path = os.path.join(("excel/"), scan_excel_files[selection - 1])
                        print(f'Selected file: {annotation_file_path}')
                    else:
                        print('Invalid selection')
    return video_folder_path, annotation_file_path
def reload_points_of_interest():
    global points_of_interest_entry
    points_of_interest_entry = []
    for e in range(len(video_filepaths)):
        points_of_interest_entry.append([])
def get_excel_path(check):
    global annotation_file_path
    #Set path to Excel file manually
    file_type = ["excel (watchers)", "excel (manual)"]
    if crop_mode == 1 or crop_mode == 2:
        if not os.path.isfile(annotation_file_path) or check == 0:
            annotation_file_path = filedialog.askopenfilename(title=f"Select the path to the {file_type[(crop_mode-1)]} file", multiple=False, initialdir=os.path.dirname(os.path.abspath(__file__)), filetypes=[("Excel Files", "*.xlsx"), ("Excel Files", "*.xls")])
def get_video_folder(check):
    global video_folder_path
    global points_of_interest_entry
    global scaned_folders
    global tree_allow
    global loaded
    loaded = 0
    # set path to folder containing mp4 files
    original_video_folder_path = video_folder_path
    if not os.path.isdir(video_folder_path) or check == 0:
        video_folder_path = filedialog.askdirectory(title="Select the video folder", initialdir=os.path.dirname(os.path.abspath(__file__)))
        if video_folder_path == "" and not original_video_folder_path == "":
            video_folder_path = original_video_folder_path
        if (check == 0 and not video_folder_path == original_video_folder_path) or check == 1:
            scan_video_files = [f for f in os.listdir(video_folder_path) if f.endswith('.mp4')]
            parent = os.path.dirname(video_folder_path)
            scaned_folders = [f for f in os.listdir(video_folder_path) if os.path.isdir(os.path.join(video_folder_path, f))]
            if not scan_video_files and scaned_folders:
                scan_child_video_files = [f for f in os.listdir(os.path.join(video_folder_path, scaned_folders[0])) if f.endswith('.mp4')]
                if scan_child_video_files:
                    video_folder_path = os.path.join(video_folder_path, scaned_folders[0])
                    tree_allow = 1
            else:
                tree_allow = 0
            if not check == 1:
                load_videos()
                reload_points_of_interest()
                ICCS_window.destroy()
                load_video_frames()
                open_ICCS_window()
            print(video_folder_path)
def switch_folder(which):
    global scaned_folders
    global video_folder_path
    global loaded
    loaded = 0
    index = scaned_folders.index(os.path.basename(os.path.normpath(video_folder_path)))
    if index > 0 and which == "left":
        video_folder_path = os.path.join(os.path.dirname(video_folder_path), scaned_folders[index-1])
        load_videos()
        reload_points_of_interest()
        ICCS_window.destroy()
        load_video_frames()
        open_ICCS_window()
    if (index + 1) < len(scaned_folders) and which == "right":
        video_folder_path = os.path.join(os.path.dirname(video_folder_path), scaned_folders[index+1])
        load_videos()
        reload_points_of_interest()
        ICCS_window.destroy()
        load_video_frames()
        open_ICCS_window()
def load_videos():
    global video_filepaths
    #Load videos
    video_filepaths = []
    video_filepaths = [os.path.join(video_folder_path, f) for f in os.listdir(video_folder_path) if f.endswith('.mp4')]
    #for filename in os.listdir(video_folder_path):
        #file_path = os.path.join(video_folder_path, filename)
        #if os.path.isfile(file_path):
            #video_filepaths.append(file_path)
def get_text_from_video(video_filepath, start_or_end):
    global x_coordinate
    global y_coordinate
    global width
    global height
    global config
    global cap
    print(video_filepath)
    # Read settings from settings_crop.ini
    config.read('settings_crop.ini', encoding='utf-8')
    try:
        x_coordinate = int(config['OCR settings'].get('x_coordinate', '0').strip())
        y_coordinate = int(config['OCR settings'].get('y_coordinate', '0').strip())
        width = int(config['OCR settings'].get('width', '500').strip())
        height = int(config['OCR settings'].get('height', '40').strip())
    except ValueError:
        # Handle cases where conversion to integer fails
        print('Error: Invalid integer value found in settings_crop.ini')
    cap = cv2.VideoCapture(video_filepath)
    text_roi = (x_coordinate, y_coordinate, width, height)  # x, y, width, height
    if start_or_end == "end":
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        second_to_last_frame_idx = total_frames - 5
        cap.set(cv2.CAP_PROP_POS_FRAMES, second_to_last_frame_idx)
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 24)
    ret, frame = cap.read()
    if ret:
        # Crop the image and pre-process it
        height, width, channels = frame.shape
        x, y, w, h = text_roi
        text_frame = frame[y:y + h, x:x + w]
        HSV_img = cv2.cvtColor(text_frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(HSV_img)
        v = cv2.GaussianBlur(v, (1, 1), 0)
        thresh = cv2.threshold(v, 220, 255, cv2.THRESH_BINARY_INV)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(1, 1))
        thresh = cv2.dilate(thresh, kernel)
        # text recognition
        OCR_text = pytesseract.image_to_string(thresh)

        # debug OCR file creation
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, "OCR images")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        OCR_file_name = ''.join([os.path.basename(video_filepath)[:-4], "_", start_or_end, ".png"])
        OCR_file_path = os.path.join(output_dir, OCR_file_name)
        cv2.imwrite(OCR_file_path, thresh)
    else:
        OCR_text = "none"
    cap.release()
    return OCR_text, frame
def submit_time(input_field):
    global dialog
    global sec_OCR
    global root
    text = input_field.get()
    if not text.isdigit() or len(text) != 2 or int(text) > 59:
        # execute code here for when text is not in "SS" format
        print(
            "Error: OCR detected text does not follow the expected format. Manual input is not in the correct format. The value will be set to an arbitrary 00.")
        text = '00'
    else:
        print("Error: OCR detected text does not follow the expected format. Resolved manually.")
    sec_OCR = text
    root.destroy()
    dialog.destroy()
    open_ICCS_window()
    root.withdraw()

def process_OCR_text(detected_text, frame):
    global cap
    global sec_OCR
    if "\n" in detected_text:
        detected_text = detected_text.replace("\n", "")
    if not len(detected_text) <= 23:
        detected_text = detected_text.rstrip()
        while not detected_text[-1].isdigit():
            detected_text = detected_text[:-1]
    correct_format = r"(0[0-9]|[1-5][0-9]):(0[0-9]|[1-5][0-9]):(0[0-9]|[1-5][0-9])"
    if re.match(correct_format, detected_text[-8:]):
        print(' '.join(["Flow:", "Text detection successful -", detected_text[-8:]]))
        return_time = detected_text[-2:]
    else:
        print(' '.join(["Flow:", "Text detection failed -", detected_text]))
        global root
        global dialog
        global sec_OCR

        # Create window with video frame
        cv2.namedWindow('Frame')
        cv2.imshow('Frame', frame)
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        img_frame_pos_x, img_frame_pos_y, img_frame_width, img_frame_height = cv2.getWindowImageRect('Frame')
        cv2.moveWindow("Frame", int((screen_width // 2) - (img_frame_width // 2)), 0)
        img_frame_pos_x, img_frame_pos_y, img_frame_width, img_frame_height = cv2.getWindowImageRect('Frame')


        # Create the dialog window
        dialog = tk.Toplevel(root)
        dialog.wm_attributes("-topmost", 1)
        dialog.title("Time Input")
        dialog_width = img_frame_width
        dialog_height = dialog.winfo_reqheight()
        dialog_pos_x = int((screen_width // 2) - (img_frame_width // 2))
        dialog_pos_y = max((img_frame_pos_y + img_frame_height),(screen_height-dialog_height))
        dialog.geometry(f"{dialog_width}x{dialog_height}+{dialog_pos_x}+{dialog_pos_y}")


        # Add label
        text_field = tk.Text(dialog, height=2, width=120, font=("Arial", 10))
        text_field.insert(tk.END, "The OCR detection apparently failed.\nEnter the last two digits of the security camera watermark (number of seconds).\nThis will ensure cropping will happen at the right times")
        text_field.configure(state="disabled", highlightthickness=1, highlightbackground="white",
                             background="white", relief="flat")
        text_field.tag_configure("center", justify="center")
        text_field.tag_add("center", "1.0", "end")
        text_field.pack(side=tk.TOP, padx=(0, 0))
        label = tk.Label(dialog, text="Enter text", font=("Arial", 10))
        label.pack(pady=2)

        # Add input field
        j = 0
        input_field = tk.Entry(dialog, font=("Arial", 10), width=4)
        input_field.pack(pady=2)
        input_field.bind("<Return>", lambda j=j: submit_time(input_field))

        # Add submit button
        submit_button = tk.Button(dialog, text="Submit", font=("Arial", 10), command=lambda j=j: submit_time(input_field))
        submit_button.pack(pady=2)

        # Focus on the input field
        dialog.lift()
        input_field.focus_set()
        dialog.mainloop()
        return_time = sec_OCR
        cv2.destroyAllWindows()
    return return_time
# define function to get start and end times for each video file
def get_video_start_end_times(video_filepath):
    video_filename = os.path.basename(video_filepath)
    print(' '.join(["Flow:", "Processing video file -" , video_filepath]))
    # get start time
    parts = video_filename[:-4].split("_")
    if len(parts) == 6:
        start_time_minutes = "_".join([parts[3], parts[4], parts[5]])
        print(' '.join(["Flow: Video name format with prefixes detected. Extracted the time values -", start_time_minutes]))
    else:
        print("Error: Some video file names have an unsupported format. Expected format is CO_LO1_SPPSPP1_YYYYMMDD_HH_MM. Script assumes format YYYYMMDD_HH_MM.")
        start_time_minutes = video_filename[:-4]
    #start_time_minutes = video_filename[:-4]
    text, frame = get_text_from_video(video_filepath, "start")
    start_time_seconds = process_OCR_text(text, frame)
    start_time_str = '_'.join([start_time_minutes, start_time_seconds])
    start_time = pd.to_datetime(start_time_str, format='%Y%m%d_%H_%M_%S')

    # get end time
    text, frame = get_text_from_video(video_filepath, "end")
    end_time_seconds = process_OCR_text(text, frame)
    end_time_str = pd.to_datetime('_'.join([start_time_minutes, end_time_seconds]), format='%Y%m%d_%H_%M_%S')
    end_time = end_time_str + pd.Timedelta(minutes=15)
    return start_time, end_time
def evaluate_string_formula(cell):
    if isinstance(cell, (int, float)):
        # If the cell contains a number, return the value as is
        return cell
    elif cell.startswith('='):
        # If the cell contains an Excel formula, use openpyxl to evaluate it
        wb = openpyxl.Workbook()
        ws = wb.active
        ws['A1'].value = cell
        value = ws['A1'].value
        wb.close()
        return value
    else:
        # If the cell contains text, return the text as is
        return cell
def load_excel_table(file_path):
    # Define the columns to extract
    cols = [0, 1, 2, 3, 4, 5, 15, 18, 19, 20, 21]
    # Read the Excel file, skipping the first two rows
    df = pd.read_excel(file_path, usecols=cols, skiprows=2, header=None, converters={0: evaluate_string_formula, 1: evaluate_string_formula, 2: evaluate_string_formula, 3: evaluate_string_formula, 4: evaluate_string_formula, 18: evaluate_string_formula})
    print(df)
    filtered_df = df[df.iloc[:, 6] == 1]
    # Convert the month abbreviations in column 2 to month numbers
    months = {'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04', 'May': '05', 'Jun': '06',
              'Jul': '07', 'Aug': '08', 'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'}
    filtered_df.iloc[:, 1] = filtered_df.iloc[:, 1].replace(months)
    filtered_df.iloc[:, 1] = filtered_df.iloc[:, 1].astype(int)
    filtered_df = filtered_df.copy()
    for i in range(5):
        j = i + 1
        filtered_df.iloc[:, j] = filtered_df.iloc[:, j].astype(int).apply(lambda x: f'{x:02}')
    filtered_df.loc[:, 11] = filtered_df.iloc[:, 0:6].apply(lambda x: f"{x[0]}{x[1]}{x[2]}_{x[3]}_{x[4]}_{x[5]}",
                                                           axis=1)
    filtered_df.iloc[:, 0] = filtered_df.iloc[:, 0].astype(int)
    print(filtered_df)
    filtered_data = filtered_df.iloc[:, [7, 11]].values.tolist()

    annotation_data_array = filtered_data
    if not os.path.exists("resources/exc/"):
        # create directory
        os.makedirs("resources/exc/")
    filtered_df.to_excel("resources/exc/output_filtered_crop.xlsx", index=False)
    return annotation_data_array
def load_csv(file_path):
    # Define the columns to extract
    cols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # Read the Excel file, skipping the first two rows - follow the custom format
    filtered_df = pd.read_excel(file_path, usecols=cols, skiprows=2, header=None,
                       converters={0: evaluate_string_formula, 1: evaluate_string_formula, 2: evaluate_string_formula,
                                   3: evaluate_string_formula, 4: evaluate_string_formula, 6: evaluate_string_formula})
    print(filtered_df)
    # Convert the month abbreviations in column 2(1) to month numbers
    months = {'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04', 'May': '05', 'Jun': '06',
              'Jul': '07', 'Aug': '08', 'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'}
    filtered_df.iloc[:, 1] = filtered_df.iloc[:, 1].replace(months)
    filtered_df.iloc[:, 1] = filtered_df.iloc[:, 1].astype(int)
    filtered_df = filtered_df.copy()
    for i in range(5):
        j = i + 1
        filtered_df.iloc[:, j] = filtered_df.iloc[:, j].astype(int).apply(lambda x: f'{x:02}')
    filtered_df.loc[:, 10] = filtered_df.iloc[:, 0:6].apply(lambda x: f"{x[0]}{x[1]}{x[2]}_{x[3]}_{x[4]}_{x[5]}",
                                                            axis=1)
    filtered_df.iloc[:, 0] = filtered_df.iloc[:, 0].astype(int)
    print(filtered_df)
    #convert to list
    filtered_data = filtered_df.iloc[:, [6, 10]].values.tolist()

    annotation_data_array = filtered_data
    if not os.path.exists("resources/exc/"):
        # create directory
        os.makedirs("resources/exc/")
    filtered_df.to_excel("resources/exc/output_filtered_crop.xlsx", index=False)
    return annotation_data_array
def capture_crop(frame, point):
    global cap
    global fps
    global frame_number_start
    global visit_duration
    x, y = point
    # Add a random offset to the coordinates, but ensure they remain within the image bounds
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    x_offset = random.randint(-(offset_range), offset_range)
    y_offset = random.randint(-(offset_range), offset_range)
    x1 = max(0, min(((x - crop_size // 2) + x_offset), frame_width - crop_size))
    y1 = max(0, min(((y - crop_size // 2) + y_offset), frame_height - crop_size))
    x2 = max(crop_size, min(((x + crop_size // 2) + x_offset), frame_width))
    y2 = max(crop_size, min(((y + crop_size // 2) + y_offset), frame_height))
    # crop the image
    crop = frame[y1:y2, x1:x2]
    # convert to correct color space
    crop_img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    if crop_img.shape[2] == 3:
        crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
    return crop_img, x1, y1, x2, y2
def generate_frames(frame, success, tag, index):
    global points_of_interest_entry
    global cap
    global fps
    global frame_number_start
    global visit_duration
    species = tag[-27:-19]
    timestamp = tag[-18:-4]
    crop_counter = 1
    # Loop through the video and crop yimages every 30th frame
    frame_count = 0
    while success:
        # Crop images every 30th frame
        if (frame_count % frame_skip == 0):
            for i, point in enumerate(points_of_interest_entry[index]):
                if cropped_frames == 1:
                    crop_img, x1, y1, x2, y2 = capture_crop(frame, point)
                    # save file
                    cv2.imwrite(f"./{output_folder}/{prefix}{species}_{timestamp}_{frame_number_start + frame_count}_{crop_counter}_{x1},{y1}_{x2},{y2}.jpg", crop_img)
            if whole_frame == 1:
                cv2.imwrite(f"./{output_folder}/whole frames/{prefix}{species}_{timestamp}_{frame_number_start + frame_count}_{crop_counter}_whole.jpg", frame)
            crop_counter += 1

        if randomize == 1:
            if (frame_skip - frame_count == 1):
                frame_count += 1
            else:
                frame_count += random.randint(1, max((frame_skip - frame_count), 2))
        else:
            frame_count += frame_skip
        # Read the next frame
        frame_to_read = frame_number_start + frame_count
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_to_read)
        success, frame = cap.read()
        if not frame_count <= (visit_duration * fps):
            # Release the video capture object and close all windows
            cap.release()
            cv2.destroyAllWindows()
            break
def get_video_data(video_filepaths):
    # loop through time annotations and open corresponding video file
    # extract video data beforehand to save processing time
    video_data = []
    i: int
    for i, filepath in enumerate(video_filepaths):
        if filepath.endswith('.mp4'):
            video_start_time, video_end_time = get_video_start_end_times(video_filepaths[i])
            video_data_entry = [video_filepaths[i], video_start_time, video_end_time]
            video_data.append(video_data_entry)
    return video_data
# Define the mouse callback function to record the point of interest
def update_entries(index, original_points):
    global points_of_interest_entry
    global modified_frames
    for each in range(index+1, len(points_of_interest_entry)):
        if len(points_of_interest_entry[each]) == 0:
            points_of_interest_entry[each] = points_of_interest_entry[index].copy()
        else:
            if points_of_interest_entry[each] == original_points:
                points_of_interest_entry[each] = points_of_interest_entry[index].copy()
            else:
                break
    first = 1
    for each in range(index, len(video_filepaths)):
        first = first - 1
        if first >= 0 and (index == 0 or not points_of_interest_entry[max(index - 1, 0)] == points_of_interest_entry[index]):
            modified_frames.append(each)
        update_button_image(frames[each].copy(), (max(each, 0) // 6), (each - ((max(each, 0) // 6) * 6)), first)
def get_mouse_position(event, x, y, flags, mode, i, j):
    global points_of_interest_entry
    global modified_frames
    if event == cv2.EVENT_LBUTTONUP:
        index = j + ((i) * 6)
        if flags & cv2.EVENT_FLAG_SHIFTKEY:
            print("shift")
            closest_point = None
            closest_distance = float('inf')
            for point in points_of_interest_entry[index]:
                distance = math.sqrt((x - point[0]) ** 2 + (y - point[1]) ** 2)
                if distance < closest_distance:
                    closest_point = point
                    closest_distance = distance
            if closest_distance < 30:
                points_of_interest_entry[index].remove(closest_point)
            print(points_of_interest_entry)
            cv2.destroyAllWindows()
        elif flags & cv2.EVENT_FLAG_CTRLKEY and flags & cv2.EVENT_FLAG_ALTKEY:
            points_of_interest_entry[index] = []
            if index in modified_frames:
                modified_frames.remove(index)
            cv2.destroyAllWindows()
        else:
            if not mode == 0:
                points_of_interest_entry[index].append((x,y))
                # index = j+((i)*6)
                # for each in range(index,len(points_of_interest)-1):
            else:
                points_of_interest_entry.append((x, y))
            print(points_of_interest_entry)
            cv2.destroyAllWindows()
def update_button_image(frame,i,j, first):
    global points_of_interest_entry
    global modified_frames
    global button_images
    global buttons
    index = j+((i)*6)
    frame = frame.copy()
    height, width, channels = frame.shape
    for point in points_of_interest_entry[index]:
        x1, y1 = max(0, point[0] - offset_range), max(0, point[1] - offset_range)
        x2, y2 = min(width, point[0] + offset_range), min(height, point[1] + offset_range)
        cv2.rectangle(frame, (point[0] - 30, point[1] - 30), (point[0] + 30, point[1] + 30), (0, 255, 0), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.line(frame, (point[0], point[1]), (x1, y1), (0, 255, 255), thickness=2, lineType=cv2.LINE_AA)
        cv2.line(frame, (point[0], point[1]), (x1, y2), (0, 255, 255), thickness=2, lineType=cv2.LINE_AA)
        cv2.line(frame, (point[0], point[1]), (x2, y1), (0, 255, 255), thickness=2, lineType=cv2.LINE_AA)
        cv2.line(frame, (point[0], point[1]), (x2, y2), (0, 255, 255), thickness=2, lineType=cv2.LINE_AA)
    if (first >= 0 or index in modified_frames) and ((index == 0 and not len(points_of_interest_entry[0]) == 0) or not points_of_interest_entry[max(index - 1, 0)] == points_of_interest_entry[index]):
        # Define the ROI
        frame = cv2.resize(frame, (276, 156), interpolation=cv2.INTER_AREA)
        height, width, channels = frame.shape
        x, y, w, h = 0, 0, height//5, height//5
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
    pil_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(pil_img, mode='RGB')
    # Resize image to fit button
    pil_img = pil_img.resize((276, 156))
    # Convert PIL image to tkinter image
    img = ImageTk.PhotoImage(pil_img)
    button_images[index] = img
    buttons[i][j].configure(image=button_images[index])
def on_button_click(i, j, button_images):
    global points_of_interest_entry
    index = j+((i)*6)
    mode = 1
    frame_tmp = frames[index].copy()
    # Ask the user if they want to select additional points of interest
    while True:
        original_points = points_of_interest_entry[index].copy()
        frame = frame_tmp.copy()
        #add_point = ask_yes_no()
        #if add_point:
        # Draw a rectangle around the already selected points of interest
        height, width, channels = frame.shape
        for point in points_of_interest_entry[index]:
            x1, y1 = max(0, point[0] - offset_range), max(0, point[1] - offset_range)
            x2, y2 = min(width, point[0] + offset_range), min(height, point[1] + offset_range)
            cv2.rectangle(frame, (point[0] - 30, point[1] - 30), (point[0] + 30, point[1] + 30), (0, 255, 0), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.line(frame, (point[0], point[1]), (x1, y1), (0, 255, 255), thickness=2, lineType=cv2.LINE_AA)
            cv2.line(frame, (point[0], point[1]), (x1, y2), (0, 255, 255), thickness=2, lineType=cv2.LINE_AA)
            cv2.line(frame, (point[0], point[1]), (x2, y1), (0, 255, 255), thickness=2, lineType=cv2.LINE_AA)
            cv2.line(frame, (point[0], point[1]), (x2, y2), (0, 255, 255), thickness=2, lineType=cv2.LINE_AA)

        # Display the image with the rectangles marking the already selected points of interest
        cv2.imshow("Frame", frame)
        cv2.setWindowProperty("Frame", cv2.WND_PROP_TOPMOST, 1)
        screen_width, screen_height = cv2.getWindowImageRect("Frame")[2:]
        window_width, window_height = int(1 * screen_width), int(1 * screen_height)
        cv2.resizeWindow("Frame", window_width, window_height)
        cv2.moveWindow("Frame", int((screen_width // 2) - (window_width // 2)), 0)

        # Prompt the user to click on the next point of interest
        cv2.setMouseCallback("Frame", lambda event, x, y, flags, mode: get_mouse_position(event, x, y, flags, mode, i, j), mode)
        key = cv2.waitKey(0)
        if key == 27 or key == 13:  # Check if the Esc key was pressed
            cv2.destroyAllWindows()
            break
        #update_button_image(frame_tmp.copy(), i, j, 1)
        update_entries(index, original_points)
        # else:
        #     update_entries(index,original_points)
        #     break
def load_video_frames():
    # Loop through each file in folder
    global frames
    global points_of_interest_entry
    frames = []
    for filename in os.listdir(video_folder_path):
        if filename.endswith(".mp4"): # Modify file extension as needed
            # Use OpenCV or other library to extract first frame of video
            # and add it to the frames list
            cap = cv2.VideoCapture(os.path.join(video_folder_path, filename))
            cap.set(cv2.CAP_PROP_POS_FRAMES, 25)
            ret, frame = cap.read()
            frames.append(frame)
def update_crop_mode(var):
    global crop_mode
    crop_mode = var
def save_progress():
    global auto_processing
    global points_of_interest_entry
    global video_filepaths
    result = ask_yes_no("Do you want to save the settings? This will overwrite any previous saves.")
    if result:
        # Create an in-memory file object
        filepath = os.path.join(video_folder_path, 'crop_information.pkl')
        with open(filepath, 'wb') as f:
            # Use the pickle module to write the data to the file
            pickle.dump([auto_processing.get(),points_of_interest_entry,video_filepaths], f)
def load_progress():
    global auto_processing
    global points_of_interest_entry
    global video_filepaths
    global frames
    global loaded
    global auto
    result = ask_yes_no("Do you want to load settings? This will overwrite any unsaved progress.")
    if result:
        # Create an in-memory file object
        filepath = os.path.join(video_folder_path, 'crop_information.pkl')
        if os.path.isfile(filepath):
            ICCS_window.destroy()
            points_of_interest_entry = []
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            print(data)
            auto = data[0]
            points_of_interest_entry = data[1].copy()
            video_filepaths_new = data[2].copy()
            set1 = set(video_filepaths_new)
            set2 = set(video_filepaths)
            if not set1 == set2:
                messagebox.showinfo("Discrepancy detected",
                                    "The contents of the video folder have changed since the save has been made. Cannot load the progress. Please start over.")
                reload_points_of_interest()
            else:
                video_filepaths = []
                video_filepaths = video_filepaths_new.copy()
            load_videos()
            load_video_frames()
            loaded = 1
            open_ICCS_window()
        else:
            messagebox.showinfo("No save detected",
                                "There are no save files in the current directory.")
def open_ICCS_window():
    # Create tkinter window
    global ICCS_window
    global tree_allow
    global scaned_folders
    global video_folder_path
    global root
    root = tk.Tk()
    ICCS_window = root
    root.wm_attributes("-topmost", 1)
    j=0
    root.title("Insect Communities Crop Suite")
    # Create frame for the rest
    outer_frame = tk.Frame(root)
    outer_frame.pack(side=tk.TOP, expand=True, fill=tk.BOTH)

    toolbar = tk.Frame(outer_frame)
    toolbar.pack(side=tk.TOP, fill=tk.BOTH)

    left_arrow = Image.open("resources/img/la.png")
    pil_img = left_arrow.resize((50, 50))
    pil_img.save("resources/img/la.png")
    left_arrow = ImageTk.PhotoImage(file="resources/img/la.png")
    if tree_allow == 1 and not scaned_folders.index(os.path.basename(os.path.normpath(video_folder_path))) == 0:
        btn_state = "normal"
    else:
        btn_state = "disabled"
    left_button = tk.Button(toolbar, image=left_arrow, compound=tk.LEFT, text="Previous folder", padx=10, pady=5, height=48, width=200, state=btn_state, command=lambda j=j: switch_folder("left"))
    left_button.pack(side=tk.LEFT)

    menu_icon = Image.open("resources/img/mn.png")
    pil_img = menu_icon.resize((50, 50))
    pil_img.save("resources/img/mn.png")
    menu_icon = ImageTk.PhotoImage(file="resources/img/mn.png")

    menu_button = tk.Button(toolbar, image=menu_icon, compound=tk.LEFT, text="Menu", padx=10, pady=5, height=48, command=lambda j=j: open_menu())
    menu_button.pack(side=tk.LEFT)

    #frame for radio
    radio_frame = tk.Frame(toolbar)
    radio_frame.pack(side=tk.LEFT)

    # create a tkinter variable to hold the selected value
    selected_option = tk.StringVar(value=crop_mode)

    one_icon = Image.open("resources/img/1.png")
    pil_img = one_icon.resize((50, 50))
    pil_img.save("resources/img/1.png")
    one_icon = ImageTk.PhotoImage(file="resources/img/1.png")

    two_icon = Image.open("resources/img/2.png")
    pil_img = two_icon.resize((50, 50))
    pil_img.save("resources/img/2.png")
    two_icon = ImageTk.PhotoImage(file="resources/img/2.png")

    three_icon = Image.open("resources/img/3.png")
    pil_img = three_icon.resize((50, 50))
    pil_img.save("resources/img/3.png")
    three_icon = ImageTk.PhotoImage(file="resources/img/3.png")

    # create the radio buttons and group them together
    rb1 = tk.Radiobutton(radio_frame, text="", image=one_icon, variable=selected_option, value=1, indicatoron=False, height=56, width=116, font=("Arial", 17), command=lambda j_=j: update_crop_mode(1))
    rb2 = tk.Radiobutton(radio_frame, text="", image=two_icon, variable=selected_option, value=2, indicatoron=False, height=56, width=116, font=("Arial", 17), command=lambda j=j: update_crop_mode(2))
    rb3 = tk.Radiobutton(radio_frame, text="", image=three_icon, variable=selected_option, value=3, indicatoron=False, height=56, width=116, font=("Arial", 17), command=lambda j=j: update_crop_mode(3))

    # arrange the radio buttons in a horizontal layout using the grid geometry manager
    rb1.grid(row=0, column=0)
    rb2.grid(row=0, column=1)
    rb3.grid(row=0, column=2)

    right_arrow = Image.open("resources/img/ra.png")
    pil_img = right_arrow.resize((50, 50))
    pil_img.save("resources/img/ra.png")
    right_arrow = ImageTk.PhotoImage(file="resources/img/ra.png")
    if tree_allow == 1 and not (scaned_folders.index(os.path.basename(os.path.normpath(video_folder_path)))+1) == len(scaned_folders):
        btn_state1 = "normal"
    else:
        btn_state1 = "disabled"
    print(tree_allow)
    print(btn_state1)
    right_button = tk.Button(toolbar, image=right_arrow, compound=tk.RIGHT, text="Next folder", padx=10, pady=5, height=48, width=200, state=btn_state1, command=lambda j=j: switch_folder("right"))
    right_button.pack(side=tk.RIGHT)

    on_image = tk.PhotoImage(width=116, height=57)
    off_image = tk.PhotoImage(width=116, height=57)
    on_image.put(("green",), to=(0, 0, 56,56))
    off_image.put(("red",), to=(57, 0, 115, 56))
    global auto_processing
    auto_processing = tk.IntVar(value=0)
    auto_processing.set(0)
    cb1 = tk.Checkbutton(toolbar, image=off_image, selectimage=on_image, indicatoron=False, onvalue=1, offvalue=0, variable=auto_processing)
    cb1.pack(side=tk.LEFT, padx=(0,0), pady=0)

    gears_icon = Image.open("resources/img/au.png")
    pil_img = gears_icon.resize((50, 50))
    pil_img.save("resources/img/au.png")
    gears_icon = ImageTk.PhotoImage(file="resources/img/au.png")

    auto_button = tk.Button(toolbar, image=gears_icon, compound=tk.LEFT, text="\t\tAutomatic evaluation\t\t", padx=10, pady=5, height=48, command=lambda j=j : auto_processing.set(1 - auto_processing.get()))
    auto_button.pack(side=tk.LEFT)

    fl_icon = Image.open("resources/img/fl.png")
    pil_img = fl_icon.resize((50, 50))
    pil_img.save("resources/img/fl.png")
    fl_icon = ImageTk.PhotoImage(file="resources/img/fl.png")

    fl_button = tk.Button(toolbar, image=fl_icon, compound=tk.LEFT, text="Select video folder", padx=10, pady=5, height=48, command=lambda j=j: get_video_folder(0))
    fl_button.pack(side=tk.LEFT)

    et_icon = Image.open("resources/img/et.png")
    pil_img = et_icon.resize((50, 50))
    pil_img.save("resources/img/et.png")
    et_icon = ImageTk.PhotoImage(file="resources/img/et.png")

    et_button = tk.Button(toolbar, image=et_icon, compound=tk.LEFT, text="Select Excel table", padx=10, pady=5, height=48, command=lambda j=j: get_excel_path(0))
    et_button.pack(side=tk.LEFT)

    ocr_icon = Image.open("resources/img/ocr.png")
    pil_img = ocr_icon.resize((50, 50))
    pil_img.save("resources/img/ocr.png")
    ocr_icon = ImageTk.PhotoImage(file="resources/img/ocr.png")

    ocr_button = tk.Button(toolbar, image=ocr_icon, compound=tk.LEFT, text="OCR", padx=10, pady=5, height=48, width=100)
    ocr_button.pack(side=tk.LEFT)

    # Create a canvas to hold the buttons
    canvas = tk.Canvas(outer_frame)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # Create a scrollbar for the canvas
    scrollbar = tk.Scrollbar(outer_frame, orient=tk.VERTICAL, command=canvas.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    # Configure the canvas to use the scrollbar
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

    #Create target frame for the rest
    target_frame = tk.Frame(root)
    target_frame.pack(side=tk.TOP)

    global button_images
    global buttons
    button_images = []
    buttons = []
    rows = math.ceil(len(video_filepaths) / 6)
    per_row = math.ceil(len(video_filepaths) // rows)
    label_frame = tk.Frame(target_frame, width=50)
    label_frame.pack(side=tk.LEFT)
    for i in range(rows):
        hour_1st = ((i) * 6)
        if (((i) * 6)+6) + 1 <= len(video_filepaths):
            hour_2nd = max(((i)*6)+6,len(video_filepaths)-1)
        else:
            hour_2nd = (len(video_filepaths) % 6)-1
        text_label = tk.Label(label_frame, text=f"{os.path.basename(video_filepaths[hour_1st])[-9:-7]}-{os.path.basename(video_filepaths[hour_2nd])[-9:-7]}", font=("Arial", 15), background=root.cget('bg'))
        text_label.pack(side=tk.TOP, padx=30, pady=67)
    for i in range(rows):
        # Loop through first 24 frames and create buttons with images
        button_frame = tk.Frame(target_frame)
        button_frame.pack(side=tk.TOP)
        row = []
        for j in range(6):
            if (j + ((i) * 6)) < len(video_filepaths):
                # Convert frame to PIL image
                pil_img = cv2.cvtColor(frames[j+((i)*6)], cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(pil_img, mode='RGB')
                # Resize image to fit button
                pil_img = pil_img.resize((276, 156))
                # Convert PIL image to tkinter image
                tk_img = ImageTk.PhotoImage(pil_img)
                button_images.append(tk_img)
                # Create button with image and add to button frame
                button = tk.Button(button_frame, image=button_images[j+((i)*6)], command=lambda i=i, j=j: on_button_click(i, j, button_images))
                button.grid(row=i, column=j, sticky="w")
                row.append(button)
            else:
                # Create a dummy frame to fill in the grid
                new_img = Image.new("RGBA", (276, 156), (0, 0, 0, 0))
                new_img = ImageTk.PhotoImage(new_img)
                dummy_frame = tk.Button(button_frame, image=new_img, foreground="white", state="disabled")
                dummy_frame.grid(row=i, column=j, sticky='w')
                row.append(dummy_frame)
        buttons.append(row)

    fl_button = tk.Button(root, image=fl_icon, compound=tk.LEFT, text="", padx=10, pady=5, height=58, command=lambda j=j: os.startfile(video_folder_path))
    fl_button.pack(side=tk.LEFT)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, annotation_file_path)
    et_button = tk.Button(root, image=et_icon, compound=tk.LEFT, text="", padx=10, pady=5, height=58, command=lambda j=j: os.startfile(file_path))
    et_button.pack(side=tk.RIGHT)

    bottom_toolbar = tk.Frame(root, pady=5)
    bottom_toolbar.pack(side=tk.TOP, fill=tk.BOTH)

    save_icon = Image.open("resources/img/sv.png")
    pil_img = save_icon.resize((50, 50))
    pil_img.save("resources/img/sv_1.png")
    save_icon = ImageTk.PhotoImage(file="resources/img/sv_1.png")
    # create a Button widget with the save icon as its image
    save_button = tk.Button(bottom_toolbar, text="Save", image=save_icon, compound=tk.LEFT, padx=10, pady=5, width=300, height=48, command=lambda j=j: save_progress())

    crop_icon = Image.open("resources/img/cr.png")
    pil_img = crop_icon.resize((50, 50))
    pil_img.save("resources/img/cr_1.png")
    crop_icon = ImageTk.PhotoImage(file="resources/img/cr_1.png")
    # create a Button widget with the save icon as its image
    crop_button = tk.Button(bottom_toolbar, text="Crop", image=crop_icon, compound=tk.LEFT, padx=10, pady=5, width=300, height=48, command=lambda j=j: crop_engine())

    sort_icon = Image.open("resources/img/so.png")
    pil_img = sort_icon.resize((50, 50))
    pil_img.save("resources/img/so.png")
    sort_icon = ImageTk.PhotoImage(file="resources/img/so.png")
    # create a Button widget with the save icon as its image
    sort_button = tk.Button(bottom_toolbar, text="Sort", image=sort_icon, compound=tk.LEFT, padx=10, pady=5, width=300,
                            height=48, command=lambda j=j: sort_engine())

    load_icon = Image.open("resources/img/lo.png")
    pil_img = load_icon.resize((50, 50))
    pil_img.save("resources/img/lo.png")
    load_icon = ImageTk.PhotoImage(file="resources/img/lo.png")
    # create a Button widget with the save icon as its image
    load_button = tk.Button(bottom_toolbar, text="Load", image=load_icon, compound=tk.LEFT, padx=10, pady=5, width=300,
                            height=48, command=lambda j=j: load_progress())
    # Specify the column for each button
    save_button.grid(row=0, column=1)
    crop_button.grid(row=0, column=2)
    sort_button.grid(row=0, column=3)
    load_button.grid(row=0, column=4)

    # Add padding between the buttons
    bottom_toolbar.grid_columnconfigure(0, weight=1)
    bottom_toolbar.grid_columnconfigure(5, weight=1)

    # Update the canvas to show the buttons
    canvas.create_window((0, 0), window=target_frame, anchor=tk.NW)
    target_frame.update_idletasks()

    # Configure the canvas to show the entire frame
    canvas.configure(scrollregion=canvas.bbox("all"))

    # Bind mouse wheel event to canvas
    canvas.bind_all("<MouseWheel>", lambda event: canvas.yview_scroll(-1 * (event.delta // 120), "units"))

    # Get the screen width and height
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    #update
    global loaded
    if loaded == 1:
        for each in range(len(video_filepaths)):
            first = -1
            print(first)
            update_button_image(frames[each].copy(), (max(each, 0) // 6), (each - ((max(each, 0) // 6) * 6)), first)
        auto_processing.set(auto)

    # Set the window size to fit the entire screen
    root.geometry(f"{screen_width-20}x{screen_height}+0+0")
    root.state('zoomed')
    root.mainloop()
def initialise():
    global loaded
    global tree_allow
    global modified_frames
    global video_folder_path
    global annotation_file_path
    config_read()
    pytesseract.pytesseract.tesseract_cmd = ocr_tesseract_path
    loaded = 0
    tree_allow = 0
    modified_frames = []
    video_folder_path, annotation_file_path = scan_default_folders()
    get_video_folder(1)
    get_excel_path(1)
    load_videos()
    #Check output folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not os.path.exists(f"./{output_folder}/whole frames/"):
        os.makedirs(f"./{output_folder}/whole frames/")
    reload_points_of_interest()
    load_video_frames()
def crop_engine():
    global points_of_interest_entry
    global video_filepaths
    global cap
    global fps
    global frame_number_start
    global visit_duration
    global crop_mode
    global root
    result = ask_yes_no("Do you want to start the cropping process?")
    if result:
        valid_annotations_array = []
        valid_annotation_data_entry = []
        print(video_filepaths)
        root.withdraw()
        if not crop_mode == 3:
            if crop_mode == 1:
                video_data = get_video_data(video_filepaths)
                annotation_data_array = load_excel_table(annotation_file_path)
            if crop_mode == 2:
                annotation_data_array = load_csv(annotation_file_path)
                video_filepaths_temp = video_filepaths
                video_filepaths = []
                for index, list in enumerate(annotation_data_array):
                    for filepath in video_filepaths_temp:
                        filename = os.path.basename(filepath)  # get just the filename from the full path
                        if annotation_data_array[index][1][:-9] in filename:
                            if datetime.timedelta() <= datetime.datetime.strptime(annotation_data_array[index][1][-8:-3], '%H_%M') - datetime.datetime.strptime(filename[-9:-4], '%H_%M') <= datetime.timedelta(minutes=15):
                                video_filepaths.append(filepath)
                video_data = get_video_data(video_filepaths)
            print(annotation_data_array)
            for index, list in enumerate(annotation_data_array):
                print(' '.join(["Flow: Annotation number:", str(index + 1)]))
                annotation_time = pd.to_datetime(annotation_data_array[index][1], format='%Y%m%d_%H_%M_%S')
                for i, list in enumerate(video_data):
                    if video_data[i][1] <= annotation_time <= video_data[i][2]:
                        for each in range(len(annotation_data_array[index])):
                            valid_annotation_data_entry.append(annotation_data_array[index][each])
                        for each in range(3):
                            valid_annotation_data_entry.append(video_data[i][each])
                        print(valid_annotation_data_entry)
                        valid_annotations_array.append(valid_annotation_data_entry)
                        valid_annotation_data_entry = []
            for index in range(len(valid_annotations_array)):
                print(f"Processing item {index}")
                annotation_time = pd.to_datetime(valid_annotations_array[index][1], format='%Y%m%d_%H_%M_%S')
                annotation_offset = (annotation_time - valid_annotations_array[index][3]).total_seconds()
                cap = cv2.VideoCapture(valid_annotations_array[index][2])
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                visit_duration = (min(((annotation_offset*fps)+(int(valid_annotations_array[index][0])*fps)), total_frames)-(annotation_offset*fps)) // fps
                frame_number_start = int(annotation_offset * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number_start)
                success, frame = cap.read()
                generate_frames(frame, success, os.path.basename(valid_annotations_array[index][2]), video_filepaths.index(valid_annotations_array[index][2]))
        else:
            for i, filepath in enumerate(video_filepaths):
                cap = cv2.VideoCapture(video_filepaths[i])
                fps = cap.get(cv2.CAP_PROP_FPS)
                visit_duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))//fps
                frame_number_start = 2
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number_start)
                success, frame = cap.read()
                generate_frames(frame, success,os.path.basename(video_filepaths[i]),i)
    if root.winfo_exists():
        root.destroy()
    open_ICCS_window()

def sort_engine():
    # Ask user if they want to run sorting script
    run_sorting = ask_yes_no("Do you want to run the sorting script on the generated images?")
    if run_sorting:
        sort_script_path = "sort.py"
        if os.path.exists(sort_script_path):
            #subprocess.run(['python', f'{sort_script_path}'])
            subprocess.call([sys.executable, f'{sort_script_path}', "--subprocess"])
        else:
            print("Error: sorting script not found.")
def open_menu():
    global output_folder
    global scan_folders
    global crop_mode
    global frame_skip
    global randomize
    global whole_frame
    global cropped_frames
    global crop_size
    global offset_range
    global prefix
    global end_values
    # Create the Tkinter window
    window = tk.Tk()
    window.title("Menu")
    window.wm_attributes("-topmost", 1)
    # Create the labels and input fields
    label_text=["Output folder path:", "Scan default folders:", "Filename prefix:", "Default crop mode:", "Frames to skip:", "Randomize interval:", "Export whole frames:", "Export cropped frames:", "Crop size:", "Offset size:"]
    labels = []
    fields = []
    outer_frame = tk.Frame(window, pady=10)
    outer_frame.pack(side=tk.TOP, fill=tk.BOTH)
    for i in range(10):
        label = tk.Label(outer_frame, text=f"{label_text[i]}")
        label.grid(row=i, column=0)
        labels.append(label)

        field = tk.Entry(outer_frame, width=120)
        field.grid(row=i, column=1)
        fields.append(field)

    # Create the save button
    def save_fields():
        global output_folder
        global scan_folders
        global crop_mode
        global frame_skip
        global randomize
        global whole_frame
        global cropped_frames
        global crop_size
        global offset_range
        global prefix
        global end_values
        end_values = []
        for i in range(10):
            end_values.append(fields[i].get())
        output_folder = str(end_values[0])
        scan_folders = str(end_values[1])
        prefix = str(end_values[2])
        crop_mode = int(end_values[3])
        frame_skip = int(end_values[4])
        randomize = int(end_values[5])
        whole_frame = int(end_values[6])
        cropped_frames = int(end_values[7])
        crop_size = int(end_values[8])
        offset_range = int(end_values[9])
        config_write()
        window.destroy()

    save_button = tk.Button(outer_frame, text="Save", command=save_fields)
    save_button.grid(row=12, column=0, columnspan=2)

    # Set initial values for the input fields
    initial_values = [output_folder, scan_folders, prefix, crop_mode, frame_skip, randomize, whole_frame, cropped_frames, crop_size, offset_range]
    for i in range(10):
        fields[i].insert(0, str(initial_values[i]))

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
def config_write():
    global ocr_tesseract_path
    global video_folder_path
    global annotation_file_path
    global output_folder
    global scan_folders
    global crop_mode
    global frame_skip
    global randomize
    global whole_frame
    global cropped_frames
    global crop_size
    global offset_range
    global config
    global prefix
    global end_values
    config = configparser.ConfigParser()
    config.read('settings_crop.ini')
    # Update values in the config file
    config.set('Resource Paths', 'OCR_tesseract_path', ocr_tesseract_path)
    config.set('Resource Paths', 'video_folder_path', video_folder_path)
    config.set('Resource Paths', 'annotation_file_path', annotation_file_path)
    config.set('Resource Paths', 'output_folder', output_folder)
    config.set('Workflow settings', 'Scan_default_folders', scan_folders)
    config.set('Crop settings', 'crop_mode', str(crop_mode))
    config.set('Crop settings', 'crop_interval_frames', str(frame_skip))
    config.set('Crop settings', 'randomize_interval', str(randomize))
    config.set('Crop settings', 'export_whole_frame', str(whole_frame))
    config.set('Crop settings', 'export_crops', str(cropped_frames))
    config.set('Crop settings', 'crop_size', str(crop_size))
    config.set('Crop settings', 'random_offset_range', str(offset_range))
    config.set('Crop settings', 'filename_prefix', str(prefix))

    # Save changes to the config file
    with open('settings_crop.ini', 'w', encoding='utf-8') as configfile:
        config.write(configfile)

initialise()
open_ICCS_window()

