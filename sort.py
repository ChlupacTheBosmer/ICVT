import os
import cv2
import numpy as np
import configparser
import pyautogui
from PIL import Image
import tkinter as tk
from tkinter import simpledialog
from tkinter import messagebox
import sys
from tkinter import ttk
from tkinter import filedialog

def ask_yes(text):
    result = messagebox.askyesno("Confirmation", text)
    return result

config = configparser.ConfigParser()
config['Resource Paths'] = {
    'image_folder': 'output',
    'target_folder_left': 'output/empty',
    'target_folder_right': 'output/visitors'
}
config['GUI settings'] = {
    'left_label': 'empty',
    'right_label': 'visitor',
    'window_size': '960'
}
config['Workflow settings'] = {
    'Scan_default_folders': '1',
    'send_left_key': 'a',
    'send_right_key': 'd',
    'quit_key': 'q'
}
# Check if settings_sort.ini exists, and create it with default values if not
if not os.path.exists('settings_sort.ini'):
    with open('settings_sort.ini', 'w', encoding='utf-8') as configfile:
        config.write(configfile)

# Read settings from settings_sort.ini
config.read('settings_sort.ini', encoding='utf-8')

# Get values from the config file
try:
    folder_path = config['Resource Paths'].get('image_folder', 'output').strip()
    left_folder_path = config['Resource Paths'].get('target_folder_left', 'output/empty').strip()
    right_folder_path = config['Resource Paths'].get('target_folder_right', 'output/visitors').strip()
    scan_folders = config['Workflow settings'].get('Scan_default_folders', '0').strip()
except ValueError:
    print('Error: Invalid folder/file path found in settings_sort.ini')
#Get GUI settings from config
try:
    text_left = config['GUI settings'].get('left_label', 'empty').strip().upper()
    text_right = config['GUI settings'].get('right_label', 'visitor').strip().upper()
    window_width = int(config['GUI settings'].get('window_size', '640').strip())
    left_key = config['Workflow settings'].get('send_left_key', 'a').strip()
    right_key = config['Workflow settings'].get('send_right_key', 'd').strip()
    quit_key = config['Workflow settings'].get('quit_key', 'q').strip()
except ValueError:
    print('Error: Invalid crop settings specified in settings_sort.ini')

#scan default folder
if scan_folders == "1":
    if not os.path.exists("output/"):
        os.makedirs("output/")
    if not os.path.exists("output/empty"):
        os.makedirs("output/empty")
    if not os.path.exists("output/visitors"):
        os.makedirs("output/visitors")
    # Detect image files
    scan_image_files = [f for f in os.listdir('output') if f.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]
    if not "--subprocess" in sys.argv:
        if scan_image_files:
             response = ask_yes("Image files detected in the default folder. Do you want to continue?")
             if response:
                 folder_path = 'output'

if not os.path.exists(folder_path):
    os.makedirs(folder_path)
if not os.path.exists(left_folder_path):
    os.makedirs(left_folder_path)
if not os.path.exists(right_folder_path):
    os.makedirs(right_folder_path)

# Get a list of all image files in the folder
image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]

#define sizes
font = cv2.FONT_HERSHEY_SIMPLEX
text_size_l = cv2.getTextSize(text_left, font, 1, 2)[0]
text_size_r = cv2.getTextSize(text_right, font, 1, 2)[0]

# Add the arrows and text to the left and right sides of the canvas
arrow_width = 80
arrow_height = 60
arrow_color_l = (0, 0, 255)
arrow_color_r = (0, 255, 0)

# Loop through each image file and show it to the user
for image_file in image_files:
    # Load the image
    image_path = os.path.join(folder_path, image_file)
    image = cv2.imread(image_path)
    image_height, image_width, channels = image.shape

    # Calculate the aspect ratio of the image
    aspect_ratio = image_width / image_height

    # Set the desired height and width
    new_height = window_width
    new_width = int(new_height * aspect_ratio)

    # Resize the image
    resized_image = cv2.resize(image,(new_width, new_height), Image.LANCZOS)

    # If the new width is less than 960, resize again to get the correct width
    if new_width < new_height:
        resized_image = cv2.resize(resized_image, (new_height, new_height), Image.LANCZOS)

    # If the new width is greater than 960, crop the image to get the correct width
    if new_width > new_height:
        crop_start = int((new_width - new_height) / 2)
        crop_end = crop_start + new_height
        resized_image = resized_image[:, crop_start:crop_end]
    image = resized_image
    image_height, image_width, channels = image.shape

    # Specify the desired width of the window
    adj_window_width = window_width + (arrow_width * 2) + text_size_l[0] + text_size_r[0]
    # Create a black canvas to hold the image and arrows
    canvas = np.zeros((image_height, adj_window_width, 3), dtype=np.uint8)

    # Put the image on the canvas, centered horizontally
    image_x = (adj_window_width - image_width) // 2
    canvas[:, image_x:image_x + image_width, :] = image

    # Add triangle to the left side of the canvas
    triangle_points = [(arrow_width, image_height // 2 - arrow_height), (0, image_height // 2),
                       (arrow_width, image_height // 2 + arrow_height)]
    cv2.drawContours(canvas, [np.array(triangle_points)], 0, arrow_color_l, -1)

    # Add text label to the left side of the canvas
    text_color = (0, 0, 255)
    text_position = (arrow_width + 5, image_height // 2 + text_size_l[1] // 2)
    cv2.putText(canvas, text_left, text_position, font, 1, text_color, 2)

    # Add triangle to the right side of the canvas
    triangle_points = [(adj_window_width - arrow_width, image_height // 2 - arrow_height),
                       (adj_window_width, image_height // 2),
                       (adj_window_width - arrow_width, image_height // 2 + arrow_height)]
    cv2.drawContours(canvas, [np.array(triangle_points)], 0, arrow_color_r, -1)

    # Add text label to the right side of the canvas
    text_color = (0, 255, 0)
    text_position = (adj_window_width - arrow_width - text_size_r[0], image_height // 2 + text_size_r[1] // 2)
    cv2.putText(canvas, text_right, text_position, font, 1, text_color, 2)

    # Show the image and wait for user input
    window_name = f"{os.path.basename(image_path)} - Image"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowTitle(window_name, os.path.basename(image_path))
    cv2.resizeWindow(window_name, adj_window_width, image_height)


    # Calculate window position
    screen_width, screen_height = pyautogui.size()
    x = (screen_width - adj_window_width) // 2
    y = 0
    cv2.moveWindow(window_name, x, y)

    cv2.imshow(window_name, canvas)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

    key = cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Move the image file to the appropriate folder based on user input
    if key == ord(quit_key):
        break
    elif key == ord(left_key):
        os.rename(image_path, os.path.join(left_folder_path, image_file))
    elif key == ord(right_key):
        os.rename(image_path, os.path.join(right_folder_path, image_file))

# Clean up
cv2.destroyAllWindows()