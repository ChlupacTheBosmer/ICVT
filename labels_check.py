from PIL import Image
import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog
import cv2
import math
import os
import shutil

def yolobbox2bbox(x,y,w,h):
    x1, y1 = (x-(w/2))*640, (y-(h/2))*640
    x2, y2 = (x+(w/2))*640, (y+(h/2))*640
    return math.ceil(x1), math.ceil(y1), math.ceil(x2), math.ceil(y2)

def bbox2yolobbox(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return x,y,w,h

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Specify the folder path
folder_path = filedialog.askdirectory(title="Select the image folder",
                                                    initialdir=os.path.dirname(os.path.abspath(__file__)))
frames = 300

# Specify the paths of the original and destination folders
tmp_folder = "resources/tmp"
create_dir(tmp_folder)

# Iterate through all files in the folder
for filename in os.listdir(folder_path):
    # Check if the file is an image file (you can customize this condition as needed)
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # Get the full file path
        file_path = os.path.join(folder_path, filename)

        # Construct the full paths to the original and destination files
        original_path = os.path.join(folder_path, filename)
        destination_path = os.path.join(tmp_folder, filename)

        # Move the file to the destination folder
        #shutil.move(original_path, destination_path)

        # Open the image
        img = Image.open(original_path)

        # Open the corresponding txt file
        txt_path = f"{folder_path}/{filename[:-4]}.txt"
        if not os.path.exists(txt_path):
            continue
        with open(txt_path, "r") as f:
            coords = f.readline().strip().split()[1:]

        # Convert the coordinates to floats and scale them for a 320x320 image
        coords = [float(coord) for coord in coords]
        box_left, box_top, box_right, box_bottom = yolobbox2bbox(*coords)
        box_width = box_right - box_left
        box_height = box_bottom - box_top

        # Calculate the centre of the bounding box in the original image
        x = box_right - box_width/2
        y = box_bottom - box_height/2

        # Open the cropped image
        img1 = cv2.imread(original_path)

        # Draw a rectangle on the image using the bounding box coordinates
        cv2.rectangle(img1, (box_left, box_top), (box_right, box_bottom), (0, 255, 0), 2)

        if frames > 1:
            # Show the image
            cv2.imshow('image', img1)
            cv2.namedWindow('image')
            # Make the window topmost
            cv2.setWindowProperty('image', cv2.WND_PROP_TOPMOST, 1)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # Move the image and txt file to the original folder
        #shutil.move(destination_path, os.path.join(folder_path, filename))
        print(f"File {filename} processed successfully")