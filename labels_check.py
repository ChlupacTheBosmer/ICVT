
import os
import cv2
import numpy as np
import pybboxes as pbx
from utils import create_dir
from tkinter import filedialog
import numpy
import glob

def yolobbox2bbox(coords):
    x, y, w, h = coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]
    x1, y1 = ((x - (w / 2)) * 640).astype(int), ((y - (h / 2)) * 640).astype(int)
    x2, y2 = ((x + (w / 2)) * 640).astype(int), ((y + (h / 2)) * 640).astype(int)
    box_width, box_height = x2 - x1, y2 - y1
    return np.column_stack((x1, y1, x2, y2, box_width, box_height))

# Specify the folder path
folder_path = filedialog.askdirectory(title="Select the image folder",
                                      initialdir=os.path.dirname(os.path.abspath(__file__)))
frames = 300

# Use os.listdir() to get all image files
image_files = [file.encode('utf-8') for file in os.listdir(folder_path) if file.lower().endswith(('.jpg', '.png'))]

# Iterate through all image files
for file_bytes in image_files:
    # Decode the file path back to string
    filename = file_bytes.decode('utf-8')
    file_path = os.path.join(folder_path, filename)

    # Construct the full path to the corresponding txt file
    txt_path = os.path.join(folder_path, os.path.splitext(filename)[0] + ".txt")
    if not os.path.exists(txt_path):
        continue

    with open(txt_path, "r", encoding="utf-8") as f:  # Specify the encoding
        coords = f.readline().strip().split()[1:]
    coords = np.array([float(coord) for coord in coords]).reshape(-1, 4)
    bbox_coords = yolobbox2bbox(coords)

    # Open the image
    stream = open(file_path, "rb")
    bytes = bytearray(stream.read())
    numpyarray = numpy.asarray(bytes, dtype=numpy.uint8)
    img = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)

    # Check if the image was successfully read
    if img is None:
        print(f"Failed to read image: {file_path}")
        continue

    # Draw rectangles on the image using the bounding box coordinates
    for box in bbox_coords:
        box_left, box_top, box_right, box_bottom, _, _ = box
        cv2.rectangle(img, (box_left, box_top), (box_right, box_bottom), (0, 255, 0), 2)

    if frames > 1:
        # Show the image using cv2.imshow()
        cv2.imshow('image', img)
        cv2.setWindowProperty('image', cv2.WND_PROP_TOPMOST, 1)

        key = cv2.waitKey(0)
        if key == 27 or key == 13:  # Check if the Esc key was pressed
            break

    print(f"File {filename} processed successfully")

# Close all cv2 windows after loop completion
cv2.destroyAllWindows()