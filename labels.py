from PIL import Image
import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog
import cv2
import math
import os
import shutil
import random
import pybboxes as pbx



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
folder_path_img = filedialog.askdirectory(title="Select the IMAGE folder",
                                                    initialdir=os.path.dirname(os.path.abspath(__file__)))
# Specify the folder path
folder_path_lbl = filedialog.askdirectory(title="Select the LABEL folder",
                                                    initialdir=folder_path_img)

#display message box and ask yes or no
answer = messagebox.askyesno("Confirmation", "Do you want to visually control the cropping process? If yes, each image will be displayed for a short while so you can make sure there are no discrepancies. If no, the process will be faster.")

if answer:
    frames = 300
else:
    frames = 1

#display message box and ask yes or no
answer = messagebox.askyesno("Confirmation", "Do you want to specify separate output folder? If yes, you will be asked to select the folder. If no, the output folder will be the same as the input folder.")

if answer:
    out_folder_path_img = filedialog.askdirectory(title="Select the IMAGE output folder",
                                          initialdir=folder_path_img)
    out_folder_path_lbl = filedialog.askdirectory(title="Select the LABEL output folder",
                                                  initialdir=folder_path_lbl)
else:
    out_folder_path_img = folder_path_img
    out_folder_path_lbl = folder_path_lbl



# Specify the paths of the original and destination folders
tmp_folder = "resources/tmp"
create_dir(tmp_folder)

# Iterate through all files in the folder
for filename in os.listdir(folder_path_img):
    # Check if the file is an image file (you can customize this condition as needed)
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # Get the full file path
        file_path = os.path.join(folder_path_img, filename)

        # Construct the full paths to the original and destination files
        original_path = os.path.join(folder_path_img, filename)
        destination_path = os.path.join(tmp_folder, filename)

        # Move the file to the destination folder
        shutil.move(original_path, destination_path)

        # Open the image
        img = Image.open(destination_path)

        # Open the corresponding txt file
        txt_path = f"{folder_path_lbl}/{filename[:-4]}.txt"
        if not os.path.exists(txt_path):
            # Calculate the centre of the random point
            x = random.randint(0, img.size[0])
            y = random.randint(0, img.size[1])

            # Calculate the coordinates of the cropped image
            factor = 320
            x1 = int(max(0, min((x - factor // 2), int(img.size[0]) - factor)))
            y1 = int(max(0, min((y - factor // 2), int(img.size[1]) - factor)))
            x2 = int(max(factor, min((x + factor // 2), int(img.size[0]))))
            y2 = int(max(factor, min((y + factor // 2), int(img.size[1]))))

            box_left, box_top, box_right, box_bottom = x1, y1, x2, y2
            box_width = box_right - box_left
            box_height = box_bottom - box_top
        else:
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

            # Calculate the coordinates of the cropped image
            factor = max(box_width, box_height)
            factor = max(factor, 320)
            x1 = int(max(0, min((x - factor // 2), int(img.size[0]) - factor)))
            y1 = int(max(0, min((y - factor // 2), int(img.size[1]) - factor)))
            x2 = int(max(factor, min((x + factor // 2), int(img.size[0]))))
            y2 = int(max(factor, min((y + factor // 2), int(img.size[1]))))

        # Crop the image
        img_cropped = img.crop((x1, y1, x2, y2))

        # Open the cropped image
        img1 = cv2.imread(f"{tmp_folder}/{filename[:-4]}.jpg")

        # Draw a rectangle on the image using the bounding box coordinates
        cv2.rectangle(img1, (box_left, box_top), (box_right, box_bottom), (0, 255, 0), 2)
        cv2.rectangle(img1, (x1, y1), (x2, y2), (0, 0, 255), 2)

        if frames > 1:
            # Show the image
            cv2.imshow('image', img1)
            cv2.namedWindow('image')
            # Make the window topmost
            cv2.setWindowProperty('image', cv2.WND_PROP_TOPMOST, 1)
            cv2.waitKey(frames)
            cv2.destroyAllWindows()

        # Resize the cropped image to 320x320 pixels
        img_resized = img_cropped.resize((320, 320))

        # Save the new image to a file
        new_img_path = f"{tmp_folder}/{filename[:-4]}_320.jpg"
        img_resized.save(new_img_path)

        # Calculate the new coordinates of the bounding box in the cropped and resized image
        new_box_left = (box_left - x1) * 320 / img_cropped.size[0]
        new_box_top = (box_top - y1) * 320 / img_cropped.size[1]
        new_box_right = (box_right - x1) * 320 / img_cropped.size[0]
        new_box_bottom = (box_bottom - y1) * 320 / img_cropped.size[1]
        new_box_width = new_box_right - new_box_left
        new_box_height = new_box_bottom - new_box_top

        if os.path.exists(txt_path):
            #Convert the coordinates to YOLOv3 format
            b = (new_box_left, new_box_top, new_box_right, new_box_bottom)
            print(b, img_resized.size[0], img_resized.size[1])
            bb = pbx.convert_bbox(b, from_type="voc", to_type="yolo", image_size=(img_resized.size[0], img_resized.size[1]))
            #bb = bbox2yolobbox((img_resized.size[0], img_resized.size[1]), b)

            # Write the new coordinates to a new txt file
            new_txt_path = f"{out_folder_path_lbl}/{filename[:-4]}_320.txt"
            with open(new_txt_path, "w") as f:
                f.write(f"0 {round(bb[0], 6)} {round(bb[1], 6)} {round(bb[2], 6)} {round(bb[3], 6)}")

        img2 = cv2.imread(f"{tmp_folder}/{filename[:-4]}_320.jpg")

        # Draw a rectangle on the image using the bounding box coordinates
        cv2.rectangle(img2, (int(new_box_left), int(new_box_top)), (int(new_box_right), int(new_box_bottom)), (0, 255, 0), 2)

        if frames > 1:
            # Show the image
            cv2.imshow('image', img2)
            cv2.namedWindow('image')
            # Make the window topmost
            cv2.setWindowProperty('image', cv2.WND_PROP_TOPMOST, 1)
            cv2.waitKey(frames)
            cv2.destroyAllWindows()

        # Move the image and txt file to the original folder
        if os.path.exists(txt_path):
            shutil.move(destination_path, os.path.join(folder_path_img, filename))
        shutil.move(new_img_path, os.path.join(out_folder_path_img, f"{filename[:-4]}_320.jpg"))
        print(f"File {filename} processed successfully")