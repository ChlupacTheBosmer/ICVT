import os
from PIL import Image
from tkinter import filedialog

# set the folder path
folder_path = filedialog.askdirectory(title="Select the image folder",
                                                    initialdir=os.path.dirname(os.path.abspath(__file__)))

# create the new folder for converted images
os.makedirs(os.path.join(folder_path, 'jpgs'), exist_ok=True)

# loop through all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.png'):
        # open the image file
        img = Image.open(os.path.join(folder_path, filename))
        # convert the image to JPG format
        img = img.convert('RGB')
        # save the converted image to the new folder
        img.save(os.path.join(folder_path, 'jpgs', os.path.splitext(filename)[0] + '.jpg'))