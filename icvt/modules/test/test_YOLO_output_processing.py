from ultralytics import YOLO
import os
import cv2
import shutil

# Load the saved YOLO model from the .pt file
model = YOLO('D:/Program files/anaconda3/envs/YOLO/runs\detect/640_def/weights/best.pt')
#print(torch.cuda.is_available())

# Specify the folder path
#folder_path = filedialog.askdirectory(title="Select the image folder", initialdir=os.path.dirname(os.path.abspath(__file__)))
folder_path = "../../output"

# Get a list of all image files in the folder
image_names = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
image_files = []
for i, filename in enumerate(image_names):
    # Construct the full file path
    file_path = os.path.join(folder_path, filename)
    image_files.append(file_path)
    if i == 200:
        break

if True == False:
    # Make video
    # Output video file path
    output_path = 'output/video.mp4'

    # Frame rate (fps) of the output video
    fps = 5

    # Get a list of all image file names in the directory
    #image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]

    # Get the dimensions of the first image
    image = cv2.imread(image_files[0])
    height, width, channels = image.shape

    # Create the VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can use other codecs as well (e.g., 'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Iterate over each image and write it to the video file
    for image_file in image_files:
        image = cv2.imread(image_file)
        out.write(image)

    # Release the VideoWriter and close the output file
    out.release()

    # Perform inference on the image

results = model(image_files, save=True, imgsz=640, conf=0.25, save_txt=True, max_det=1, stream=True)  # list of Results objects
#results2 = model('output/video.mp4', save=True, imgsz=640, conf=0.25, stream=True, save_txt=True, max_det=1)


# Construct the full paths to the original and destination files
for i, result in enumerate(results):
    print(result.boxes.data[0])
    boxes = result.boxes.data
    original_path = os.path.join(image_files[i])
    empty_path = os.path.join("output/e", os.path.basename(image_files[i]))
    visitor_path = os.path.join("output/v", os.path.basename(image_files[i]))
    if len(result.boxes.data) > 0:
        #print(f"mám jí tu svini: {result.path}")
        # Move the file to the destination folder
        shutil.move(original_path, visitor_path)
        with open(f"{visitor_path[:-4]}.txt", 'w') as file:
            # Write the box_data to the file
            txt = str(result.boxes.xywh[0].tolist())
            file.write(f"0 {txt.replace('[', '').replace(']', '').replace(',', '')}")
    else:
        #print(f"nemám jí tu svini: {result.path}")
        shutil.move(original_path, empty_path)
        with open(f"{empty_path[:-4]}.txt", 'w') as file:
            # Write the box_data to the file
            txt = str(result.boxes.xywh[0].tolist())
            file.write(f"0 {txt.replace('[', '').replace(']', '').replace(',', '')}")

