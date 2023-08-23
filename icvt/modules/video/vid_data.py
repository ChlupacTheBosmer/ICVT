# This file contains the video data classes
#
# Modules of ICVT
from modules.utility import utils
from modules.vision import vision_AI

# Other modules
import pandas as pd
import cv2
import tkinter as tk
from tkinter import simpledialog
from PIL import Image, ImageTk
from hachoir.metadata import extractMetadata
from hachoir.parser import createParser
import imageio

# Default python modules
import datetime
from datetime import datetime
from datetime import timedelta
import time
import os

class Video_file():

    def __init__(self, filepath, root = None, ocr_roi: tuple = (0, 0, 500, 60), initiate_start_and_end_times: bool = True):

        # Define logger
        self.logger = utils.log_define()

        # Define variables
        self.filepath = filepath
        self.main_window = root
        self.ocr_roi = ocr_roi

        # Determine video origin
        if filepath.endswith(".mp4"):
            self.video_origin = "VT"
        elif filepath.endswith(".avi"):
            self.video_origin = "MS"

        # Chose a different method based on whether the video is a vivotek .mp4 or a milesight .avi
        if self.video_origin == "VT":
            self.cap = cv2.VideoCapture(self.filepath)

        # Get basic video properties
        self.fps = self.get_video_fps()
        self.total_frames = self.get_video_total_frames()

        # Check requirements
        if filepath.endswith(".mp4") or filepath.endswith(".avi"):
            if initiate_start_and_end_times:
                self.start_time, self.end_time = self.get_video_start_end_times()
        else:
            self.logger.error("Invalid file type. Provide path to a valid video file.")

    def get_video_start_end_times(self):

        # Define logger
        self.logger.debug(f"Running function get_video_start_end_times({self.filepath})")
        video_filename = os.path.basename(self.filepath)
        self.logger.debug(' '.join(["Processing video file -", self.filepath]))

        # Get start time from metadata but because the metadata often contain wrong hour number, we will only use the seconds
        start_time_meta, success = self.get_metadata_from_video(self.filepath, "start")

        # If failed to get time from metadata obtain it manually
        if not success:

            # Get the time in seconds manually
            # frame = self.get_video_frame("start")
            _ , frame = self.read_video_frame(24)

            success, time = vision_AI.get_text_with_OCR(frame)
            if not success:
                start_time_seconds, success = self.get_text_manually(frame)
            else:
                start_time_seconds = str(time.second)

        else:
            start_time_seconds = start_time_meta[-2:]

        # Now get the date, hour and minute from the filename
        filename_parts = video_filename[:-4].split("_")
        start_time_minutes = "_".join(
            [filename_parts[len(filename_parts) - 3], filename_parts[len(filename_parts) - 2],
             filename_parts[len(filename_parts) - 1]])  # creates timestamp

        # Construct the string
        start_time_str = '_'.join([start_time_minutes, start_time_seconds])
        start_time = pd.to_datetime(start_time_str, format='%Y%m%d_%H_%M_%S')
        self.start_time = start_time

        # Get end time
        end_time_meta, success = self.get_metadata_from_video(self.filepath, "end")

        # If failed to get time from metadata obtain it manually
        if not success:
            # Get the time in seconds manually
            # frame = self.get_video_frame("end")
            _, frame = self.read_video_frame(self.total_frames - 10)

            success, time = vision_AI.get_text_with_OCR(frame)
            if not success:
                end_time_seconds, success = self.get_text_manually(frame)
            else:
                end_time_seconds = str(time.second)

            # 15 minut duration of the video is assumed and the manually extracted seconds are added
            delta = 15 + (int(end_time_seconds) // 60)
            end_time_seconds = str(int(end_time_seconds) % 60)

            # Construct the string
            end_time_str = '_'.join([start_time_minutes, end_time_seconds])
            end_time = pd.to_datetime(end_time_str, format='%Y%m%d_%H_%M_%S')
            end_time = end_time + pd.Timedelta(minutes=int(delta))
        else:
            end_time_str = '_'.join(
                [filename_parts[len(filename_parts) - 3], filename_parts[len(filename_parts) - 2], end_time_meta[-5:]])
            end_time = pd.to_datetime(end_time_str, format='%Y%m%d_%H_%M_%S')
        #print(f"start: {start_time}. end: {end_time}")
        self.end_time = end_time
        return start_time, end_time

    def get_metadata_from_video(self, video_filepath, start_or_end):
        return_time = None
        success = False
        if start_or_end == "start":
            try:
                # Get the creation date from metadata
                parser = createParser(video_filepath)
                metadata = extractMetadata(parser)
                modify_date = str(metadata.get("creation_date")) # 2022-05-24 08:29:09

                #Converter it into the correct string format
                original_datetime = datetime.strptime(modify_date, '%Y-%m-%d %H:%M:%S')
                return_time = original_datetime.strftime('%Y%m%d_%H_%M_%S')
                self.logger.debug("Obtained video start time from metadata.")
                success = True
            except:
                success = False
        elif start_or_end == "end":
            try:
                # Get the creation date and duration
                modify_date = self.start_time
                duration = self.get_video_duration()

                # Calculate the end time by adding the duration to the creation time
                end_time = modify_date + duration

                # Convert end time to the desired format
                return_time = end_time.strftime('%Y%m%d_%H_%M_%S')
                self.logger.debug("Obtained video end time from metadata.")
                success = True
            except:
                success = False
        return return_time, success

    def get_text_manually(self, frame):

        def submit_time(input_field):
            self.logger.debug(f'Running function submit_time({input_field})')

            # Define variables
            text = input_field.get()
            dialog = self.manual_text_input_window

            # Validate text
            if not text.isdigit() or len(text) != 2 or int(text) > 59:
                # execute code here for when text is not in "SS" format
                self.logger.warning(
                    "Manual input is not in the correct format. The value will be set to an arbitrary 00.")
                text = '00'
            else:
                self.logger.debug("Video times were not extracted from metadata. Resolved manually.")
            self.sec_OCR = text
            while True:
                try:
                    if dialog.winfo_exists():
                        dialog.quit()
                        dialog.destroy()
                        break
                except:
                    time.sleep(0.1)
                    break

        # Define variables
        root = self.main_window

        # Loop until the main window is created
        while True:
            try:
                if root.winfo_exists():
                    break
            except:
                time.sleep(0.1)

        # Create the dialog window
        dialog = tk.Toplevel(root)
        self.manual_text_input_window = dialog
        try:
            screen_width = dialog.winfo_screenwidth()
            screen_height = dialog.winfo_screenheight()
        except:
            screen_width = 1920
            screen_height = 1080

        dialog.wm_attributes("-topmost", 1)
        dialog.title("Time Input")

        # convert frame to tkinter image
        text_roi = self.ocr_roi
        x, y, w, h = text_roi
        img_frame_width = min(screen_width // 2, w * 2)
        img_frame_height = min(screen_height // 2, h * 2)
        frame = frame[y:y + h, x:x + w]
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_frame_width, img_frame_height), Image.LANCZOS)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(img)
        img_width = img.width()
        img_height = img.height()

        # Add image frame containing the video frame
        img_frame = tk.Frame(dialog, width=img_width, height=img_height)
        img_frame.pack(side=tk.TOP, pady=(0, 0))
        img_frame.pack_propagate(0)

        # Add image to image frame
        img_label = tk.Label(img_frame, image=img)
        img_label.pack(side=tk.TOP, pady=(0, 0))

        # Get the name of the video time to display it in the label. THe user then can decide whether ti is ok and eventually try to change it.
        filename_parts = os.path.basename(self.filepath)[:-4].split("_")
        start_time_minutes = ":".join(
            [filename_parts[len(filename_parts) - 2],
             filename_parts[len(filename_parts) - 1]])

        # Add label
        text_field = tk.Text(dialog, height=4, width=120, font=("Arial", 10))
        text_field.insert(tk.END,
                          "The OCR detection apparently failed."
                          "\nEnter the last two digits of the security camera watermark (number of seconds)."
                          "\nThis will ensure cropping will happen at the right times."
                         f"\nStart of the video from filename: {start_time_minutes}")
        text_field.configure(state="disabled", highlightthickness=1, relief="flat", background=dialog.cget('bg'))
        text_field.tag_configure("center", justify="center")
        text_field.tag_add("center", "1.0", "end")
        text_field.pack(side=tk.TOP, padx=(0, 0))
        label = tk.Label(dialog, text="Enter text", font=("Arial", 10), background=dialog.cget('bg'))
        label.pack(pady=2)

        # Add input field
        j = 0
        input_field = tk.Entry(dialog, font=("Arial", 10), width=4)
        input_field.pack(pady=2)
        input_field.bind("<Return>", lambda j=j: submit_time(input_field))
        input_field.focus()

        # Add submit button
        submit_button = tk.Button(dialog, text="Submit", font=("Arial", 10),
                                  command=lambda j=j: submit_time(input_field))
        submit_button.pack(pady=2)

        # Position the window
        dialog_width = dialog.winfo_reqwidth() + img_frame_width
        dialog_height = dialog.winfo_reqheight() + img_frame_height
        dialog_pos_x = int((screen_width // 2) - (img_frame_width // 2))
        dialog_pos_y = 0
        dialog.geometry(f"{dialog_width}x{dialog_height}+{dialog_pos_x}+{dialog_pos_y}")

        # Start the dialog
        dialog.mainloop()

        # When dialog is closed
        return_time = self.sec_OCR
        if len(return_time) > 0:
            success = True
        else:
            success = False
        return return_time, success

    def read_video_frame(self, frame_number: int = 0):

        frame = None
        if self.video_origin == "MS":
            success, frame = self.imageio_reader(self.filepath, frame_number)
        else:
            success, frame = self.opencv_reader(self.filepath, frame_number)
        if frame is not None:
            return success, frame
        else:
            return False, None

    def opencv_reader(self, filepath, frame_number):
        frame = None
        try:
            # Read the frame
            cap = cv2.VideoCapture(filepath)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            success, frame = cap.read()
            cap.release()
        except Exception as e:
            self.logger.warning(f'Unable to read video frame. Video: {self.filepath}, Exception: {e}')
            success = False
        return success, frame

    def imageio_reader(self, filepath, frame_number):
        frame = None
        try:
            # Open the video file using imageio
            video = imageio.get_reader(filepath)

            # Read the first frame
            frame = video.get_data(frame_number)

        except IndexError:
            original_frame_number = frame_number
            success = False
            self.logger.warning(f'IndexError when reading a frame number: {original_frame_number} from file: {filepath}')
            try:

                # Read the frame using open-cv
                success, frame = self.opencv_reader(filepath, original_frame_number)

            except Exception as e:
                self.logger.warning(f'Error occurred when reading the frame using open-cv: {e}')
                pass
            self.logger.warning(f'Attempted reading the frame with open-cv, status: {success}')
            if not success:
                self.logger.warning(f'Scanning for existing frames...')
                while not success:
                    try:
                        if frame_number >= 0:
                            frame_number -= 1
                            frame = video.get_data(max(1, frame_number))
                        else:
                            break
                    except:
                        success = False
                    else:
                        self.logger.warning(f'Frame found at index: {frame_number}')
                        success = True
        else:
            # Convert the frame to BGR color space
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Reading successful
            success = True

        return success, frame

    def get_frame_shape(self):

        # Will retrieve the dimensions of the video
        if self.video_origin == "MS":
            video_reader = imageio.get_reader(self.filepath)

            try:
                metadata = video_reader.get_meta_data()
                frame_width = metadata["size"][0]
                frame_height = metadata["size"][1]
            except:
                # Read the first frame to get its shape properties
                first_frame = video_reader.get_data(0)
                frame_height, frame_width, _ = first_frame.shape
        else:
            try:
                cap = cv2.VideoCapture(self.filepath)
                cap.set(cv2.CAP_PROP_POS_FRAMES, 2)
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
            except:
                raise

        return frame_width, frame_height



    def get_video_fps(self):

        try:
            if self.video_origin == "MS":
                # Get the creation date from metadata
                parser = createParser(self.filepath)
                metadata = extractMetadata(parser)
                fps = float(metadata.get("frame_rate"))
            else:
                cap = cv2.VideoCapture(self.filepath)
                fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
        except Exception as e:
            self.logger.warning(f'Unable to read video fps: {self.filepath}. Exception: {e}')
            fps = 25
        return int(fps)

    def get_video_total_frames(self):

        if self.video_origin == "MS":

            # Will return timedelta object
            duration = self.get_video_duration()

            # Get total number of seconds
            duration = duration.total_seconds()

            # Get fps
            if self.fps is not None:
                fps = self.fps
            else:
                fps = self.get_video_fps()

            # Calculate total_frames
            total_frames = int(duration * fps)

        else:
            cap = cv2.VideoCapture(self.filepath)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

        return total_frames

    def get_video_duration(self):

        try:
            # Get the creation date from metadata
            parser = createParser(self.filepath)
            metadata = extractMetadata(parser)
            duration = str(metadata.get("duration"))
        except Exception as e:
            self.logger.warning(f'Unable to read video duration from video: {self.filepath}. Exception: {e}')
            duration = simpledialog.askstring(f"Unable to read video duration from video: {self.filepath}", "Enter duration time (hh:mm:ss):")

        # Split duration string into hours, minutes, and seconds
        duration_parts = duration.split(':')
        hours, minutes, seconds = map(float, duration_parts)

        # Calculate the timedelta for the duration
        duration = timedelta(hours=hours, minutes=minutes, seconds=seconds)

        return duration
