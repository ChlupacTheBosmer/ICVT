# Import ICVT components
from iccs import ICCS
import utils
import vid_data

# Part of python modules
import asyncio
import configparser
import json
import os

class mtcCrop(ICCS):
    def __init__(self):

        # Define logger
        self.logger = self.log_define()

        # First log
        self.logger.info("Initializing mtcCropE - Metacentrum Crop Engine class...")

        # Create config and read file
        self.config = self.config_create()
        self.config_read()

        # Define variables
        self.scanned_folders = []
        self.dir_hierarchy = False
        self.loaded = False
        self.crop_mode: int = 3
        self.auto = 0
        self.frames = []
        self.modified_frames = []
        self.video_filepaths = []
        self.points_of_interest_entry = []
        self.button_images = []
        self.buttons = []
        self.gui_imgs = []
        self.cap = None
        self.image_details_dict = {}
        self.main_window = None
        self.visit_index = 0

        # Call the function to get a valid output folder
        self.output_folder = self.get_valid_output_folder("output")

        # Call the function to get a valid output folder
        self.video_folder_path = self.get_valid_output_folder("video")

        # Construct ROI data
        self.reload_roi_entries()

        # Create output folders
        utils.create_dir(self.output_folder)
        utils.create_dir(os.path.join(self.output_folder, "whole frames"))

        self.load_videos()

        self.load_progress()

        self.crop_engine()

    def get_valid_output_folder(self, folder_name):
        while True:
            folder_path = input(f'Enter the path to your {folder_name} folder. Make sure it is correct. Do not use quotes: ')

            # Check if the path exists and is a directory
            if os.path.exists(folder_path) and os.path.isdir(folder_path):
                return folder_path
            else:
                print("Invalid path. Please provide a valid directory path.")

    def config_create(self):
        # Set default values
        config = configparser.ConfigParser()
        config['Crop settings'] = {
            'crop_mode': '3',
            'crop_interval_frames': '30',
            'frames_per_visit': '0',
            'yolo_processing': '0',
            'default_label_category': '6',
            'yolo_conf': '0.25',
            'randomize_interval': '0',
            'export_whole_frame': '0',
            'export_crops': '1',
            'crop_size': '640',
            'random_offset_range': '100',
            'filename_prefix': ''
        }

        # Check if settings_crop.ini exists, and create it with default values if not
        if not os.path.exists('settings_mtc_crop.ini'):
            with open('settings_mtc_crop.ini', 'w', encoding='utf-8') as configfile:
                config.write(configfile)
        return config

    def config_read(self):

        # Define logger
        self.logger.debug('Running function config_read()')

        try:
            # Read settings from settings_crop.ini
            self.config.read('settings_mtc_crop.ini', encoding='utf-8')

            # Get crop values from config
            self.crop_mode = int(self.config['Crop settings'].get('crop_mode', '1').strip())
            self.frame_skip = int(self.config['Crop settings'].get('crop_interval_frames', '30').strip())
            self.frames_per_visit = int(self.config['Crop settings'].get('frames_per_visit', '0').strip())
            self.yolo_processing = int(self.config['Crop settings'].get('yolo_processing', '0').strip())
            self.default_label_category = int(self.config['Crop settings'].get('default_label_category', '6').strip())
            self.yolo_conf = float(self.config['Crop settings'].get('yolo_conf', '0.25').strip())
            self.randomize = int(self.config['Crop settings'].get('randomize_interval', '0').strip())
            self.whole_frame = int(self.config['Crop settings'].get('export_whole_frame', '0').strip())
            self.cropped_frames = int(self.config['Crop settings'].get('export_crops', '1').strip())
            self.crop_size = int(self.config['Crop settings'].get('crop_size', '640').strip())
            self.offset_range = int(self.config['Crop settings'].get('random_offset_range', '100').strip())
            self.prefix = self.config['Crop settings'].get('filename_prefix', '').strip()

        except ValueError:
            self.logger.warning('Invalid folder/file path or crop settings found in settings_crop.ini')

    def config_write(self):

        # Define logger
        self.logger.debug("Running function config_write()")
        config = self.config
        config.read('settings_mtc_crop.ini')

        # Update values in the config file
        config.set('Crop settings', 'crop_mode', str(self.crop_mode))
        config.set('Crop settings', 'crop_interval_frames', str(self.frame_skip))
        config.set('Crop settings', 'frames_per_visit', str(self.frames_per_visit))
        config.set('Crop settings', 'yolo_processing', str(self.yolo_processing))
        config.set('Crop settings', 'default_label_category', str(self.default_label_category))
        config.set('Crop settings', 'yolo_conf', str(self.yolo_conf))
        config.set('Crop settings', 'randomize_interval', str(self.randomize))
        config.set('Crop settings', 'export_whole_frame', str(self.whole_frame))
        config.set('Crop settings', 'export_crops', str(self.cropped_frames))
        config.set('Crop settings', 'crop_size', str(self.crop_size))
        config.set('Crop settings', 'random_offset_range', str(self.offset_range))
        config.set('Crop settings', 'filename_prefix', str(self.prefix))

        # Save changes to the config file
        with open('settings_mtc_crop.ini', 'w', encoding='utf-8') as configfile:
            config.write(configfile)

    # TODO: Modify
    def load_progress(self):

        # Define logger
        self.logger.debug(f"Running function load_progress()")

        # Confirm with the user if they want to load the settings
        result = str(input("Do you want to load settings? This will overwrite any unsaved progress. (y/n):"))

        if result == "y":

            # Call the function to get a valid output folder
            save_file_folder = self.get_valid_output_folder("save")

            if utils.check_path(save_file_folder, 0):

                # Create an in-memory file object
                filepath = os.path.join(save_file_folder, 'crop_information.json')
                if os.path.isfile(filepath):
                    try:
                        if self.main_window.winfo_exists():
                            self.main_window.destroy()
                    except:
                        self.logger.debug("Error: Unexpected, window destroyed before reference.")
                    self.points_of_interest_entry = []
                    with open(filepath, "r") as json_file:
                        data_combined = json.load(json_file)

                        # Restore data from dictionary
                        self.auto = (data_combined["auto_processing"])
                        data_matched = data_combined["video_data"]
                        self.points_of_interest_entry = [item for item in data_matched.values()]
                        video_filepaths_new = [item[1] for item in data_matched.values()]

                    # Compare the loaded and the currently found video filepaths.
                    set1 = set({os.path.basename(filepath) for filepath in video_filepaths_new})
                    print(set1)
                    set2 = set({os.path.basename(filepath) for filepath in self.video_filepaths})
                    print(set2)
                    if not set1 == set2:
                        print("The contents of the video folder have changed since the save has been made. Cannot load the progress. Please start over.")
                        self.reload_roi_entries()
                    else:
                        self.video_filepaths = []
                        self.video_filepaths = video_filepaths_new.copy()
                else:
                    print("There are no save files in the current directory.")
            else:
                print("Invalid video folder path")

    # TODO: Modify
    def crop_engine(self):

        # Define logger
        self.logger.debug("Running function crop_engine()")

        # Define variables
        self.image_details_dict = {}

        # Ask to confirm whether the process should begin
        # Confirm with the user if they want to load the settings
        result = str(input("Do you want to start the cropping process? (y/n):"))

        if result == "y":
            self.logger.info("Initializing the cropping engine...")

            # Check if some ROIs were selected
            if len(self.points_of_interest_entry[0][0]) == 0:
                print("No regions of interest selected. Please select at least one ROI.")
                return

            # Define arrays
            valid_annotations_array = []
            valid_annotation_data_entry = []

            # Close window and start cropping
            self.logger.debug(f"Start cropping on the following videos: {self.video_filepaths}")

            # Check validity of paths
            video_ok = utils.check_path(self.video_folder_path, 0)
            if not video_ok:
                print(f"Unspecified path to a video folder.")
                return

            # The whole frame settings is artificially altered to allow for whole frame generation - messy
            orig_wf = self.whole_frame
            self.whole_frame = 0

            # Starting from the second frame of every video frames are generated.
            for i, filepath in enumerate(self.video_filepaths):
                self.video_file_object = vid_data.Video_file(filepath, self.main_window, initiate_start_and_end_times=False)
                self.fps = self.video_file_object.fps
                total_frames = self.video_file_object.total_frames
                self.visit_duration = total_frames // self.fps
                frame_number_start = 2
                success, frame = self.video_file_object.read_video_frame(frame_number_start)
                img_paths = asyncio.run(self.generate_frames(frame, success, os.path.basename(self.video_filepaths[i]), i, frame_number_start))
            self.whole_frame = orig_wf

if __name__ == '__main__':
    mtc_crop = mtcCrop()