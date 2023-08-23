
# ICVT modules
from modules.utility import utils

# extra packages
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
from tkinter import filedialog

#PIL Default pyhton packages
import cv2
import random
import os

class ImageGridWindow:
    def __init__(self, visit_number, image_details_dict):
        self.visit_number = visit_number
        self.image_details_dict = image_details_dict
        self.gui_imgs = []
        self.on_window_close_callback = None

        # Filter images for the given visit number
        self.filtered_images = [img for img, details in self.image_details_dict.items() if details[3] == visit_number]
        self.filtered_details = [details for key, details in self.image_details_dict.items() if details[3] == visit_number]
        #print(self.filtered_details)
        self.roi_numbers = set(details[2] for details in self.image_details_dict.values())

        # Set resources folder
        script_directory = os.path.dirname(os.path.abspath(__file__))
        two_parent_folders_up = os.path.abspath(os.path.join(script_directory, '..', '..'))
        self.resources_folder = os.path.join(two_parent_folders_up, 'resources')

        self.window = tk.Tk()
        self.window.title(f"Visit Number: {visit_number}")
        self.create_grid()

    def create_grid(self):

        # Define variables
        outer_frame = self.window
        icon_size = (200, 200)

        # Create a canvas to hold the tile buttons
        canvas = tk.Canvas(outer_frame)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Create a scrollbar for the canvas
        scrollbar = tk.Scrollbar(outer_frame, orient=tk.VERTICAL, command=canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Configure the canvas to use the scrollbar
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        # Create target frame for the rest
        target_frame = tk.Frame(outer_frame)
        target_frame.pack(side=tk.TOP)

        self.checkbox_vars = []
        self.checkboxes = []

        for i, roi in enumerate(self.roi_numbers):
            filtered_by_roi = [details for details in self.filtered_details if
                                     details[2] == roi]
            filtered_by_detected_visitors = [details for details in filtered_by_roi if
                                     details[4] == 1]
            filtered_by_empty_frames = [details for details in filtered_by_roi if
                                     details[4] == 0]

            # Add a button frame
            button_frame = tk.Frame(target_frame)
            button_frame.pack(side=tk.TOP)

            checkbox_var = tk.IntVar(value=0, master=self.window)  # Create an IntVar for each checkbox
            self.checkbox_vars.append(checkbox_var)  # Store the IntVar in the list

            checkbox = ttk.Checkbutton(button_frame, variable=self.checkbox_vars[i])
            checkbox.grid(row=i, column=0, sticky='ew')
            self.checkboxes.append(checkbox)

            # Create a label and checkbox for each row
            label_text = f"{len(filtered_by_detected_visitors)}/{len(filtered_by_roi)}"
            label = ttk.Label(button_frame, text=label_text, padding=(0,20,20,20))
            label.grid(row=i, column=1, sticky='ew')

            # Create individual buttons
            counter = 0

            # Choose which image (visitor/empty) to display - visitor images take preference
            for col in range(5):
                if col < len(filtered_by_detected_visitors) and os.path.exists(filtered_by_detected_visitors[col][0]):
                    image_path = filtered_by_detected_visitors[col][0]
                elif counter < len(filtered_by_empty_frames) and os.path.exists(filtered_by_empty_frames[counter][0]):
                    image_path = filtered_by_empty_frames[counter][0]
                    counter += 1
                else:
                    # Create empty button
                    new_img = Image.new("RGBA", icon_size, (0, 0, 0, 0))
                    new_img = ImageTk.PhotoImage(new_img, master=self.window)
                    dummy_frame = tk.Button(button_frame, image=new_img, foreground="white", state="disabled")
                    dummy_frame.grid(row=i, column=col+2, sticky='ew')
                    continue

                # Create button
                img_tk = self.load_icon(image_path, icon_size, self.window)
                button = ttk.Button(button_frame, image=img_tk, command=lambda image_path=image_path: self.show_enlarged_image(image_path))
                button.grid(row=i, column=col+2, sticky='ew')

            # Create a list of both visitor and empty images for the relevant visit and roi combination
            image_paths = [sublist[0] for sublist in filtered_by_empty_frames] + [sublist[0] for sublist in filtered_by_detected_visitors]

            # Create "show other images" button
            img_else = self.load_icon(os.path.join(self.resources_folder, "img", "more_images.png"), icon_size, self.window)
            button = ttk.Button(button_frame, image=img_else,
                                command=lambda image_paths=image_paths: self.view_images(image_paths))
            button.grid(row=i, column=7, sticky='ew')

        # Update the canvas to show the buttons
        canvas.create_window((0, 0), window=target_frame, anchor=tk.NW)
        target_frame.update_idletasks()

        # Configure the canvas to show the entire frame
        canvas.configure(scrollregion=canvas.bbox("all"))

        # Bind mouse wheel event to canvas
        canvas.bind_all("<MouseWheel>",
                        lambda event: canvas.yview_scroll(-1 * (event.delta // 120), "units"))

        # Create save button
        save_button = tk.Button(target_frame, text="Save", command=lambda: self.close_window(), width=50, height=2)
        save_button.pack(side=tk.TOP)

        # Set the geometry of the main window to fit the content
        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()
        self.window.geometry(f"{min(8*icon_size[0],screen_width-100)}x{min((len(self.roi_numbers)+1)*icon_size[0], screen_height-100)}+{0}+{0}")

    def close_window(self):

        try:
            if self.window.winfo_exists():
                self.window.withdraw()
                self.window.quit()
                self.window.destroy()
        except:
            pass

    def save_checkbox_values(self):
        checkbox_values = [var.get() for var in self.checkbox_vars]
        return checkbox_values, self.filtered_details

    def get_roi_count(self, roi_number):
        count = 0
        for image_name, details in self.image_details_dict.items():
            _, _, current_roi_number, current_visit_number, last_list_value = details
            if current_roi_number == roi_number and current_visit_number == self.visit_number and last_list_value == 1:
                count += 1
        return count

    def show_enlarged_image(self, image_path):
        img = self.load_icon(image_path, (640, 640), self.window, True)
        img.show()

    def load_icon(self, path, size: tuple = (50, 50), master=None, pil_ok: bool = False, cv2_ok: bool = False):

        try:
            stream = open(path, "rb")
            bytes = bytearray(stream.read())
            numpyarray = np.asarray(bytes, dtype=np.uint8)
            img1 = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)

            if cv2_ok:
                return img1

            # Convert the BGR image to RGB (Tkinter uses RGB format)
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

            # Convert the NumPy array to a PIL image
            pil_img = Image.fromarray(img1)

            # Resize the image
            img = pil_img.resize(size)

            if pil_ok:
                return img
            if not master == None:
                img = ImageTk.PhotoImage(master=master, image=img)
            else:
                img = ImageTk.PhotoImage(img)
            self.gui_imgs.append(img)
            return img
        except:
            raise

    def view_images(self, image_paths):
        self.image_paths = image_paths
        self.current_index = 0
        while True:
            image_path = self.image_paths[self.current_index]
            if os.path.exists(image_path):
                image = self.load_icon(image_path, cv2_ok=True)

                cv2.imshow("Image Viewer", image)

                key = cv2.waitKey(0)
                if key == ord('a'):
                    self.prev_image()
                elif key == ord('d'):
                    self.next_image()
                elif key == 27:  # Press Esc to exit
                    break
            else:
                break

        cv2.destroyAllWindows()

    def next_image(self):
        self.current_index = (self.current_index + 1) % len(self.image_paths)

    def prev_image(self):
        self.current_index = (self.current_index - 1) % len(self.image_paths)


def main():

    folder_path = filedialog.askdirectory(title="Select the image folder",
                                          initialdir=os.path.dirname(os.path.abspath(__file__)))

    # Create the dictionary from the names of the image files in the folder
    image_details_dict = gather_image_details(folder_path)

    # Run the windows and ask the user which rois to keep and which to delete
    survey_visits_for_sorting(image_details_dict)

def survey_visits_for_sorting(image_details_dict):

    # Create a window for each visit number
    visit_numbers = set(details[3] for details in image_details_dict.values())
    for visit_number in visit_numbers:

        # Create the window
        image_grid_window = ImageGridWindow(visit_number, image_details_dict)

        # Start the tkinter main loop
        image_grid_window.window.mainloop()

        # When the window closes get the checkbox values
        checkbox_values, filtered_details = image_grid_window.save_checkbox_values()

        # Delete the instance
        del image_grid_window

        for i, var in enumerate(checkbox_values):
            if var == 0:
                filtered_by_roi = [details for details in filtered_details if
                                   details[2] == i + 1]
                for detail in filtered_by_roi:
                    utils.delete_image_and_related_txt(detail[0])
            else:
                pass

def generate_random_dict():
    image_details_dict = {}
    for i in range(100):
        image_name = f"Image{i}"
        image_path = r"D:/Dílna/Kutění/Python/ICCS\output\visitor\GR2_L2_LavSto2_20220525_09_32_1_5490_17_501,77_1141,717.jpg"
        frame_number = random.randint(1, 20000)
        roi_number = random.randint(1, 5)
        visit_number = random.randint(1, 3)
        last_list_value = random.randint(0, 1)
        image_details_dict[image_name] = [image_path, frame_number, roi_number, visit_number, last_list_value]
    return image_details_dict

def gather_image_details(directory_path):
    image_details_dict = {}

    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.png')):
                image_path = os.path.join(root, file)
                file_name, file_extension = os.path.splitext(file)

                parts = file_name.split('_')
                prefix = parts[0]
                timestamp = '_'.join(parts[3:6])
                roi_number = int(parts[6])
                frame_number = int(parts[7])
                visit_number = int(parts[8])

                # Determine if the image is in "empty" or "visitor" folder
                is_visitor = any("visitor" in p.lower() for p in os.path.relpath(root, directory_path).split(os.path.sep))

                txt_file_path = os.path.join(root, file_name + '.txt')
                has_txt_file = os.path.exists(txt_file_path)

                image_details = [image_path, frame_number, roi_number, visit_number, int(is_visitor)]
                image_details_dict[file_name] = image_details

    return image_details_dict

if __name__ == "__main__":
    main()