import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import cv2
import random

class ImageGridWindow:
    def __init__(self, visit_number, image_details_dict):
        self.visit_number = visit_number
        self.image_details_dict = image_details_dict
        self.gui_imgs = []

        # Filter images for the given visit number
        self.filtered_images = [img for img, details in self.image_details_dict.items() if details[3] == visit_number]
        self.filtered_details = [details for key, details in self.image_details_dict.items() if details[3] == visit_number]
        self.roi_numbers = set(details[2] for details in self.image_details_dict.values())
        #print(self.roi_numbers)

        self.window = tk.Tk()
        self.window.title(f"Visit Number: {visit_number}")
        self.create_grid()

    def create_grid(self):

        # Define variables
        outer_frame = self.window

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
        checkboxes = []

        for i, roi in enumerate(self.roi_numbers):
            num_rows = len(self.roi_numbers)
            filtered_by_roi = [details for details in self.filtered_details if
                                     details[2] == roi]
            filtered_by_detected_visitors = [details for details in filtered_by_roi if
                                     details[4] == 1]
            filtered_by_empty_frames = [details for details in filtered_by_roi if
                                     details[4] == 0]

            # Add a button frame
            button_frame = tk.Frame(target_frame)
            button_frame.pack(side=tk.TOP)

            #print(filtered_by_empty_frames)
            #print(filtered_by_detected_visitors)

            for col in range(5):
                if col < len(filtered_by_detected_visitors):
                    image_path = filtered_by_detected_visitors[col][0]
                elif abs(col-5) < len(filtered_by_empty_frames):
                    image_path = filtered_by_empty_frames[abs(col-5)][0]
                else:
                    continue
                print(image_path)
                img_tk = self.load_icon(image_path, (100, 100), self.window)

                button = ttk.Button(button_frame, image=img_tk, command=lambda image_path=image_path: self.show_enlarged_image(image_path))
                button.image = img_tk
                button.grid(row=i, column=col)

            image_paths = [sublist[0] for sublist in filtered_by_empty_frames] + [sublist[0] for sublist in filtered_by_detected_visitors]

            button = ttk.Button(button_frame, image=img_tk,
                                command=lambda image_paths=image_paths: self.view_images(image_paths))
            button.image = img_tk
            button.grid(row=i, column=5)

            # Create a label and checkbox for each row
            label_text = f"{len(filtered_by_detected_visitors)}/{len(filtered_by_roi)}"
            label = ttk.Label(button_frame, text=label_text)
            label.grid(row=i, column=6)

            checkbox_var = tk.IntVar()  # Create an IntVar for each checkbox
            self.checkbox_vars.append(checkbox_var)  # Store the IntVar in the list

            checkbox = ttk.Checkbutton(button_frame, variable=checkbox_var)
            checkbox.grid(row=i, column=7)
            checkboxes.append(checkbox)

        # Update the canvas to show the buttons
        canvas.create_window((0, 0), window=target_frame, anchor=tk.NW)
        target_frame.update_idletasks()

        # Configure the canvas to show the entire frame
        canvas.configure(scrollregion=canvas.bbox("all"))

        # Bind mouse wheel event to canvas
        canvas.bind_all("<MouseWheel>",
                        lambda event: canvas.yview_scroll(-1 * (event.delta // 120), "units"))

        # Create save button
        save_button = tk.Button(target_frame, text="Save", command=lambda: self.close_window())
        save_button.pack(side=tk.TOP)

        # Set the geometry of the main window to fit the content
        self.window.geometry(f"{1000}x{600}")

    def close_window(self):
        self.window.withdraw()
        self.window.destroy()

    def save_checkbox_values(self):
        checkbox_values = [var.get() for var in self.checkbox_vars]
        print("Checkbox Values:", checkbox_values)
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

    def load_icon(self, path, size: tuple = (50, 50), master=None, pil_ok: bool = False):

        stream = open(path, "rb")
        bytes = bytearray(stream.read())
        numpyarray = np.asarray(bytes, dtype=np.uint8)
        img1 = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)

        # Convert the BGR image to RGB (Tkinter uses RGB format)
        # img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

        # Convert the NumPy array to a PIL image
        pil_img = Image.fromarray(img1)

        img = pil_img.resize(size)
        if pil_ok:
            return img
        if not master == None:
            img = ImageTk.PhotoImage(master=master, image=img)
        else:
            img = ImageTk.PhotoImage(img)
        self.gui_imgs.append(img)
        return img

    def view_images(self, image_paths):
        self.image_paths = image_paths
        self.current_index = 0
        while True:
            image_path = self.image_paths[self.current_index]
            image = cv2.imread(image_path)

            cv2.imshow("Image Viewer", image)

            key = cv2.waitKey(0)
            if key == ord('a'):
                self.prev_image()
            elif key == ord('d'):
                self.next_image()
            elif key == 27:  # Press Esc to exit
                break

        cv2.destroyAllWindows()

    def next_image(self):
        self.current_index = (self.current_index + 1) % len(self.image_paths)

    def prev_image(self):
        self.current_index = (self.current_index - 1) % len(self.image_paths)


def main():
    # Assuming you have self.image_details_dict as your dictionary containing image details
    # For demonstration purposes, let's generate some random data
    image_details_dict = {}
    for i in range(100):
        image_name = f"Image{i}"
        image_path = f"/Users/chlup/PycharmProjects/ICVT/resources/img/placeholder_640.png"
        frame_number = random.randint(1, 20000)
        roi_number = random.randint(1, 5)
        visit_number = random.randint(1, 3)
        last_list_value = random.randint(0, 1)
        image_details_dict[image_name] = [image_path, frame_number, roi_number, visit_number, last_list_value]

    # Create a window for each visit number
    visit_numbers = set(details[3] for details in image_details_dict.values())
    for visit_number in visit_numbers:
        app = ImageGridWindow(visit_number, image_details_dict)
        tk.mainloop()

        # Call the method to get checkbox values after the mainloop exits
        checkbox_values, filtered_details = app.save_checkbox_values()
        print("Checkbox Values outside the class instance:", checkbox_values)

        for i, var in enumerate(checkbox_values):
            if var == 0:
                filtered_by_roi = [details for details in filtered_details if
                                   details[2] == i]
                for detail in filtered_by_roi:
                    print(detail[0])
            else:
                print("var was 1")

if __name__ == "__main__":
    main()