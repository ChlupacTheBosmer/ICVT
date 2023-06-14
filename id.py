import tkinter as tk
import os
os.environ["PATH"] = os.path.dirname(__file__) + os.pathsep + os.environ["PATH"]
import mpv
import requests
import json
import logging
import configparser
from utils import ask_yes_no

def config_read():
    global video_folder_path
    global annotation_file_path
    global scan_folders
    global config
    global logger
    logger.debug('Running function config_read()')
    # load config or create the file
    # Set default values
    config = configparser.ConfigParser()
    config['Resource Paths'] = {
        'video_folder_path': '',
        'annotation_file_path': ''
    }
    config['ICID settings'] = {
        'Scan_default_folders': '1'
    }

    # Check if settings_crop.ini exists, and create it with default values if not
    if not os.path.exists('settings_ICID.ini'):
        with open('settings_ICID.ini', 'w', encoding='utf-8') as configfile:
            config.write(configfile)

    # Read settings from settings_crop.ini
    config.read('settings_ICID.ini', encoding='utf-8')

    # Get values from the config file
    try:
        video_folder_path = config['Resource Paths'].get('video_folder_path', '').strip()
        annotation_file_path = config['Resource Paths'].get('annotation_file_path', '').strip()
    except ValueError:
        print('Error: Invalid folder/file path found in settings_ICID.ini')
    # Get ICID settings values from config
    try:
        scan_folders = config['ICID settings'].get('Scan_default_folders', '0').strip()
    except ValueError:
        print('Error: Invalid ICID settings specified in settings_ICID.ini')
    resources = [video_folder_path, annotation_file_path]
    settings = [scan_folders]
    return resources, settings

def select_file(selected_file_index, index, root):
    global logger
    logger.debug(f'Running function select_file({selected_file_index}, {index}, {root})')
    selected_file_index.set(index + 1)
    root.destroy()

def scan_default_folders():
    global logger
    logger.debug('Running function scan_default_folders()')

    # scan default folders
    file_type = ["excel (watchers)", "excel (manual)"]
    video_folder_path: str = ""
    annotation_file_path: str = ""

    # Check if scan folder feature is on
    if scan_folders == "1":

        # Create directories if they do not exist
        if not os.path.exists("videos/"):
            os.makedirs("videos/")
        if not os.path.exists("excel/"):
            os.makedirs("excel/")

        # Detect video files
        scan_video_files = [f for f in os.listdir('videos') if f.endswith('.mp4')]
        if scan_video_files:
            response = ask_yes_no(f"Video files detected in the default folder. Do you want to continue?")
            if response:
                video_folder_path = 'videos'

        # Check if the current default crop mode requires an annotation file
        if crop_mode == 1 or crop_mode == 2:

            # Detect Excel files
            scan_excel_files = [f for f in os.listdir('excel') if f.endswith('.xlsx') or f.endswith('.xls')]
            if scan_excel_files:
                response = ask_yes_no(f"Excel files detected in the default folder. Do you want to continue?")
                if response:

                    # Create the window for selecting the Excel file
                    excel_files_win = tk.Tk()
                    excel_files_win.title("Select file")
                    excel_files_win.wm_attributes("-topmost", 1)

                    # Create window contents
                    label_frame = tk.Frame(excel_files_win)
                    label_frame.pack(side=tk.TOP, expand=True, fill=tk.BOTH, padx=20)
                    prompt_label = tk.Label(excel_files_win,
                                            text=f"Please select the {file_type[(crop_mode - 1)]} file you\nwant to use as the source of visit times.")
                    prompt_label.pack()
                    label = tk.Label(excel_files_win, text="Excel files in the folder:")
                    label.pack()
                    outer_frame = tk.Frame(excel_files_win)
                    outer_frame.pack(side=tk.TOP, expand=True, fill=tk.BOTH, padx=20, pady=20)
                    for i, f in enumerate(scan_excel_files):
                        button = tk.Button(outer_frame, text=f"{i + 1}. {f}", width=30,
                                           command=lambda i=i: select_file(selected_file_index, i, excel_files_win))
                        button.pack(pady=0)
                    selected_file_index = tk.IntVar()

                    # Set the window position to the center of the screen
                    excel_files_win.update()
                    screen_width = excel_files_win.winfo_screenwidth()
                    screen_height = excel_files_win.winfo_screenheight()
                    window_width = excel_files_win.winfo_reqwidth()
                    window_height = excel_files_win.winfo_reqheight()
                    x_pos = int((screen_width - window_width) / 2)
                    y_pos = int((screen_height - window_height) / 2)
                    excel_files_win.geometry(f"{window_width}x{window_height}+{x_pos}+{y_pos}")
                    excel_files_win.mainloop()
                    selection = selected_file_index.get()

                    # Assign the path of the selected file to a variable
                    if selection > 0 and selection <= len(scan_excel_files):
                        annotation_file_path = os.path.join(("excel/"), scan_excel_files[selection - 1])
                        print(f'Selected file: {annotation_file_path}')
                    else:
                        print('Invalid selection')
    return video_folder_path, annotation_file_path

def identify_insect(frame_image_path):
    url = "https://api.inaturalist.org/v1/identifications"
    files = {"file": open(frame_image_path, "rb")}

    payload = {
        "identification": {
            "taxon_id": 47157,  # Insecta taxon ID, change if you want to limit the search to a specific taxon
            "observation_id": "",  # Leave empty if you don't have an observation ID
            "body": "Insect identification request",
        }
    }

    headers = {"Content-Type": "application/json"}

    response = requests.post(url, files=files, json=payload, headers=headers)
    if response.status_code == 200:
        data = response.json()
        if data["total_results"] > 0:
            # Extract the order or family information from the response
            order = data["results"][0]["taxon"]["name"]
            family = data["results"][0]["taxon"]["preferred_common_name"]

            return order, family

    return None, None

# Function to set the geometry of a window to the top-left corner and a specific size
def set_window_geometry(window, width, height, x, y):
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    window.geometry(f"{width}x{height}+{x}+{y}")

def pause_shit(player):
    if player.pause == True:
        player.pause = False
    else:
        player.pause = True

def zoom_vid(player, where):
    global zoom_level
    if where > 0:
        zoom_level = zoom_level + 0.1 # Adjust the value as needed
        player.video_zoom = zoom_level
    else:
        zoom_level = zoom_level - 0.1  # Adjust the value as needed
        player.video_zoom = zoom_level

def pan_vid(player, where):
    global video_pan_x
    global video_pan_y
    if where == "left":
        video_pan_x = video_pan_x - 0.05 # Adjust the value as needed
        player.video_pan_x = video_pan_x
    elif where == "right":
        video_pan_x = video_pan_x + 0.05  # Adjust the value as needed
        player.video_pan_x = video_pan_x
    elif where == "up":
        video_pan_y = video_pan_y - 0.05  # Adjust the value as needed
        player.video_pan_y = video_pan_y
    elif where == "down":
        video_pan_y = video_pan_y + 0.05  # Adjust the value as needed
        player.video_pan_y = video_pan_y

def center_vid(player):
    global video_pan_x
    global video_pan_y
    global zoom_level
    player.video_pan_x = 0
    player.video_pan_y = 0
    player.video_zoom = 0.0

def speed_vid(player, how):
    global speed
    if how == 0:
        speed = 1.0
        player.speed = 1.0
    else:
        speed = speed + how  # Adjust the value as needed
        player.speed = speed

def id_insect():
    global order
    global family
    order, family = identify_insect("1.png")
    if order and family:
        print(f"Insect Order: {order}")
        print(f"Insect Family: {family}")
    else:
        print("Unable to identify the insect.")

def log_write():
    global logger
    # Create a logger instance
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Create a file handler that logs all messages, and set its formatter
    file_handler = logging.FileHandler('runtime.log', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Create a console handler that logs only messages with level INFO or higher, and set its formatter
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Write log messages
    # logger.debug('This message will be written to the file only.')
    # logger.info('This message will be written to both the file and the console.')
    # logger.warning('This message will be written to both the file and the console.')
    # logger.error('This message will be written to both the file and the console.')
    # logger.critical('This message will be written to both the file and the console.')

def open_ICID_window():
    global zoom_level
    global video_pan_x
    global video_pan_y
    global speed

    zoom_level = 0.0
    video_pan_x = 0.0
    video_pan_y = 0.0
    speed = 1.0
    j = 0

    # Create the main tkinter window (mpv_window)
    mpv_window = tk.Tk()
    set_window_geometry(mpv_window, mpv_window.winfo_screenwidth() * 2 // 3, mpv_window.winfo_screenheight(), -100, 0)
    mpv_window.title("MPV Window")

    # Create the MPV player and play the video
    player = mpv.MPV(player_operation_mode='pseudo-gui',
                     script_opts='osc-deadzonesize=0,osc-minmousemove=1',
                     input_default_bindings=True,
                     input_vo_keyboard=True,
                     osc=True)
    player.geometry = f'{mpv_window.winfo_screenwidth() * 2 // 3}x{mpv_window.winfo_screenheight()}+-50+-20'
    player.play('videos/GR2_L2_LavSto2_20220524_09_29.mp4')

    # Create the second tkinter window (control_panel)
    control_panel = tk.Toplevel(mpv_window)
    set_window_geometry(control_panel, mpv_window.winfo_screenwidth() // 3, mpv_window.winfo_screenheight(), mpv_window.winfo_screenwidth() * 2 // 3 - 100, 0)
    control_panel.title("Control Panel")

    # Create the input fields and labels in control_panel
    label_texts = ["Label 1", "Label 2", "Label 3", "Label 4", "Label 5", "Label 6", "Label 7",
                   "Label 8", "Label 9", "Label 10", "Label 11", "Label 12", "Label 13"]
    initial_values = ["Value 1", "Value 2", "Value 3", "Value 4", "Value 5", "Value 6", "Value 7",
                      "Value 8", "Value 9", "Value 10", "Value 11", "Value 12", "Value 13"]

    # Create the input fields and labels
    for i in range(len(label_texts)):
        label = tk.Label(control_panel, text=label_texts[i])
        label.grid(row=i, column=1, padx=10, pady=5)

        entry = tk.Entry(control_panel)
        entry.insert(tk.END, initial_values[i])
        entry.grid(row=i, column=2, padx=10, pady=5)

    # Create the additional labels and input fields
    visitor_species_label = tk.Label(control_panel, text="Visitor Species")
    visitor_species_label.grid(row=len(label_texts), column=1, padx=10, pady=5)

    visitor_species_entry = tk.Entry(control_panel)
    visitor_species_entry.grid(row=len(label_texts), column=2, padx=10, pady=5)

    visitor_order_label = tk.Label(control_panel, text="Visitor Order")
    visitor_order_label.grid(row=len(label_texts) + 1, column=1, padx=10, pady=5)

    visitor_order_entry = tk.Entry(control_panel)
    visitor_order_entry.grid(row=len(label_texts) + 1, column=2, padx=10, pady=5)

    # Create the buttons for video playback control
    button_frame = tk.Frame(control_panel)
    button_frame.grid(row=len(label_texts) + 2, column=1, columnspan=2, pady=10)

    play_button = tk.Button(button_frame, text="Play", command=lambda j=j: player.play('videos/GR2_L2_LavSto2_20220524_09_29.mp4'))
    play_button.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

    pause_button = tk.Button(button_frame, text="Pause", command=lambda j=j: pause_shit(player))
    pause_button.grid(row=0, column=1, padx=5, pady=5, sticky="nsew", columnspan=2)

    stop_button = tk.Button(button_frame, text="Stop", command=lambda j=j: player.stop(True))
    stop_button.grid(row=0, column=3, padx=5, pady=5, sticky="nsew")

    frame_back_button = tk.Button(button_frame, text="<- Frame", command=lambda j=j: player.frame_back_step())
    frame_back_button.grid(row=1, column=0, padx=5, pady=5, sticky="nsew", columnspan=2)

    frame_forw_button = tk.Button(button_frame, text="Frame ->", command=lambda j=j: player.frame_step())
    frame_forw_button.grid(row=1, column=2, padx=5, pady=5, sticky="nsew", columnspan=2)

    zoom_in_button = tk.Button(button_frame, text="Zoom In", command=lambda j=j: zoom_vid(player, 1))
    zoom_in_button.grid(row=2, column=0, padx=5, pady=5, sticky="nsew", columnspan=2)

    zoom_out_button = tk.Button(button_frame, text="Zoom Out", command=lambda j=j: zoom_vid(player, -1))
    zoom_out_button.grid(row=2, column=2, padx=5, pady=5, sticky="nsew", columnspan=2)

    sshot_button = tk.Button(button_frame, text="Screenshot", command=lambda j=j: player.screenshot_to_file("1.png"))
    sshot_button.grid(row=3, column=0, padx=5, pady=5, sticky="nsew", columnspan=2)

    id_button = tk.Button(button_frame, text="Identify", command=lambda j=j: id_insect())
    id_button.grid(row=3, column=2, padx=5, pady=5, sticky="nsew", columnspan=2)

    pan_up_button = tk.Button(button_frame, text="Pan Up", command=lambda j=j: pan_vid(player, "down"))
    pan_up_button.grid(row=4, column=2, padx=5, pady=5, sticky="nsew")

    pan_left_button = tk.Button(button_frame, text="Pan Left", command=lambda j=j: pan_vid(player, "right"))
    pan_left_button.grid(row=5, column=1, padx=5, pady=5, sticky="nsew")

    center_button = tk.Button(button_frame, text="Center", command=lambda j=j: center_vid(player))
    center_button.grid(row=5, column=2, padx=5, pady=5, sticky="nsew")

    pan_right_button = tk.Button(button_frame, text="Pan right", command=lambda j=j: pan_vid(player, "left"))
    pan_right_button.grid(row=5, column=3, padx=5, pady=5, sticky="nsew")

    pan_down_button = tk.Button(button_frame, text="Pan Down", command=lambda j=j: pan_vid(player, "up"))
    pan_down_button.grid(row=6, column=2, padx=5, pady=5, sticky="nsew")

    speed_up1_button = tk.Button(button_frame, text="+0.5×", command=lambda j=j: speed_vid(player, 0.5))
    speed_up1_button.grid(row=8, column=0, padx=5, pady=5, sticky="nsew", columnspan=1)

    speed_up2_button = tk.Button(button_frame, text="+0.25×", command=lambda j=j: speed_vid(player, 0.25))
    speed_up2_button.grid(row=8, column=1, padx=5, pady=5, sticky="nsew", columnspan=1)

    speed_norm_button = tk.Button(button_frame, text="1×", command=lambda j=j: speed_vid(player, 0))
    speed_norm_button.grid(row=8, column=2, padx=5, pady=5, sticky="nsew", columnspan=1)

    speed_down_button = tk.Button(button_frame, text="-0.25×", command=lambda j=j: speed_vid(player, -0.25))
    speed_down_button.grid(row=8, column=3, padx=5, pady=5, sticky="nsew", columnspan=1)

    speed_down2_button = tk.Button(button_frame, text="-0.5×", command=lambda j=j: speed_vid(player, -0.5))
    speed_down2_button.grid(row=8, column=4, padx=5, pady=5, sticky="nsew", columnspan=1)

    mpv_window.mainloop()

    print('calling player.terminate.')
    player.terminate()
    print('terminate player.returned.')

def initialise():
    log_write()
    logger.debug("Running function initialise()")
    video_folder_path, annotation_file_path = scan_default_folders()
