import cv2
from ultralytics import YOLO

# Global variables
selected_roi = None
set_roi = False
tracking = False
tracker = None
paused = False
frame = None

# Function to handle keyboard events
def keyboard_event(key):
    global paused

    if key == ord('p'):
        paused = not paused

def run_object_detection():
    global selected_roi, set_roi, tracking, frame, tracker, box_coords_int, box_coords_global, box_coords_global1

    crop = frame[selected_roi[1]:selected_roi[3], selected_roi[0]:selected_roi[2]]

    height, width, _ = crop.shape
    print(height)

    model = YOLO(r"/resources/yolo/best.pt")
    results = model(crop, save=True, imgsz=height, conf=0.75, save_txt=False, max_det=1,
                    stream=True)
    for i, result in enumerate(results):
        boxes = result.boxes
        box_xywh = boxes.xywh
        box_xyxy = boxes.xyxy
        # print(box_xywh[0].tolist())
        # print(tuple(box_xywh[0].tolist()))




    if box_xywh.numel() > 0:
        print("jo")
        tracking = True
        tracker = cv2.TrackerCSRT_create()
        box_coords_int1 = tuple(int(coord) for coord in box_xywh[0].tolist())
        box_coords_int = tuple(int(coord) for coord in box_xyxy[0].tolist())

        print(box_coords_int)

        # cv2.rectangle(crop, (box_coords_int[0], box_coords_int[1]), (
        #     box_coords_int[2],
        #     box_coords_int[3]), (255, 255, 255), 2)
        # cv2.imshow("Frame", crop)


        box_coords_global1 = [box_coords_int[0] + selected_roi[0], box_coords_int[1] + selected_roi[1],
                             box_coords_int[2] + selected_roi[0], box_coords_int[3] + selected_roi[1]]
        box_coords_global = [box_coords_int1[0] + selected_roi[0], box_coords_int1[1] + selected_roi[1],
                             (box_coords_int[2] + selected_roi[0])-(box_coords_int1[0] + selected_roi[0]), (box_coords_int[3] + selected_roi[1])-(box_coords_int1[1] + selected_roi[1])]

        tracker.init(frame_copy, box_coords_global)


# Load the video
video_path = r"/videos/GR2_L2_LavSto2_20220524_10_29.mp4"
cap = cv2.VideoCapture(video_path)

# Set the position of the first frame to frame number 5016
fps = cap.get(cv2.CAP_PROP_FPS)
frame_number = ((0*60) + 1) * int(fps)
#cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

# Global variable for timeout
max_frames_still = 10
frames_still = 0
x, y, w, h = 0, 0, 0, 0

# Global variable for tracking the previous bounding box
prev_bbox = None

# Create a window and set the mouse and keyboard event callbacks
cv2.namedWindow("Video")

x1, y1, x2, y2 = None, None, None, None

while True:
    # Read a frame from the video
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    while not ret:
        ret, frame = cap.read()

    # Create a copy of the frame
    frame_copy = frame.copy()

    # if frame_number % 20 == 0:
    #     print(frame_number)
    #
    #     # Step 1: Get original dimensions
    #     height, width, _ = frame_copy.shape
    #
    #     # Step 2: Find the maximum dimension
    #     max_dim = max(height, width)
    #
    #     # Step 3: Create a new blank image
    #     new_image = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
    #
    #     # Step 4: Calculate padding
    #     pad_w = abs(width - max_dim) // 2
    #     pad_h = abs(height - max_dim) // 2
    #
    #     # Step 5: Calculate the position of the original frame in the new image
    #     pos_x = pad_w if width < max_dim else 0
    #     pos_y = pad_h if height < max_dim else 0
    #
    #     # Step 6: Calculate the final position of the original frame
    #     final_pos_x = pos_x + pad_w
    #     final_pos_y = pos_y + pad_h
    #
    #     # Step 7: Copy the original frame to the new image
    #     new_image[final_pos_y:final_pos_y + height, final_pos_x:final_pos_x + width] = frame
    #     #resized_image = cv2.resize(new_image, (new_image.shape[1] // 2, new_image.shape[0] // 2))
    #     resized_image = cv2.resize(new_image, (640, 640))
    #     print(resized_image.shape[0])
    #
    #     # Define and run the model
    #     model = YOLO(r"D:\Dílna\Kutění\Python\ICCS\resources\yolo\best.pt")
    #     results = model(resized_image, save=False, imgsz=resized_image.shape[0], conf=0.25, save_txt=False, max_det=1,
    #                     stream=True)
    #     for i, result in enumerate(results):
    #         boxes = result.boxes.data
    #         print(boxes)
    #
    #     scaled_coords = boxes.clone().detach()
    #
    #     # Reverse scaling
    #     scaled_coords[:, [0, 2]] = (scaled_coords[:, [0, 2]] * (new_image.shape[1] / resized_image.shape[1])) - final_pos_x
    #     scaled_coords[:, [1, 3]] = (scaled_coords[:, [1, 3]] * (new_image.shape[0] / resized_image.shape[0])) - final_pos_y
    #
    #     # Display the modified frame with bounding box
    #     for coord in scaled_coords:
    #         x1, y1, x2, y2, _, _ = coord
    #         x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    # if x1 is not None and y1 is not None and x2 is not None and y2 is not None:
    #     cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # If tracking is enabled, update the tracker

    if tracking and tracker is not None:
        if frame_number % 2 == 0:
            success, bbox = tracker.update(frame_copy)
            if success:
                # cv2.rectangle(frame_copy, (box_coords_global1[0], box_coords_global1[1]), (
                # box_coords_global1[2],
                # box_coords_global1[3]), (255, 255, 255), 2)

                x, y, w, h = [int(i) for i in bbox]
                x = x
                y = y

                # Check for tracking failure based on position change
                if prev_bbox is not None:
                    prev_x, prev_y, prev_w, prev_h = prev_bbox
                    delta_x = abs(x - prev_x)
                    delta_y = abs(y - prev_y)
                    if delta_x < 5 and delta_y < 5:
                        frames_still += 1

                if frames_still > max_frames_still:
                    selected_roi = [x-128, y-128, x+128, y+128, 256, 256]
                    frames_still = 0
                    run_object_detection()
                prev_bbox = (x, y, w, h)
            else:
                print("FAILEEEED")
                x, y, w, h = [int(i) for i in bbox]
                selected_roi = [x - 128, y - 128, x + 128, y + 128, 256, 256]
                frames_still = 0
                run_object_detection()
        cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame_copy, "Insect", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame
    #cv2.imshow('Modified Frame', new_image)
    cv2.imshow("Video", frame_copy)

    frame_number += 1

    # Handle keyboard events
    key = cv2.waitKey(1)
    if key == ord("s"):
        # select the bounding box of the object we want to track (make
        # sure you press ENTER or SPACE after selecting the ROI)
        initBB = cv2.selectROI("Video", frame_copy, fromCenter=False,
                               showCrosshair=True)
        print(initBB)
        # start OpenCV object tracker using the supplied bounding box
        # coordinates, then start the FPS throughput estimator as well

        max_dim = max(initBB[2], initBB[3])
        max_dim = max_dim + (32 - (max_dim % 32)) % 32
        selected_roi = [initBB[0], initBB[1], initBB[0] + max_dim, initBB[1] + max_dim, max_dim, max_dim]

        height, width, _ = frame.shape

        selected_roi[2] = min(selected_roi[0] + selected_roi[4], width)
        selected_roi[3] = min(selected_roi[1] + selected_roi[5], height)

        print(selected_roi)

        run_object_detection()

    if key == ord("q"):
        break
    keyboard_event(key)

    # Check if playback is paused
    if paused:
        while paused:
            key = cv2.waitKey(1)
            keyboard_event(key)

# Release the video capture and close the window
cap.release()
cv2.destroyAllWindows()