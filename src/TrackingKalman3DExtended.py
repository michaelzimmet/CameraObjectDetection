import glob
import os
import time

import cv2
import numpy as np
import pandas as pd
import plotly.express as pex
from ultralytics import YOLO

import Config
from Kalman3DExtendedFilter import Kalman3D
from WorldCoordSystem import get_3d_position


def extract_number(filename):
    """
    Extracts the number from the given filename to sort images in their correct sequence
    :param filename: Name of the file
    :return: Number extracted from the filename
    """
    return int(''.join(filter(str.isdigit, os.path.basename(filename))))

def get_poses(position_path, orientation_path):
    """
    Creates a list containing relevant camera pose data per frame
    :param position_path: path to the camera position file
    :param orientation_path: path to the camera orientation file
    :return: List containing camera pose data as list -> [[px, py, pz, qx, qy, qz, qw], ...]
    """
    positions = []
    with position_path.open('r') as f:
        for idx, line in enumerate(f):
            px, py, pz = map(float, line.replace('(', '').replace(')', '').replace('\n', '').split(','))
            entry = [px, py, pz]
            positions.append(entry)

    orientations = []
    with orientation_path.open('r') as f:
        for idx, line in enumerate(f):
            qx, qy, qz, qw = map(float, line.replace('(', '').replace(')', '').replace('\n', '').split(','))
            entry = [qx, qy, qz, qw]
            orientations.append(entry)

    return [a + b for a,b in zip(positions, orientations)]

model = YOLO("../res/last.pt")
tracks = {}

image_folder = "../res/images/db_file_8/left_image_rect_color/"
depth_folder = "../res/images/db_file_8/depth_registered/"

images = sorted(glob.glob(os.path.join(image_folder, "*.png")), key=extract_number)
depth_maps = sorted(glob.glob(os.path.join(depth_folder, "*.tiff")), key=extract_number)
poses = get_poses(Config.IMAGE_DIR / 'db_file_8' / 'pose' / 'pose_position.txt', Config.IMAGE_DIR / 'db_file_8' / 'pose' / 'pose_orientation.txt')

camera_pos = []
track_dict = []
# Intrinsic camera matrix containing the focal length and center relative to the camera for normalization
K = np.array([[480.3142395,          0., 328.18130493],
                    [         0., 480.3142395,  181.9155426],
                    [         0.,          0.,          1.]], dtype=float)

model_process_times = []
total_process_times = []
kalman_process_times = []
for idx, image_path in enumerate(images):
    start_time = time.perf_counter()

    frame = cv2.imread(image_path)
    depth_map = cv2.imread(depth_maps[idx], cv2.IMREAD_UNCHANGED).astype(np.float32)
    px, py, pz, qx, qy, qz, qw = poses[idx]
    camera_pos.append([px, py, pz])

    results = model(frame)
    model_process_times.append((time.perf_counter() - start_time) * 1000)
    detections = results[0].boxes.data.cpu().numpy()

    kalman_detection_times = 0
    for det in detections:
        kalman_time = time.perf_counter()
        x_top_left, y_top_left, x_bottom_right, y_bottom_right, confidence, class_num = det

        # Only process tracking if confidence is above a certain factor
        if confidence > 0.8:
            bounding_box = (x_top_left, y_top_left, x_bottom_right, y_bottom_right)
            normalized_pylon_position = get_3d_position(bounding_box, depth_map, K, px, py, pz, qx, qy, qz, qw)

            if normalized_pylon_position is not None:

                assigned_id = None
                min_dist = float("inf")

                # Loop through all tracks to find a matching track
                for track_id, tracker in tracks.items():
                    pred_pos = tracker.predict()
                    #dist = np.linalg.norm(pred_pos - normalized_pylon_position) # Euclidean distance

                    if tracker.history:
                        window_size = 10
                        if len(tracker.history) >= window_size:
                            avg_pos = np.mean(tracker.history[-window_size:], axis=0)
                        else:
                            avg_pos = np.mean(tracker.history, axis=0)
                    else:
                        avg_pos = tracker.predict()

                    dist = np.linalg.norm(avg_pos - normalized_pylon_position)

                    # Distance threshold
                    if dist < 1.0 and dist < min_dist and tracker.pred_class == class_num:
                        min_dist = dist
                        assigned_id = track_id

                track_dict.append({'id': assigned_id,
                                   'x': normalized_pylon_position[0],
                                   'y': normalized_pylon_position[1],
                                   'z': normalized_pylon_position[2]})

                # create a new track if no suitable track was found
                if assigned_id is None:
                    assigned_id = len(tracks)
                    tracks[assigned_id] = Kalman3D(class_num)
                    tracks[assigned_id].initialize(*normalized_pylon_position)
                else:
                    tracks[assigned_id].update(normalized_pylon_position)

                cv2.putText(frame, f"ID {assigned_id}", (int(x_top_left), int(y_top_left) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.rectangle(frame, (int(x_top_left), int(y_top_left)), (int(x_bottom_right), int(y_bottom_right)), (0, 255, 0), 2)
        kalman_detection_times += ((time.perf_counter() - kalman_time) * 1000)
    kalman_process_times.append(np.array(kalman_detection_times).mean())

    total_process_times.append((time.perf_counter() - start_time) * 1000)
    cv2.imshow("Kalman3D Tracking (Weltkoordinaten)", frame)
    if cv2.waitKey(30) & 0xFF == 27:
        break

cv2.destroyAllWindows()

camera_pos = np.array(camera_pos)
df = pd.DataFrame({'x': camera_pos[:, 1], 'y': camera_pos[:, 0], 'z': camera_pos[:, 2], 'id': 'camera'})
tr = pd.DataFrame(track_dict)
concat = pd.concat([df, tr], axis=0)
min = round(concat[['x', 'y', 'z']].min().min())
max = round(concat[['x', 'y', 'z']].max().max())

fig = pex.scatter_3d(concat, x='x', y='y', z='z', color='id', range_x=[max, min], range_y=[max, min], range_z=[max, min], labels={'x': 'X links/rechts in meter', 'y': 'Y tiefe in meter', 'z': 'Z höhe in meter'})
fig.show()
