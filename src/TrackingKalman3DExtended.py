import glob
import os

import cv2
import numpy as np
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

image_folder = "../res/images/db_file_6/left_image_rect_color/"
depth_folder = "../res/images/db_file_6/depth_registered/"

images = sorted(glob.glob(os.path.join(image_folder, "*.tiff")), key=extract_number)
depth_maps = sorted(glob.glob(os.path.join(depth_folder, "*.tiff")), key=extract_number)
poses = get_poses(Config.IMAGE_DIR / 'db_file_6' / 'pose' / 'pose_position.txt', Config.IMAGE_DIR / 'db_file_6' / 'pose' / 'pose_orientation.txt')

# Intrinsic camera matrix containing the focal length and center relative to the camera for normalization
K = np.array([[480.3142395,          0., 328.18130493],
                    [         0., 480.3142395,  181.9155426],
                    [         0.,          0.,          1.]], dtype=float)


for idx, image_path in enumerate(images):
    frame = cv2.imread(image_path)
    depth_map = cv2.imread(depth_maps[idx], cv2.IMREAD_UNCHANGED).astype(np.float32)
    px, py, pz, qx, qy, qz, qw = poses[idx]

    results = model(frame)
    detections = results[0].boxes.data.cpu().numpy()

    for det in detections:
        x_top_left, y_top_left, x_bottom_right, y_bottom_right, confidence, class_num = det

        # Only process tracking if confidence is above a certain factor
        if confidence > 0.6:
            bounding_box = (x_top_left, y_top_left, x_bottom_right, y_bottom_right)
            normalized_pylon_position = get_3d_position(bounding_box, depth_map, K, px, py, pz, qx, qy, qz, qw)

            if normalized_pylon_position is not None:

                assigned_id = None
                min_dist = float("inf")

                # Loop through all tracks to find a matching track
                for track_id, tracker in tracks.items():
                    pred_pos = tracker.predict()
                    dist = np.linalg.norm(pred_pos - normalized_pylon_position)

                    # Distance threshold
                    if dist < 1.0 and dist < min_dist and tracker.pred_class == class_num:
                        min_dist = dist
                        assigned_id = track_id

                # create a new track if no suitable track was found
                if assigned_id is None:
                    assigned_id = len(tracks)
                    tracks[assigned_id] = Kalman3D(class_num)
                    tracks[assigned_id].initialize(*normalized_pylon_position)
                else:
                    tracks[assigned_id].update(normalized_pylon_position)

                cv2.putText(frame, f"ID {assigned_id} Z:{normalized_pylon_position[2]:.2f}m", (int(x_top_left), int(y_top_left) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.rectangle(frame, (int(x_top_left), int(y_top_left)), (int(x_bottom_right), int(y_bottom_right)), (0, 255, 0), 2)

    cv2.imshow("Kalman3D Tracking (Weltkoordinaten)", frame)
    if cv2.waitKey(30) & 0xFF == 27:
        break

cv2.destroyAllWindows()
