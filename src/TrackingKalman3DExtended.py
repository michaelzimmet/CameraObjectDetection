import glob
import os

import cv2
import matplotlib
import numpy as np
from matplotlib import pyplot as plt, cm
from ultralytics import YOLO

import Config
from Kalman3DExtendedFilter import Kalman3D
from WorldCoordSystem import get_3d_position

def extract_number(filename):
    """Extrahiert die Zahl aus dem Dateinamen."""
    return int(''.join(filter(str.isdigit, os.path.basename(filename))))

def get_poses(position_path, orientation_path):
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

    return [a + b for a, b in zip(positions, orientations)]

model = YOLO("../res/last.pt")
kf3d_tracker = {}

image_folder = "../res/images/db_file_6/left_image_rect_color/"
depth_folder = "../res/images/db_file_6/depth_registered/"

images = sorted(glob.glob(os.path.join(image_folder, "*.tiff")), key=extract_number)
depth_maps = sorted(glob.glob(os.path.join(depth_folder, "*.tiff")), key=extract_number)
poses = get_poses(Config.IMAGE_DIR / 'db_file_6' / 'pose' / 'pose_position.txt', Config.IMAGE_DIR / 'db_file_6' / 'pose' / 'pose_orientation.txt')

K = np.array([[480.3142395, 0., 328.18130493],
              [0., 480.3142395, 181.9155426],
              [0., 0., 1.]], dtype=float)

for idx, image_path in enumerate(images):
    frame = cv2.imread(image_path)
    depth_map = cv2.imread(depth_maps[idx], cv2.IMREAD_UNCHANGED).astype(np.float32)
    # Hole die Kamerapose für diesen Frame (simuliert)
    px, py, pz, qx, qy, qz, qw = poses[idx]

    results = model(frame)
    detections = results[0].boxes.data.cpu().numpy()

    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        if conf > 0.8:
            bbox = (x1, y1, x2, y2)
            pos_world = get_3d_position(bbox, depth_map, K, px, py, pz, qx, qy, qz, qw)
            if pos_world is not None:
                assigned_id = None
                min_dist = float("inf")
                # Finde einen passenden Track anhand der euklidischen Distanz
                for track_id, tracker in kf3d_tracker.items():
                    pred_pos = tracker.predict()
                    dist = np.linalg.norm(pred_pos - pos_world)
                    if dist < 1.0 and dist < min_dist:  # Schwellwert (20) ggf. anpassen
                        min_dist = dist
                        assigned_id = track_id
                # Starte einen neuen Track, falls keiner passt
                if assigned_id is None:
                    assigned_id = len(kf3d_tracker)
                    kf3d_tracker[assigned_id] = Kalman3D()
                    kf3d_tracker[assigned_id].initialize(*pos_world)
                else:
                    kf3d_tracker[assigned_id].update(pos_world)

                # Optional: Visualisierung der Detektion (zeige Bounding Box und ID im Bild)
                cv2.putText(frame, f"ID {assigned_id} Z:{pos_world[2]:.2f}m", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    cv2.imshow("Kalman3D Tracking (Weltkoordinaten)", frame)
    if cv2.waitKey(30) & 0xFF == 27:
        break

cv2.destroyAllWindows()
