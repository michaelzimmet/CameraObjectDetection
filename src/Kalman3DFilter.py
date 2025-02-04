import glob
import os

import cv2
import numpy as np
from filterpy.kalman import KalmanFilter
from ultralytics import YOLO


class Kalman3D:
    def __init__(self):
        """Erstellt einen 3D-Kalman-Filter für (x, y, z) Tracking."""
        self.kf = KalmanFilter(dim_x=6, dim_z=3)

        self.kf.F = np.array([[1, 0, 0, .07, 0,  0],
                              [0, 1, 0, 0,  .07, 0],
                              [0, 0, 1, 0,  0,  .07],
                              [0, 0, 0, 1,  0,  0],
                              [0, 0, 0, 0,  1,  0],
                              [0, 0, 0, 0,  0,  1]])

        self.kf.H = np.array([[1, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0]])

        self.kf.R *= 0.1
        self.kf.Q = np.eye(6) * 0.01
        self.kf.P *= 10

    def initialize(self, x, y, z):
        """Setzt die Anfangswerte des Kalman-Filters."""
        self.kf.x[:3] = np.array([[x], [y], [z]])
        self.kf.x[3:] = np.array([[0], [0], [0]])

    def predict(self):
        """Gibt die vorhergesagte Position (x, y, z) zurück."""
        self.kf.predict()
        return self.kf.x[:3].flatten()

    def update(self, measurement):
        """Aktualisiert den Kalman-Filter mit einer neuen Messung (x, y, z)."""
        self.kf.update(np.array(measurement).reshape(3, 1))


def get_3d_position(bbox, depth_map):
    x1, y1, x2, y2 = map(int, bbox)

    x1, x2 = max(0, x1), min(depth_map.shape[1] - 1, x2)
    y1, y2 = max(0, y1), min(depth_map.shape[0] - 1, y2)

    depth_values = depth_map[y1:y2, x1:x2].flatten()
    depth_values = depth_values[(depth_values > 0) & (~np.isnan(depth_values)) & (~np.isinf(depth_values))]

    if len(depth_values) == 0:
        return None

    depth_value = np.median(depth_values)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    return np.array([cx, cy, depth_value])


def extract_number(filename):
    """Extrahiert die Zahl aus dem Dateinamen."""
    return int(''.join(filter(str.isdigit, os.path.basename(filename))))

model = YOLO("../res/last.pt")
kf3d_tracker = {}

image_folder = "../res/images/db_file_8/"
depth_folder = "../res/images/db_file_8/depth_registered/"

images = sorted(glob.glob(image_folder + "*.png"), key=extract_number)
depth_maps = sorted(glob.glob(depth_folder + "*.tiff"), key=extract_number)

for idx, image in enumerate(images):
    frame = cv2.imread(image)
    depth_map = cv2.imread(depth_maps[idx], cv2.IMREAD_UNCHANGED).astype(np.float32)

    results = model(frame)
    detections = results[0].boxes.data.cpu().numpy()

    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        if conf > 0.6:
            bbox = (x1, y1, x2, y2)
            pos = get_3d_position(bbox, depth_map)

            if pos is not None:
                assigned_id = None
                min_dist = float("inf")

                for track_id, tracker in kf3d_tracker.items():
                    pred_pos = tracker.predict()
                    dist = np.linalg.norm(pred_pos - pos)

                    if dist < 20:
                        assigned_id = track_id
                        min_dist = dist

                if assigned_id is None:
                    assigned_id = len(kf3d_tracker)
                    kf3d_tracker[assigned_id] = Kalman3D()
                    kf3d_tracker[assigned_id].initialize(*pos)

                kf3d_tracker[assigned_id].update(pos)

                cv2.putText(frame, f"ID {assigned_id} Z:{pos[2]:.2f}m", (int(pos[0]), int(pos[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.circle(frame, (int(pos[0]), int(pos[1])), 5, (0, 255, 0), -1)

    cv2.imshow("Kalman3D Tracking", frame)
    if cv2.waitKey(30) & 0xFF == 27:
        break

cv2.destroyAllWindows()
