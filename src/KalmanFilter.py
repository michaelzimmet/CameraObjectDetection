import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

# Kalman-Filter funktioniert stabil, wenn die Objekte sich linear bewegen, sobald es jedoch zu Kameradrehungen kommt,
# kann es zu Fehlern kommen, da der Filter nur die Geschwindigkeit der Objekte speichert und keine Richtungsinformationen hat.

# Kalman-Filter Klasse
class KalmanFilter:
    def __init__(self):
        self.tracks = defaultdict(lambda: {"position": None, "velocity": None})

    def update(self, object_id, new_position):
        if object_id in self.tracks:
            last_pos = self.tracks[object_id]["position"]
            if last_pos is not None:
                velocity = np.array(new_position) - np.array(last_pos)
                self.tracks[object_id]["velocity"] = velocity
        self.tracks[object_id]["position"] = new_position

    def predict(self, object_id):
        if self.tracks[object_id]["velocity"] is not None:
            return np.array(self.tracks[object_id]["position"]) + self.tracks[object_id]["velocity"]
        return self.tracks[object_id]["position"]


# YOLOv8 Model laden
model = YOLO("../res/last.pt")
kf = KalmanFilter()
cap = cv2.VideoCapture("../res/videos/db_file_6.mp4")

track_id = 0
tracks = {}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO-Detektion
    results = model(frame)
    detections = results[0].boxes.data.cpu().numpy()

    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        if conf > 0.6:  # Nur sichere Erkennungen tracken
            center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2

            # Falls ein Track existiert, verwende Kalman-Vorhersage
            assigned_id = None
            min_dist = float("inf")

            for t_id, track in tracks.items():
                pred_x, pred_y = kf.predict(t_id)
                dist = np.linalg.norm([center_x - pred_x, center_y - pred_y])

                if dist < 70:  # Falls Track nahe genug ist, übernehme ID
                    assigned_id = t_id
                    min_dist = dist

            if assigned_id is None:  # Falls kein Track passt, neuen Track starten
                assigned_id = track_id
                track_id += 1

            tracks[assigned_id] = (center_x, center_y)
            kf.update(assigned_id, (center_x, center_y))

            # Zeichne Tracking-IDs
            cv2.putText(frame, f"ID {assigned_id}", (int(center_x), int(center_y) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.circle(frame, (int(center_x), int(center_y)), 5, (0, 255, 0), -1)

    cv2.imshow("Tracking", frame)
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
