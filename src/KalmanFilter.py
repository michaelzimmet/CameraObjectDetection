import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

class KalmanFilter:
    def __init__(self):
        self.tracks = defaultdict(lambda: {"position": None, "velocity": None})
        self.delta_time = 1

    def update(self, object_id, new_position):
        """
        Refresh the Position of the object based on the old and new position
        velocity is the calculated by the kinematic equation -> velocity= (new_position - current_position) / delta_time
        :param object_id:
        :param new_position: new position of the object if its moving. Contains x and y coordinates
        :return: None
        """
        if object_id in self.tracks:
            current_position = self.tracks[object_id]["position"]
            if current_position is not None:
                velocity = (np.array(new_position) - np.array(current_position)) / self.delta_time
                self.tracks[object_id]["velocity"] = velocity
        self.tracks[object_id]["position"] = new_position


    def predict(self, object_id):
        """
        Predict the next position of the object based on the current position and velocity.
        In case velocity is None, the current position will be returned
        :param object_id: unique id for a tracked Object
        :return: predicted position of the object
        """
        if self.tracks[object_id]["velocity"] is not None:
            return np.array(self.tracks[object_id]["position"]) + (self.tracks[object_id]["velocity"] * self.delta_time)
        return self.tracks[object_id]["position"]


# YOLOv8 Model laden
model = YOLO("../res/last.pt")
kf = KalmanFilter()
cap = cv2.VideoCapture("../res/videos/db_file_6.mp4")

tracks = {}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO-Detektion
    results = model(frame)
    detections = results[0].boxes.data.cpu().numpy()

    for det in detections:
        x_top_left, y_top_left, x_bottom_right, y_bottom_right, confidence, class_num = det

        # Only process tracking if confidence is above a certain factor
        if confidence > 0.6:
            center_x, center_y = (x_top_left + x_bottom_right) / 2, (y_top_left + y_bottom_right) / 2

            assigned_id = None
            min_dist = float("inf")

            for track_id, track in tracks.items():
                pred_x, pred_y = kf.predict(track_id)

                # euclidean distance -> sqrt((center_x-pred_x)^2 + (center_y-pred_y)^2)
                dist = np.linalg.norm([center_x - pred_x, center_y - pred_y])

                # distance threshold in pixel
                if dist < 70:
                    assigned_id = track_id
                    min_dist = dist

            # Start a new track if no track is assigned
            if assigned_id is None:
                assigned_id = len(tracks)

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
