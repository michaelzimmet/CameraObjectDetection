import numpy as np

def quaternion_to_rotation_matrix(qx, qy, qz, qw):
    """
    Konvertiert ein Quaternion (qx, qy, qz, qw) in eine 3x3 Rotationsmatrix.
    """
    # Normiere das Quaternion
    norm = np.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
    qx /= norm
    qy /= norm
    qz /= norm
    qw /= norm
    # Standardformel für die Rotationsmatrix
    R = np.array([
        [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw,     1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw,     2*qy*qz + 2*qx*qw,     1 - 2*qx*qx - 2*qy*qy]
    ])
    return R


def pixel_to_world(center_x, center_y, depth, px, py, pz, qx, qy, qz, qw, K):
    """
    Transformiert einen Pixelpunkt (u, v) und einen Tiefenwert in einen 3D-Weltpunkt.

    :param center_x: center of the bounding box in x direction
    :param center_y: center of the bounding box in y direction
    :param depth: Tiefenwert an dieser Stelle (in Metern)
    :param px, py, pz: Kamera-Position in Weltkoordinaten
    :param qx, qy, qz, qw: Kamera-Orientierung als Quaternion
    :param K: Intrinsische Kameramatrix (3x3)
    :return: 3D-Punkt in Weltkoordinaten als np.array([x, y, z])
    """

    # 1️⃣ Pixel in normierte Kamerakoordinaten transformieren
    p = np.array([center_x, center_y, 1.0])
    p_norm = np.linalg.inv(K) @ p  # Normierte Koordinaten

    # 2️⃣ Berechnung der 3D-Kamerakoordinaten mit Pythagoras-Korrektur
    X_cam = p_norm[0] * depth
    Y_cam = p_norm[1] * depth
    Z_cam = np.sqrt(depth**2 - (X_cam**2 + Y_cam**2))  # Z-Korrektur

    pylon_position = np.array([X_cam, Y_cam, Z_cam])

    # 3️⃣ Konvertiere das Quaternion in eine Rotationsmatrix
    R = quaternion_to_rotation_matrix(qx, qy, qz, qw)

    # 4️⃣ Transformiere den Punkt ins Weltkoordinatensystem
    camera_position = np.array([px, py, pz])
    normalized_pylon_position = R @ pylon_position + camera_position

    return normalized_pylon_position

def get_3d_position(bbox, depth_map, K, px, py, pz, qx, qy, qz, qw):
    """
    Berechnet aus einer Bounding Box und der zugehörigen Depth Map den 3D-Punkt in Weltkoordinaten.

    Schritte:
      - Beschränkung der Bounding Box auf die Bildgrenzen.
      - Extraktion der relevanten Tiefenwerte und Berechnung des Medianwerts.
      - Berechnung des Mittelpunkts (Pixelkoordinaten) der Bounding Box.
      - Transformation in Weltkoordinaten mittels pixel_to_world.
    """
    x_top_left, y_top_left, x_bottom_right, y_bottom_right = map(round, bbox)
    # Beschränke die Bounding Box auf die Bildgrenzen
    x_top_left, x_bottom_right = max(0, x_top_left), min(depth_map.shape[1] - 1, x_bottom_right)
    y_top_left, y_bottom_right = max(0, y_top_left), min(depth_map.shape[0] - 1, y_bottom_right)


    depth_values = depth_map[y_top_left:y_bottom_right, x_top_left:x_bottom_right].flatten()
    depth_values = depth_values[(depth_values > 0) &
                                (~np.isnan(depth_values)) &
                                (~np.isinf(depth_values))]
    if len(depth_values) == 0:
        return None

    depth_value = np.median(depth_values)
    # Berechne den Mittelpunkt der Bounding Box (Pixelkoordinaten)
    center_x, center_y = (x_top_left + x_bottom_right) // 2, (y_top_left + y_bottom_right) // 2

    # Transformation in Weltkoordinaten
    normalized_pylon_position = pixel_to_world(center_x, center_y, depth_value, px, py, pz, qx, qy, qz, qw, K)
    return normalized_pylon_position