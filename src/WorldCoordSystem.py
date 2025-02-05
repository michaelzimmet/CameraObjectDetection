import numpy as np

def quaternion_to_rotation_matrix(qx, qy, qz, qw):
    """
    Convert a Quaternion into a rotation matrix based on the orientation params from the camera
    """
    # normalize the quaternion
    norm = np.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
    qx /= norm
    qy /= norm
    qz /= norm
    qw /= norm

    return np.array([
        [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw,     1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw,     2*qy*qz + 2*qx*qw,     1 - 2*qx*qx - 2*qy*qy]])


def pixel_to_world(center_x, center_y, depth, px, py, pz, qx, qy, qz, qw, K):
    """
    Transforms a pixel coordinate (center_x, center_y) and a depth value into a 3D point in world coordinates.
    The position change of the camera (px py pz) and the orientation (qx qy qz qw) are also considered.

    :param center_x: center of the bounding box in x direction
    :param center_y: center of the bounding box in y direction
    :param depth: depth value of the object
    :param px, py, pz: new camera position
    :param qx, qy, qz, qw: new camera orientation as quaternion
    :param K: intrinsic camera matrix
    :return: normalized position of the object
    """

    # normalize the pixel coordinates by the intrinsic camera matrix
    p = np.array([center_x, center_y, 1.0])
    p_norm = np.linalg.inv(K) @ p

    # calc the 3D point in camera coordinates
    X_coord = p_norm[0] * depth
    Y_coord = p_norm[1] * depth
    Z_coord = np.sqrt(depth**2 - (X_coord**2 + Y_coord**2))  # depth != Z -> Pythagoras

    pylon_position = np.array([X_coord, Y_coord, Z_coord])

    # create rotation matrix from quaternion
    R = quaternion_to_rotation_matrix(qx, qy, qz, qw)

    # transform point to normalized coordinates depending on the camera change
    camera_position = np.array([px, py, pz])
    normalized_pylon_position = R @ pylon_position + camera_position

    return normalized_pylon_position

def get_3d_position(bbox, depth_map, K, px, py, pz, qx, qy, qz, qw):
    """
    Calculates a normalized 3D position of an object based on a bounding box and a depth map.
    """
    x_top_left, y_top_left, x_bottom_right, y_bottom_right = map(round, bbox)

    # Replace coordinates, if they are out of the image
    x_top_left, x_bottom_right = max(0, x_top_left), min(depth_map.shape[1] - 1, x_bottom_right)
    y_top_left, y_bottom_right = max(0, y_top_left), min(depth_map.shape[0] - 1, y_bottom_right)


    # create a list of all pixels in the bounding box excluding NaN and Inf values
    depth_values = depth_map[y_top_left:y_bottom_right, x_top_left:x_bottom_right].flatten()
    depth_values = depth_values[(depth_values > 0) &
                                (~np.isnan(depth_values)) &
                                (~np.isinf(depth_values))]
    if len(depth_values) == 0:
        return None

    # calculate the median depth value for stability
    depth_value = np.median(depth_values)

    # calc the center of the bounding box
    center_x, center_y = (x_top_left + x_bottom_right) // 2, (y_top_left + y_bottom_right) // 2

    normalized_pylon_position = pixel_to_world(center_x, center_y, depth_value, px, py, pz, qx, qy, qz, qw, K)
    return normalized_pylon_position