from collections import defaultdict
from pathlib import Path
from pprint import pprint

import cv2
import os
import rclpy.serialization
from sensor_msgs.msg import Image
from stereo_msgs.msg import DisparityImage
from cv_bridge import CvBridge

import Config

bridge = CvBridge()


def cvt_message_to_dict(message, deserialize_function):
    data = message['data']
    res = deserialize_function(data)
    return message, res

def show_image(image):
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def deserialize_image(message_data):
    img_msg = rclpy.serialization.deserialize_message(message_data, Image)
    return {'image': bridge.imgmsg_to_cv2(img_msg, desired_encoding='passthrough')}

def deserialize_disparity_image(message_data):
    disparity_msg = rclpy.serialization.deserialize_message(message_data, DisparityImage)

    disparity_image_msg = disparity_msg.image  # Die eigentliche Disparitätskarte
    focal_length = disparity_msg.f  # Brennweite in Pixeln
    baseline = disparity_msg.t  # Basislinie der Stereo-Kameras

    disparity_cv = bridge.imgmsg_to_cv2(disparity_image_msg, desired_encoding='passthrough')

    # Falls die Disparity-Map in 32-Bit Floating-Point gespeichert ist, konvertieren wir sie in 8-Bit für die Anzeige
    disparity_vis = cv2.normalize(disparity_cv, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    return {'image': disparity_vis, 'disparity_map': disparity_cv, 'focal_length': focal_length, 'baseline': baseline}

def query_data(messages: list, deserialize_function):
    data = defaultdict(list)
    for i in messages:
        return_dict = cvt_message_to_dict(i, deserialize_function)[1]
        for k, v in return_dict.items():
            data[k].append(v)
    return data


def play_image_sequence(images, delay=.03):
    window_name = 'Image Sequence'
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    for img in images:
        cv2.imshow(window_name, img)
        if cv2.waitKey(int(delay * 1000)) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

def store_image(image, file: Path) -> None:
    path = Config.IMAGE_DIR / file
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)

def create_video(image_path: Path, video_name: str):
    images = [img for img in os.listdir(image_path) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(image_path, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, 24, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_path, image)))

    cv2.destroyAllWindows()
    video.release()

# if __name__ == '__main__':
#     create_video(Config.IMAGE_DIR / 'db_file_6', 'db_file_6.avi')

