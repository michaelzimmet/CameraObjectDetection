import cv2
import rclpy.serialization
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

bridge = CvBridge()


def cvt_message_to_cv2(message):
    timestamp = message['timestamp']
    data = message['data']
    return deserialize_image(data)


def show_image(image):
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def deserialize_image(message_data):
    img_msg = rclpy.serialization.deserialize_message(message_data, Image)
    return bridge.imgmsg_to_cv2(img_msg, desired_encoding='passthrough')
