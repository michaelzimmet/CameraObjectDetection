import cv2
import rclpy.serialization
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

bridge = CvBridge()


def cvt_message_to_cv2(message):
    data = message['data']
    return message, deserialize_image(data)



def show_image(image):
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def deserialize_image(message_data):
    img_msg = rclpy.serialization.deserialize_message(message_data, Image)
    return bridge.imgmsg_to_cv2(img_msg, desired_encoding='passthrough')

def query_video_images(messages: list):
    images = []
    for i in messages:
        images.append(cvt_message_to_cv2(i)[1])

    return images


def play_image_sequence(images, delay=.03):
    window_name = 'Image Sequence'
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    for img in images:
        cv2.imshow(window_name, img)
        if cv2.waitKey(int(delay * 1000)) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()