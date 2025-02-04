import os
import sys

from BaseUtils import read_config_file

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import Config

from DatabaseUtilities import query_topic_data
from CV2Utilities import deserialize_camera_info

# Class to extract Images from a ROS-Bag sqlite database with topic type CameraInfo:
#/zed/zed_node/right_raw/camera_info
#/zed/zed_node/right/camera_info
#/zed/zed_node/left/camera_info
#/zed/zed_node/rgb_raw/camera_info
#/zed/zed_node/depth/camera_info
#/zed/zed_node/left_raw/camera_info
#/zed/zed_node/rgb/camera_info


def main():
    config = read_config_file(Config.PARAMETERS_YML)

    db_file_param = 'db_file_8'
    topic_name = 'left_camera_info'

    data = query_topic_data(config, db_file_param, topic_name, deserialize_camera_info)

    data_path = Config.IMAGE_DIR / f'{db_file_param}/{topic_name}'
    with open(data_path / 'intrinsic_matrix.txt', 'w') as file:
        for item in data['K']:
            print(item)
            file.write(f"{item}\n")

    with open(data_path / 'distortion_coefficient.txt', 'w') as file:
        for item in data['D']:
            print(item)
            file.write(f"{item}\n")


if __name__ == '__main__':
    main()