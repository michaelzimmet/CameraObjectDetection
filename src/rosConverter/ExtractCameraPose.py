import os
import sys

from BaseUtils import read_config_file

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import Config

from DatabaseUtilities import query_topic_data
from CV2Utilities import deserialize_pose

# Class to extract Images from a ROS-Bag sqlite database with topic type PoseStamped:
# /zed/zed_node/pose

def main():
    config = read_config_file(Config.PARAMETERS_YML)

    db_file_param = 'db_file_8'
    topic_name = 'pose'

    data = query_topic_data(config, db_file_param, topic_name, deserialize_pose)

    data_path = Config.IMAGE_DIR / f'{db_file_param}/{topic_name}'
    with open(data_path / 'pose_position.txt', 'w') as file:
        for item in data['position']:
            x, y, z = item.x, item.y, item.z
            file.write(f"{x, y, z}\n")

    with open(data_path / 'pose_orientation.txt', 'w') as file:
        for item in data['orientation']:
            x, y, z, w = item.x, item.y, item.z, item.w
            file.write(f"{x, y, z, w}\n")


if __name__ == '__main__':
    main()