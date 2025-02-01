import os
import sys
from pprint import pprint
from tqdm import tqdm

from BaseUtils import read_config_file

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import Config

from pathlib import Path
from DatabaseUtilities import query_topic_data
from CV2Utilities import (play_image_sequence,
                          store_image, deserialize_image)

# Class to extract Images from a ROS-Bag sqlite database with topic type image:
# /zed/zed_node/right/image_rect_color
# /zed/zed_node/right_gray/image_rect_gray
# /zed/zed_node/stereo/image_rect_color
# /zed/zed_node/left/image_rect_color
# /zed/zed_node/stereo_raw/image_raw_color
# /zed/zed_node/depth/depth_registered
# /zed/zed_node/confidence/confidence_map
# /zed/zed_node/left_gray/image_rect_gray
# /zed/zed_node/rgb/image_rect_color
# /zed/zed_node/rgb_gray/image_rect_gray


def main():
    config = read_config_file(Config.PARAMETERS_YML)

    db_file_param = 'db_file_8'
    topic_name = 'depth_registered'

    data = query_topic_data(config, db_file_param, topic_name, deserialize_image)

    play_image_sequence(data['image'])

    pprint(data['image'][0])
    for i, img in tqdm(enumerate(data['image'], start=1), desc='Processing'):
        store_image(img, Path(f'{db_file_param}/{topic_name}/img_{i}.tiff'))


if __name__ == '__main__':
    main()