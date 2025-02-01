import os
import sys
from tqdm import tqdm

from rosConverter.BaseUtils import read_config_file
from rosConverter.CV2Utilities import deserialize_disparity_image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import Config

from pathlib import Path
from DatabaseUtilities import query_topic_data
from CV2Utilities import (play_image_sequence,
                          store_image)

# Class to extract Images from a ROS-Bag sqlite database with topic type DisparityImage:
#/zed/zed_node/disparity/disparity_image

def main():
    config = read_config_file(Config.PARAMETERS_YML)

    db_file_param = 'db_file_8'
    topic_name = 'right_image_rect_color'

    data = query_topic_data(config, db_file_param, topic_name, deserialize_disparity_image)

    play_image_sequence(data['image'])

    data_path = Path(f'{db_file_param}/{topic_name}')
    for i, img in tqdm(enumerate(data['image'], start=1), desc='Processing'):
        store_image(img, data_path / f'img_{i}.png')

    with open(data_path / 'focal_length.txt', 'w') as file:
        for item in data['focal_length']:
            file.write(f"{item}\n")

    with open(data_path / 'baseline.txt') as file:
        for item in data['baseline']:
            file.write(f"{item}\n")

if __name__ == '__main__':
    main()