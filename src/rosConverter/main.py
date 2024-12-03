import os
import sys
import yaml
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import Config

from pathlib import Path
from DatabaseUtilities import SQLiteConnection, load_sql_template
from CV2Utilities import query_video_images, play_image_sequence


def main():
    config = read_config_file(Config.PARAMETERS_YML)

    con = SQLiteConnection(config['database_files']['db_file_1'])

    params = {'topic_name': config['topics']['image_rect_color']}
    sql_query = load_sql_template(Config.SQL_SELECT_ROWS_BY_TOPICNAME, params)
    print(sql_query)
    messages = con.execute_select(sql_query, params=params)
    images = query_video_images(messages)
    play_image_sequence(images)



def read_config_file(file: Path):
    with file.open('r') as f:
        config = yaml.safe_load(f)
    return config


if __name__ == '__main__':
    main()