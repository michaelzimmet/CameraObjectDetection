from pathlib import Path

import yaml

from CV2Utilities import cvt_message_to_cv2, show_image
from DatabaseUtilities import SQLiteConnection, load_sql_template


def main():
    config_file = Path('parameters.yml')
    config = read_config_file(config_file)

    con = SQLiteConnection(config['database_files']['db_file_1'])

    params = {'topic_name': config['topics']['image_rect_color']}
    sql_query = load_sql_template(Path('sql/select_rows_by_topicname.sql'), params)
    print(sql_query)
    messages = con.execute_select(sql_query, fetch_num=1, params=params)
    show_image(cvt_message_to_cv2(messages))

def read_config_file(file: Path):
    with file.open('r') as f:
        config = yaml.safe_load(f)
    return config


if __name__ == '__main__':
    main()