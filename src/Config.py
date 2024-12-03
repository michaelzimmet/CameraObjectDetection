from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
SRC_DIR = ROOT_DIR / 'src'
RES_DIR = ROOT_DIR / 'res'
SQL_DIR = RES_DIR / 'sql'

# Configs for rosConverter
ROS_CONVERTER_DIR = SRC_DIR / 'rosConverter'
PARAMETERS_YML = RES_DIR / 'parameters.yml'
SQL_SELECT_ROWS_BY_TOPICNAME = SQL_DIR / 'select_rows_by_topicname.sql'