import sqlite3
from jinja2 import Template
from pathlib import Path

import Config
from rosConverter.CV2Utilities import query_data


class SQLiteConnection:
    def __init__(self, db_file: Path):
        self.db_file = db_file
        self.connection = create_connection(db_file)

    def _execute_query(self, query: str, params=None):
        if not self.connection:
            print('No connection to connect too. First open a connection with : ')

        try:
            self.connection.row_factory = sqlite3.Row
            cursor = self.connection.cursor()
            res = cursor.execute(query, params)
            print('Query executed successfully: ', query)
            return res
        except sqlite3.Error as e:
            print('Error occured during query execution: ', query)
            print('Error: ', e)

    def execute_select(self, query: str, fetch_num: int = 0, params=None):
        res = self._execute_query(query, params)
        if fetch_num == 0:
            return res.fetchall()
        elif fetch_num == 1:
            return res.fetchone()
        else:
            return res.fetchmany(fetch_num)


    def open_connection(self):
        if not self.connection:
            self.connection = create_connection(self.db_file)
        else:
            print('Connection already exists: ', self.db_file)


    def close_connection(self):
        if self.connection:
            self.connection.close()
            print('Connection to SQLite DB closed: ', self.db_file)
        else:
            print('No existing Connection to close: ', self.db_file)

def create_connection(db_file: Path) -> sqlite3.Connection:
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        print('Connected to SQLite DB: ')
        print('SQLite version: ', sqlite3.version)
        print('DB file: ', db_file)
    except sqlite3.Error as e:
        print('Error occured during connection to SQLite DB: ')
        print('DB file: ', db_file)
        print('Error: ', e)
    return conn

def load_sql_template(template_file: Path, params):
    with template_file.open('r') as f:
        template = Template(f.read())
        return template.render(params)


def query_topic_data(config, db_file_param, topic_name, deserialize_function):
    con = SQLiteConnection(config['database_files'][db_file_param])
    params = {'topic_name': config['topics'][topic_name]}
    sql_query = load_sql_template(Config.SQL_SELECT_ROWS_BY_TOPICNAME, params)
    print(sql_query)
    messages = con.execute_select(sql_query, params=params)
    data = query_data(messages, deserialize_function)
    return data