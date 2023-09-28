""" PostgreSQL-Pandas Utilities

"""


# region Imported Dependencies
import pandas as pd
import psycopg2
from tqdm import tqdm
# endregion Imported Dependencies


def insert_rows(a_conn: psycopg2.connect, a_tbl_name: str, a_df: pd.DataFrame):
    with a_conn.cursor() as cursor:
        for _, row in tqdm(a_df.iterrows(), desc='Inserting rows'):
            insert_query = f'INSERT INTO {a_tbl_name} VALUES ({", ".join(["%s"] * len(row))});'
            cursor.execute(insert_query, tuple(row))
