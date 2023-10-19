""" PostgreSQL Table Utilities

"""


# region Imported Dependencies
import psycopg2
# endregion Imported Dependencies


def exist_tbl(a_conn: psycopg2.connect, a_tbl_name: str) -> bool:
    with a_conn.cursor() as cursor:
        cursor.execute(
            f"SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'schema_name' AND table_name = %s);",
            (a_tbl_name,))
        tbl_exists = cursor.fetchone()
    return tbl_exists


def drop_tbl(a_conn: psycopg2.connect, a_tbl_name: str):
    tbl_exists = exist_tbl(a_conn=a_conn, a_tbl_name=a_tbl_name)
    if tbl_exists:
        print(f"Table '{a_tbl_name}' already exists. Dropping the table.")
        # Drop the table
        with a_conn.cursor() as cursor:
            cursor.execute(f"DROP TABLE IF EXISTS {a_tbl_name};")
            a_conn.commit()


def create_tbl(a_conn: psycopg2.connect, a_tbl_name: str, a_columns: str):
    with a_conn.cursor() as cursor:
        create_table_query = f'CREATE TABLE {a_tbl_name} ({a_columns});'
        cursor.execute(create_table_query)
        a_conn.commit()
