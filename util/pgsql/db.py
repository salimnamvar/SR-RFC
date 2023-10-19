""" PostgreSQL Database Utilities

"""


# region Imported Dependencies
import psycopg2
# endregion Imported Dependencies


def exist_db(a_conn: psycopg2.connect, a_db_name: str) -> bool:
    with a_conn.cursor() as cursor:
        cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s;", (a_db_name,))
        database_exists = cursor.fetchone()
    return database_exists


def create_db(a_db_params: dict):
    # Create connection
    conn = psycopg2.connect(user=a_db_params['user'], password=a_db_params['password'], host=a_db_params['host'],
                            port=a_db_params['port'])
    conn.autocommit = True

    # Check if the database exists
    database_exists = exist_db(a_conn=conn, a_db_name=a_db_params['database'])

    # Create the database
    if not database_exists:
        with conn.cursor() as cursor:
            print(f"Database '{a_db_params['database']}' does not exist. Creating the new database.")
            cursor.execute(f"CREATE DATABASE {a_db_params['database']};")
            cursor.execute(f"GRANT ALL PRIVILEGES ON DATABASE {a_db_params['database']} TO {a_db_params['user']}")
            cursor.close()

    # Close the connection
    conn.autocommit = False
    conn.close()
