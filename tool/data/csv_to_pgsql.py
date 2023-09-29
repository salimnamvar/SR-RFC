""" Convert CSV to PostgreSQL Table

"""


# region Imported Dependencies
import argparse
import pandas as pd
import psycopg2
from util.pgsql.db import create_db
from util.pgsql.mapper import map_type
from util.pgsql.pd import insert_rows
from util.pgsql.tbl import drop_tbl, create_tbl
# endregion Imported Dependencies


# region Sub-Functions
def csv_to_pgsql(a_csv_file: str, a_tbl_name: str, a_db_params: dict):
    # Create database if it does not exist
    create_db(a_db_params=a_db_params)

    # Connect to the database
    conn = psycopg2.connect(**a_db_params)

    # Drop the table
    drop_tbl(a_conn=conn, a_tbl_name=a_tbl_name)

    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(a_csv_file)

    # Create the table
    columns = ", ".join([f'"{col}" {map_type(datatype)}' for col, datatype in zip(df.columns, df.dtypes)])
    create_tbl(a_conn=conn, a_tbl_name=a_tbl_name, a_columns=columns)

    # Insert rows
    insert_rows(a_conn=conn, a_tbl_name=a_tbl_name, a_df=df)

    conn.commit()
    conn.close()
# endregion Sub-Functions


# region Tool
def main():
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Convert a CSV file into a PostgreSQL table.")

    # Define command-line arguments
    parser.add_argument("--csv_file", default="G:/Challenges/RNA/data/sample_submission.csv",
                        help="Path to the CSV file to import.")
    parser.add_argument("--tbl_name", default='submission', help="Name of the PostgreSQL table to create.")
    parser.add_argument("--host", default="localhost", help="Database host (default: localhost).")
    parser.add_argument("--port", default="5432", help="Database port (default: 5432).")
    parser.add_argument("--database", default="rfc", help="Database name.")
    parser.add_argument("--user", default="postgres", help="Database user.")
    parser.add_argument("--password", default="1234", help="Database password.")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Construct the database connection parameters
    db_params = {
        "user": args.user,
        "password": args.password,
        "host": args.host,
        "port": args.port,
        "database": args.database,
    }

    # Call the function to convert CSV to PostgreSQL
    csv_to_pgsql(a_csv_file=args.csv_file, a_tbl_name=args.tbl_name, a_db_params=db_params)


if __name__ == "__main__":
    main()
# endregion Tool
