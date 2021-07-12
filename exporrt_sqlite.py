import sqlite3
from sqlite3 import Error
import pandas as pd
import sys
sys.path.append(r'D:\Users\Gregory\GitHub\SQLite3_Tutorial')
import CreateDB as CDB

if __name__ == '__main__':

    database = r"D:\Users\Gregory\GitHub\IRS\IRS_roll.db"

    # create a database connection
    conn = CDB.create_connection(database)

    query = '''
    SELECT
    Currency,Curve, CloseDate, ForwardStartDate, Maturity, Implied
    FROM IRS 
    WHERE (ForwardStartDateIndex=1 OR ForwardStartDateIndex=2);
    '''

    with conn:
        print("read it in df")
        futures_df = pd.read_sql_query(query, conn)
        print(futures_df)
