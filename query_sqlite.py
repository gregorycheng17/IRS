import sqlite3
from sqlite3 import Error
import sys
sys.path.append(r'D:\Users\Gregory\GitHub\SQLite3_Tutorial')
import CreateDB as CDB


def select_task_by_priority(conn):

    # work in checking date validity
    # Check whether a valid date was imported
    query1='''
    SELECT
    ForwardStartDate, JULIANDAY(ForwardStartDate), date(ForwardStartDate), time(ForwardStartDate)
    FROM
    IRS;
    '''

    #show columns datatype
    query_col_data='''PRAGMA table_info('IRS');'''

    #todate
    query_time= """select datetime('now','localtime')"""

    #first few datas
    query_first_few_data= """SELECT * FROM IRS LIMIT 5;"""

    #between date
    query_date_difference='''
    SELECT
    ForwardStartDate, ForwardStartDateIndex, CloseDate, Maturity,JULIANDAY(ForwardStartDate) - JULIANDAY(CloseDate) AS date_difference
    FROM IRS 
    WHERE (date_difference < 10);
    '''

    #count
    query_count='''
    SELECT
    COUNT(*)
    FROM IRS ;    
    '''

    #between date
    query_='''
    SELECT
    Currency,Curve, CloseDate, ForwardStartDate, Maturity, Implied
    FROM IRS 
    WHERE (ForwardStartDateIndex=1 OR ForwardStartDateIndex=2);
    '''

    cur = conn.cursor()
    cur.execute(query_count)

    rows = cur.fetchall()

    for row in rows:
        print(row)
    print(type(row[0]))

if __name__ == '__main__':

    database = r"D:\Users\Gregory\GitHub\IRS\IRS_roll.db"

    # create a database connection
    conn = CDB.create_connection(database)
    with conn:
        print("1. Query task:")
        select_task_by_priority(conn)
        conn.close
