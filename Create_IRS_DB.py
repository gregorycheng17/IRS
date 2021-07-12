# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import sys
sys.path.append(r'D:\Users\Gregory\GitHub\SQLite3_Tutorial')
import CreateDB as CDB


if __name__ == '__main__':
    database = r"D:\Users\Gregory\GitHub\IRS\IRS_roll.db"

    # LastTradeDate datetimeoffset(7) NOT NULL,
    # ValueTime datetimeoffset(7),
    sql_create_futures_table = """ CREATE TABLE IF NOT EXISTS IRS (
                                            Currency TEXT NOT NULL,
                                            Curve TEXT NOT NULL,
                                            CloseDate REAL NOT NULL,
                                            ForwardStartDate REAL NOT NULL,
                                            ForwardStartDateIndex INTEGER NOT NULL,
                                            Maturity TEXT NOT NULL,
                                            Implied REAL NOT NULL
                                        ); """

    # create a database connection
    conn = CDB.create_connection(database)

    # create tables
    if conn is not None:
        # create futures table
        CDB.create_table(conn, sql_create_futures_table)

    else:
        print("Error! cannot create the database connection.")