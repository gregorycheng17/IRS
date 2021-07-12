import sqlite3
import csv
import os

if __name__ == '__main__':

    database = r"D:\Users\Gregory\GitHub\IRS\IRS_roll.db"

    directory_in_str=r"D:\Users\Gregory\GitHub\IRS\meta"
    directory_out_str = r"D:\Users\Gregory\GitHub\IRS\output"

    with os.scandir(directory_in_str) as it:
        for entry in it:
            if entry.name.endswith(".csv") and entry.is_file():

                con = sqlite3.connect(database)
                cur = con.cursor()

                a_file = open(entry.path)
                rows = csv.reader(a_file)
                cur.executemany("INSERT INTO IRS VALUES (?, ?, ?, ?, ?, ?, ?)", rows)  #no index

                con.commit()
                con.close