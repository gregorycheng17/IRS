import pandas as pd
from pathlib import Path
import os
#Reformat before loading into sqlite, e.g. date format.

if __name__ == '__main__':

    directory_in_str=r"D:\Users\Gregory\GitHub\IRS\input"
    directory_out_str = r"D:\Users\Gregory\GitHub\IRS\meta"

    with os.scandir(directory_in_str) as it:
        for entry in it:
            if entry.name.endswith(".csv") and entry.is_file():
                print(entry.name, entry.path)

                df=pd.DataFrame()
                #change date to a better format before import into sqlite
                df = pd.read_csv(entry.path)
                df['Close Date'] = pd.to_datetime(df['Close Date'],dayfirst=True)
                df['Forward Start Date'] = pd.to_datetime(df['Forward Start Date'],dayfirst=True) #dayfirst means input is DDMMYYYY
                df.drop(df.columns[0],axis=1,inplace=True)
                df.to_csv(directory_out_str+"\\"+entry.name,index=False)  #removed index by default
                #print(df[['Close Date','Forward Start Date']].head(5)) ##correct should be April... not Jan vs Feb

