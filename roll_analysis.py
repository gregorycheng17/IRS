#IRS Spread Analytics
#04Aug2020 - Debug so that all data would be available in the benchmark reports. (At first, some benchmark date
#bank holiday so the date was dropped). Now if date is not avaiable, the first biz day after that day would be picked.

##
import sqlite3
from sqlite3 import Error
import sys
sys.path.append(r'D:\Users\Gregory\GitHub\SQLite3_Tutorial')
import CreateDB as CDB

#from newdir import newdirectory
import os
import time, datetime, os #time
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cbook as cbook
import seaborn as sns

database = r"D:\Users\Gregory\GitHub\IRS\IRS_roll.db"
directory_in_str = r"D:\Users\Gregory\GitHub\IRS\meta"
directory_out_str = r"D:\Users\Gregory\GitHub\IRS\output"
Pos_dir=r"D:\Users\Gregory\GitHub\IRS\input"


#customisation
benchmark=14 #14 days from fix

#Assumption
pd.set_option("display.precision", 10)

# create a database connection
conn = CDB.create_connection(database)

query = '''
SELECT
Currency,Curve, CloseDate, ForwardStartDate, ForwardStartDateIndex, Maturity, Implied
FROM IRS
WHERE (ForwardStartDateIndex=1 OR ForwardStartDateIndex=2);
'''
df = pd.read_sql_query(query, conn)
conn.close

#Live Portfolio position to read
IRSPosition=os.path.join(Pos_dir, "Pos.xlsx")  #File to EMS Portfolio Position.... i.e Actual Positions, which are ex-FOX
metaIRSPos=pd.read_excel(IRSPosition)

#Variable
years = mdates.YearLocator()   # every year
months = mdates.MonthLocator(bymonth=None, interval=3)  # occurance every thress month
yearsFmt = mdates.DateFormatter('%Y')
monthsFmt = mdates.DateFormatter('%b') # %b mean Jan, Feb, etc.

#Data Massage - IRS Data
#Tables: a) metatable; b) irs_metatable; c)IMM, d)IMM1, e)IMM2, f)RollTable, g)result (which overall all before)

metatable=df
#metatable=metatable.iloc[:,1:] #remove 1st coln
metatable['CloseDate'] = pd.to_datetime(metatable['CloseDate'])
metatable['ForwardStartDate'] = pd.to_datetime(metatable['ForwardStartDate'])
#New Col - Ticker
metatable["Ticker"]=metatable["Curve"]+"_"+metatable["Maturity"]

#Filter out NDF only
metatable=metatable[metatable["CloseDate"]>'2017-01-01'] #downsize sample
##

#Filter out irs again by IMM fwd date (i.e. VD) this time
cond1=metatable['ForwardStartDateIndex']!=0
irs_metatable=metatable[(cond1)]

#Add Col: Days2Roll
irs_metatable['RemainDays']=(irs_metatable['ForwardStartDate']-irs_metatable['CloseDate']).dt.days
irs_metatable=irs_metatable.sort_values(by=['Ticker','CloseDate','ForwardStartDate']) #Reorder the fixing date

#Create IMM1 and IMM2 table which are for Spread Calculation
#IMM=irs_metatable.groupby(['Ticker','Close Date']).head(2) #Keep first two dates, only IMM1 and IMM2 of each value day

condimm1=irs_metatable['ForwardStartDateIndex']==1
condimm2=irs_metatable['ForwardStartDateIndex']==2

IMM1=irs_metatable[(condimm1)]   #Near Date....x
IMM2=irs_metatable[(condimm2)]   #Far Date....y
IMM2=IMM2[['Ticker','CloseDate','ForwardStartDate','Implied']]  #Only kept useful Columns

##
#Merge DFs, Add Coln, Reorder and Eliminate some Col
RollTable=pd.merge(IMM1, IMM2, how='inner', on=['Ticker', 'CloseDate'])
RollTable['RollingPt']=RollTable['Implied_y']-RollTable['Implied_x'] #Far-Near
RollTable['RollingBps']=RollTable['RollingPt']*10000

## ccy pair, valuetime, fixingdate, valuex, valuey, rollingpoint, rollingpip
RollTable=RollTable[['Ticker','CloseDate','ForwardStartDate_x','ForwardStartDate_y','RemainDays','Maturity'
                     ,'Implied_x','Implied_y','RollingBps']] #Only useful Columns
RollTable.columns=['Ticker','ValueDate','FSDate1','FSDate2','LD_to_Roll','Maturity'
                     ,'Value1','Value2','RollingBps'] #Rename the colns
#ValueDate should be renamed to SpotDate

##
RollTable['roll_pt_diff'] = RollTable.groupby(['Ticker','FSDate1'])['RollingBps'].apply(lambda x: x.diff())
RollTable['BmkD1']=RollTable['FSDate1']- np.timedelta64(benchmark,'D') #Get the benchmark date #obsolete but keep first

#Get last day before IMM1 FX rate
DailyPnL=RollTable.groupby(['Ticker','FSDate1','ValueDate'],as_index=True).first() #First and Last is the same here.
lastValueday=DailyPnL.groupby(level=['Ticker','FSDate1']).tail(1) #All currencies' value day which is closest to the fixing, FS1
IRSRate=lastValueday['Value1'] #of each pair, every fix, found the closest outright e.g. 13Mar20' rate with fix(FS1) 16Mar20
##
#Get bmk date and rates
#select the first row within benchmark date interval... so picked the first date on benchmark date
#or the day after (if on holiday)
bmk_VD=RollTable[RollTable['LD_to_Roll']<=benchmark].groupby(['Ticker','FSDate1'],as_index=True).first()
bmk_VD = bmk_VD.set_index('ValueDate', append=True)
bmk_IRSRate=bmk_VD['Value1'] #of each pair, on each IMM, found the rate at fix-benchmark
#Merge DFs
#Rename
#Set index
result = pd.merge(DailyPnL.reset_index(), IRSRate.reset_index(),on=['Ticker','FSDate1'],
                  how='left')        #Merge orig table with last date so to computer PnL in USD notional.

##
result.columns=['Ticker','FSDate1','ValueDate','FSDate2','LD_to_Roll',
                'Maturity','Value1','Value2','RollingBps','roll_pt_diff','BmkD1','Last_Valid_Date',
                'Last_Valid_IRSRate'] #Rename the colns
#Plus benchmark colns
bmk_result = pd.merge(result, bmk_IRSRate.reset_index(),on=['Ticker','FSDate1'],
                  how='left')        #Merge orig table with last date so to computer PnL in USD notional.
bmk_result.columns=['Ticker','FSDate1','ValueDate','FSDate2','LD_to_Roll',
                'Maturity','Value1','Value2','RollingBps','roll_pt_diff','BmkD1','Last_Valid_Date',
                'Last_Valid_IRSRate',
                   'LastBmk_Valid_Date','LastBmk_Valid_IRSRate'] #Rename the colns
bmk_result.set_index(['Ticker','FSDate1','ValueDate'],inplace=True)
#After combined, we have colns of last available dates and rates and colns of benchmark date and rates
#If benchamark date is not available yet, filled with last available dates and rates.
bmk_result.LastBmk_Valid_Date.fillna(bmk_result.Last_Valid_Date, inplace=True)
bmk_result.LastBmk_Valid_IRSRate.fillna(bmk_result.Last_Valid_IRSRate, inplace=True)
#add cols_
bmk_result['PnL_return']=bmk_result['roll_pt_diff']#/bmk_result['Value1']               #PnL not yet PV!!!
#bmk_result['lastBmk_valid_PnL']=bmk_result['roll_pt_diff']/bmk_result['LastBmk_Valid_FXRate']               #PnL not yet PV!!!
#shift before combining with Portfolio
bmk_result['roll_pt_diff']=bmk_result['roll_pt_diff'].shift(-1)
bmk_result['PnL_return']=bmk_result['PnL_return'].shift(-1)
#bmk_result['lastBmk_valid_PnL']=bmk_result['lastBmk_valid_PnL'].shift(-1)
##

#Data Massage- Portfolio File
#Tables: a) metaFXPos
metaIRSPos.set_index('VD',inplace=True)    #Date as index
#metaIRSPos.rename(columns=lambda x: x[1:],inplace=True) #Remove first character 'F' in Prod
metaIRSPos=metaIRSPos.unstack()
metaIRSPos.index.names = ['Ticker','ValueDate'] #Prepare to merge
metaIRSPosd=metaIRSPos.reset_index()
#change order of the name to match another file's name
metadelimit=metaIRSPosd.Ticker.str.split('-',expand=True)
metaIRSPosd.Ticker=metadelimit[1]+'_'+metadelimit[3]+'_'+metadelimit[2]


##Merge DFs
finalresult = pd.merge(bmk_result.reset_index(), metaIRSPosd,on=['Ticker','ValueDate'],
                  how='left')
#Rename
finalresult=finalresult.rename(columns = {0:'Pos'})

#PnL using the available day
finalresult['Pos_D_PnL']=finalresult['Pos']*finalresult['PnL_return']
#Cumsum in reverse using [::-1], -1 step each time for whole series
finalresult['Pos_PnL'] = finalresult.groupby(['Ticker','FSDate1'])['Pos_D_PnL'].apply(lambda x: x[::-1].cumsum()[::-1])

#Create a df to store all cum PnL in Benchmark date.
selected_rows=(finalresult["ValueDate"]==finalresult["LastBmk_Valid_Date"])
selected_columns=['Ticker','FSDate1','Pos_PnL']
benchmark_PnL=finalresult.loc[selected_rows,selected_columns]
##

#Merge DFs
#Rename
final_bmk_result = pd.merge(finalresult, benchmark_PnL,on=['Ticker','FSDate1'],
                  how='left')        #Merge orig table with last date so to computer PnL in USD notional.
final_bmk_result.rename(columns={'Pos_PnL_x':'Cum_Pos_PnL', 'Pos_PnL_y':'Bmk_PnL'},inplace=True)
final_bmk_result['Cum_Bmk_PnL']=final_bmk_result['Cum_Pos_PnL']-final_bmk_result['Bmk_PnL']
## Not yet
##

# Graph1: to plot the rolling graphs
Graph1 = RollTable[['NDF_Pair', 'ValueTime', 'Fwd_Fwd_pip']].dropna()
# Testmode
# Graph1=Graph1[(Graph1['NDF_Pair']=='USDPHP')]

Graph1storage = {}  # to store
timer_start = time.time()

# counter - to change color
num = 0

for key, value in Graph1.groupby(
        ['NDF_Pair']):  # seperate dataframes by column value using group by, the col being groupby becomes key

    Graph1storage[key] = value  # store dataframes in dict

    # Linegraph
    # Data
    x = value['ValueTime'].values
    y = value['Fwd_Fwd_pip'].values

    # Reset setting everytime
    mpl.rcParams.update(mpl.rcParamsDefault)
    # create a color palette
    palette = plt.get_cmap('Set1')

    sns.set()  # seaborn style
    fig, ax = plt.subplots(figsize=(20, 10))  # size of the plot
    ax.plot(x, y, marker='', color=palette(num), linewidth=1.9, alpha=0.9)  # Plot the lineplot

    # format the ticks
    # x-axis
    ax.xaxis.set_major_locator(years)  # major tick
    ax.xaxis.set_major_formatter(yearsFmt)
    ax.tick_params(which='major', length=3, color='k', direction='in', top="off",
                   pad=15)  # pad is distance between tick and label

    ax.xaxis.set_minor_locator(months)  # minor tick
    ax.xaxis.set_minor_formatter(monthsFmt)
    ax.tick_params(which='minor', length=3, color='k', direction='out', top="off")

    # y-axis
    ax.yaxis.set_minor_locator(plt.MaxNLocator(20))
    ax.yaxis.set_major_locator(plt.MaxNLocator(10))

    # format the ticks
    # y-axis
    ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))

    # gridlines
    ax.grid(which='major', color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.grid(which='minor', color='black', linestyle='--', linewidth=0.5, alpha=0.3)

    # Tick min and max
    datemin = np.datetime64(x.min(), 'Y')
    datemax = np.datetime64(x.max(), 'Y') + np.timedelta64(1, 'Y')  # round to nearest years...
    ax.set_xlim(datemin, datemax)
    plt.ylim(y.min() - abs(y.min()) / 20, y.max() + abs(y.max() / 20))

    # Add axis names
    plt.xlabel('Date')
    plt.ylabel('IMM Pip Spread')

    # Get Max and Min value and get in titles
    xmax = x[np.argmax(y)]
    ymax = y.max()
    xmin = x[np.argmin(y)]
    ymin = y.min()

    xmax = pd.to_datetime(str(xmax)).strftime('%Y.%m.%d')
    xmin = pd.to_datetime(str(xmin)).strftime('%Y.%m.%d')

    # Timer on process
    timer_end = time.time()
    speed = int(timer_end - timer_start)

    # Add Title and subtitle
    plt.title("Rolling Spread  of {0} between 1st and 2nd IMM across time".format(key)
              + "\nThe report took {0} seconds to be generated.".format(speed)
              + "\nMax: {:1,.0f} pips on {}".format(ymax, xmax)
              + "\nMin: {:1,.0f} pips on {}".format(ymin, xmin)
              , loc='left', fontsize=12, fontweight=0, color='green', fontstyle='italic')  # Add title to the plot
    plt.suptitle(key, fontsize=18)

    plt.savefig(my_path + '/output/RollingPoint/' + key + '.png')
    plt.clf()  # to clean the memory
    plt.cla()
    plt.close()

    num = num + 10  # counter to change color