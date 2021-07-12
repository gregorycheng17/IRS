#IES Spread Analytics
#04Aug2020 - Debug so that all data would be available in the benchmark reports. (At first, some benchmark date
#bank holiday so the date was dropped). Now if date is not avaiable, the first biz day after that day would be picked.

from newdir import newdirectory
import os
import time, datetime, os #time
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cbook as cbook
import seaborn as sns

#customisation
benchmark=14 #14 days from fix

#Assumption
pd.set_option("display.precision", 10)

#Directory
my_path=os.path.join("K:\\TheGregStation\IRS\RollAnalysis"+ os.sep) # Change directory
#output_path=os.path.join(my_path, "Health_Check"+ os.sep) # Report Directory
#Backup_path=os.path.join(output_path, "Archive"+ os.sep)# Backup Directory

#FX Point Data File to Read
input_path=os.path.join(my_path, "prices_out.csv")  #File from DB FX Data
todaystr = datetime.datetime.today().strftime('%Y_%m_%d')
metatable=pd.read_csv(input_path)

'''
#Live Portfolio position to read
FXPosition=os.path.join(my_path, "FXPosition.xlsx")  #File to EMS Portfolio Position.... i.e Actual Positions, which are ex-FOX
metaFXPos=pd.read_excel(FXPosition)
'''

#Variable
years = mdates.YearLocator()   # every year
months = mdates.MonthLocator(bymonth=None, interval=3)  # occurance every thress month
yearsFmt = mdates.DateFormatter('%Y')
monthsFmt = mdates.DateFormatter('%b') # %b mean Jan, Feb, etc.

#Data Massage - FX Data
#Tables: a) metatable; b) irs_metatable; c)IMM, d)IMM1, e)IMM2, f)RollTable, g)result (which overall all before)

'''
#New Df, combined with inputfile later
pip = [{'USDBRL':10000,'USDCNY': 10000,'USDMYR': 10000,'USDRUB': 10000,'USDPEN': 10000,'USDARS': 10000
          ,'USDCLP': 100,'USDINR': 100,'USDKRW': 100
         ,'USDCOP': 1,'USDIDR': 1
         ,'USDPHP': 1000,'USDTWD': 1000}]
pip_table=pd.DataFrame(pip)
pip_table=pip_table.T.reset_index()
pip_table.columns.values[0]='CurrencyPair'
pip_table.columns.values[1]='Pip multiplier'
'''
metatable=metatable.iloc[:,1:] #remove 1st coln

#Change type #suddenly there is no hour, minute and second now
metatable['Close Date'] = pd.to_datetime(metatable['Close Date'])
metatable['Forward Start Date'] = pd.to_datetime(metatable['Forward Start Date'])


#New Col - Ticker
metatable["Ticker"]=metatable["Curve"]+"_"+metatable["Maturity"] #Create Tikcer

#Filter out NDF only
metatable=metatable[metatable["Close Date"]>'2012-01-01'] #downsize sample

#Merge DFs.. add col
#metatable=pd.merge(metatable, pip_table, how='left', on=['CurrencyPair'])

'''
#Reorder and Eliminate some Col
metatable=metatable[['CurrencyPair', 
 'FORWARDDATE',
 'FIXINGDATE',
 'ValueTime',                    
 'Value',
 'Ticker',
 'Pip multiplier'
 ]]
 '''

#Filter out irs again by IMM fwd date (i.e. VD) this time
cond1=metatable['Forward Start Date Index']!=0
irs_metatable=metatable[(cond1)]

#Add Col: Days2Roll
irs_metatable['RemainDays']=(irs_metatable['Forward Start Date']-irs_metatable['Close Date']).dt.days
irs_metatable=irs_metatable.sort_values(by=['Ticker','Close Date','Forward Start Date']) #Reorder the fixing date

#Create IMM1 and IMM2 table which are for Spread Calculation
#IMM=irs_metatable.groupby(['Ticker','Close Date']).head(2) #Keep first two dates, only IMM1 and IMM2 of each value day

condimm1=irs_metatable['Forward Start Date Index']!=2
condimm2=irs_metatable['Forward Start Date Index']!=1
IMM1=irs_metatable[(condimm1)]   #Near Date....x
IMM2=irs_metatable[(condimm2)]   #Far Date....y
IMM2=IMM2[['Ticker','Close Date','Forward Start Date','Value']]  #Only useful Columns

#See IMM switch here
#IMM2.head(132).tail(6)

#finished here.

#Merge DFs
#Add Coln
#Reorder and Eliminate some Col
RollTable=pd.merge(IMM1, IMM2, how='inner', on=['Ticker', 'Close Date'])
RollTable['RollingPt']=RollTable['Value_y']-RollTable['Value_x'] #Far-Near
RollTable['RollingBps']=RollTable['RollingPt']*10000 #RollTable['Pip multiplier'] #Far-Near
RollTable=RollTable[['Ticker','Chain','Close Date','Forward Start Date_x','Forward Start Date_y','RemainDays','Maturity'
                     ,'Value_x','Value_y','RollingBps']] #Only useful Columns
RollTable.columns=['Ticker','CloseTime','ValueDate','FSDate1','FSDate2','LD_to_Roll','Maturity'
                     ,'Value1','Value2','RollingBps'] #Rename the colns

#Add Coln
RollTable['roll_pt_diff'] = RollTable.groupby(['Ticker','FSDate1'])['RollingBps'].apply(lambda x: x.diff())
RollTable['BmkD1']=RollTable['FSDate1']- np.timedelta64(benchmark,'D') #Get the benchmark date #obsolete but keep....

#Massage in order to get last day of IMM1 FX rate
DailyPnL=RollTable.groupby(['Ticker','FSDate1','ValueDate'],as_index=True).first() #First and Last is the same here.
#show only first (only one)row in each ticker and valuedate
lastValueday=DailyPnL.groupby(level=['Ticker','FSDate1']).tail(1) #All currencies' value day which is closest to the fixing
IRSRate=lastValueday['Value1'] #of each pair, every fix, found the closest outright e.g. 13Mar20' rate with fix 16Mar20

#Massage in order to get bmk date
#select the first row within benchmark date interval... so picked the first date on benchmark date
#or the day after (if on holiday)
bmk_VD=RollTable[RollTable['LD_to_Roll']<=benchmark].groupby(['Ticker','FSDate1'],as_index=True).first()
bmk_VD = bmk_VD.set_index('ValueDate', append=True)
#The data on benchmark day ONLY
bmk_IRSRate=bmk_VD['Value1'] #of each pair, on each IMM, found the rate at fix-14D

#Merge DFs
#Rename
#Set index
result = pd.merge(DailyPnL.reset_index(), IRSRate.reset_index(),on=['Ticker','FSDate1'],
                  how='left')        #Merge orig table with last date so to computer PnL in USD notional.

result.columns=['Ticker','FSDate1','ValueDate','CloseTime','FSDate2','LD_to_Roll',
                'Maturity','Value1','Value2','RollingBps','roll_pt_diff','BmkD1','Last_Valid_Date',
                'Last_Valid_IRSRate'] #Rename the colns
#result.set_index(['NDF_Pair','FixD1','ValueTime'],inplace=True)

#Merge DFs
#Rename
#Set index
bmk_result = pd.merge(result, bmk_IRSRate.reset_index(),on=['Ticker','FSDate1'],
                  how='left')        #Merge orig table with last date so to computer PnL in USD notional.
bmk_result.columns=['Ticker','FSDate1','ValueDate','CloseTime','FSDate2','LD_to_Roll',
                'Maturity','Value1','Value2','RollingBps','roll_pt_diff','BmkD1','Last_Valid_Date',
                'Last_Valid_IRSRate',
                   'LastBmk_Valid_Date','LastBmk_Valid_IRSRate'] #Rename the colns
bmk_result.set_index(['Ticker','FSDate1','ValueDate'],inplace=True)

#not yet start


#After combined, we have colns of last available dates and rates and colns of benchmark date and rates
#If benchamark date is not available yet, filled with last available dates and rates.
bmk_result.LastBmk_Valid_Date.fillna(bmk_result.Last_Valid_Date, inplace=True)
bmk_result.LastBmk_Valid_FXRate.fillna(bmk_result.Last_Valid_FXRate, inplace=True)

#add cols_
bmk_result['PnL_return']=bmk_result['roll_pt_diff']/bmk_result['IMM1_Fwd']               #PnL not yet PV!!!
#bmk_result['lastBmk_valid_PnL']=bmk_result['roll_pt_diff']/bmk_result['LastBmk_Valid_FXRate']               #PnL not yet PV!!!

#shift before combining with Portfolio
bmk_result['roll_pt_diff']=bmk_result['roll_pt_diff'].shift(-1)
bmk_result['PnL_return']=bmk_result['PnL_return'].shift(-1)
#bmk_result['lastBmk_valid_PnL']=bmk_result['lastBmk_valid_PnL'].shift(-1)

#Data Massage- Portfolio File
#Tables: a) metaFXPos
metaFXPos.set_index('BusinessDate',inplace=True)    #Date as index
metaFXPos.rename(columns=lambda x: x[1:],inplace=True) #Remove first character 'F' in Prod
metaFXPos=metaFXPos.unstack()
metaFXPos.index.names = ['NDF_Pair','ValueTime'] #Prepare to merge

##Merge DFs
finalresult = pd.merge(bmk_result.reset_index(), metaFXPos.reset_index(),on=['NDF_Pair','ValueTime'],
                  how='left')
#Rename
finalresult=finalresult.rename(columns = {0:'Pos'})

#PnL using the available day
finalresult['Pos_D_PnL']=finalresult['Pos']*finalresult['PnL_return']
#Cumsum in reverse using [::-1], -1 step each time for whole series
finalresult['Pos_PnL'] = finalresult.groupby(['NDF_Pair','FixD1'])['Pos_D_PnL'].apply(lambda x: x[::-1].cumsum()[::-1])

#Create a df to store all cum PnL in Benchmark date.
selected_rows=(finalresult["ValueTime"]==finalresult["LastBmk_Valid_Date"])
selected_columns=['NDF_Pair','FixD1','Pos_PnL']
benchmark_PnL=finalresult.loc[selected_rows,selected_columns]

#Merge DFs
#Rename
#Set index
final_bmk_result = pd.merge(finalresult, benchmark_PnL,on=['NDF_Pair','FixD1'],
                  how='left')        #Merge orig table with last date so to computer PnL in USD notional.
final_bmk_result.columns=['NDF_Pair','FixD1','ValueTime','IMM1_Fwd','IMM2_Fwd','Fwd_Fwd_pt',
                'Fwd_Fwd_pip','LD_to_Roll','roll_pt_diff','Bmk_D1','Last_Valid_Date','Last_Valid_FXRate',
                   'LastBmk_Valid_Date','LastBmk_Valid_FXRate','PnL_return','Pos','Pos_D_PnL','Pos_PnL',
                   'Bmk_PnL']       #Rename the colns
#final_bmk_result.set_index(['NDF_Pair','FixD1','ValueTime'],inplace=True)
final_bmk_result['Cum_Bmk_PnL']=final_bmk_result['Pos_PnL']-final_bmk_result['Bmk_PnL']
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

    # Graph2: to plot the PnL_by_IMM graphs using FacetGrid
    Graph2 = final_bmk_result[['NDF_Pair', 'FixD1', 'LD_to_Roll', 'Pos_PnL']].dropna()  # will segment later.

    # Graph2=Graph2[Graph2['NDF_Pair']=='USDTWD'] #testmode
    Graph2['FixD1'] = Graph2['FixD1'].map(lambda x: x.strftime('%b%y'))

    # Generates Graph just IMM by current
    Graph2storage = {}  # to store
    timer_start = time.time()

    for key, value in Graph2.groupby(
            ['NDF_Pair']):  # seperate dataframes by column value using group by, the col being groupby becomes key

        Graph2storage[key] = value  # store dataframes in dict

        # https://python-graph-gallery.com/242-area-chart-and-faceting/
        # https://seaborn.pydata.org/generated/seaborn.FacetGrid.html

        # Reset setting everytime
        mpl.rcParams.update(mpl.rcParamsDefault)
        sns.set()  # seaborn style
        g = sns.FacetGrid(Graph2storage[key], col='FixD1', hue='FixD1', col_wrap=8, sharex=False, sharey=False)
        # Create a grid : initialize it
        g = g.map(plt.plot, 'LD_to_Roll', 'Pos_PnL')  # Add the line over the area with the plot function
        g = g.map(plt.fill_between, 'LD_to_Roll', 'Pos_PnL', alpha=0.2)  # Fill the area with fill_between
        g = g.set_titles("{col_name}")  # Set Tile for each Grid
        g = g.set(xlim=(Graph2storage[key]['LD_to_Roll'].max(), 0))  # Desecending order

        # loop each grid to format
        for ax in g.axes[:]:
            # format the ticks
            # x-axis
            ax.xaxis.set_minor_locator(plt.MultipleLocator(5))  # show at every interval of 1
            ax.xaxis.set_major_locator(plt.MultipleLocator(10))  # show at every interval of 5

            # y-axis
            ax.yaxis.set_minor_locator(plt.MaxNLocator(5))
            ax.yaxis.set_major_locator(plt.MaxNLocator(10))
            # y-axis
            ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))

            # gridlines
            ax.grid(which='major', color='black', linestyle='--', linewidth=0.5, alpha=0.5)
            ax.grid(which='minor', color='black', linestyle='--', linewidth=0.5, alpha=0.1)

            # Tick tailor made
            ax.tick_params(direction='out', length=2, width=1, labelcolor='black', labelsize=9)

        # Add suptitle
        plt.subplots_adjust(top=0.92)  # not crash suptitle to subplot
        now = time.time()
        speed = int(now - timer_start)
        g = g.fig.suptitle(key + " The report has taken {0} seconds to generate this report".format(speed), fontsize=12)
        plt.savefig(my_path + '/output/PnL_by_IMM/' + key + '.png')
        plt.clf()  # to clean the memory
        plt.cla()
        plt.close()

# Graph3: to plot the Day2Roll_PnL graphs

Graph3_metadata = final_bmk_result[['NDF_Pair', 'LD_to_Roll', 'Pos_PnL']]
# Graph3_metadata=Graph3_metadata[(Graph3_metadata['NDF_Pair']=='USDPHP')] #testmode

# Groupby to form sum of every LD_to_roll in each FX Pair
Graph3 = Graph3_metadata.groupby(['NDF_Pair', 'LD_to_Roll'], as_index=True).sum().dropna()
Graph3 = Graph3.reset_index(level=1).reset_index(level=0)  # so no index..

graph3storage = {}  # to store

timer_start = time.time()

for key, value in Graph3.groupby(
        ['NDF_Pair']):  # seperate dataframes by column value using group by, the col being groupby becomes key

    graph3storage[key] = value  # store dataframes in dict, e.g dict_ccy_rollingpt['USDTWD']

    # lollipop
    # Data
    x = value['LD_to_Roll'].values
    y = value['Pos_PnL'].values

    my_color = np.where(y >= 0, 'blue', 'red')  # Color code of +ve and -ve groups

    sns.set()  # seaborn style
    fig, ax = plt.subplots(figsize=(20, 10), dpi=300)  # size of the plot

    # Reset setting everytime
    mpl.rcParams.update(mpl.rcParamsDefault)
    # Style use
    plt.style.use('seaborn-whitegrid')

    # Chart Type use
    plt.vlines(x=x, ymin=0, ymax=y, color=my_color, alpha=0.9, linewidth=1.9)  # Plot the verticle lines

    # gridlines
    ax.grid(which='major', color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.grid(which='minor', color='black', linestyle='--', linewidth=0.5, alpha=0.1)

    # Tick min and max
    ax.set_xlim(x.min(), x.max() + 1)
    plt.ylim(y.min() - abs(y.min()) / 20, y.max() + abs(y.max() / 20))

    # format the ticks
    # x-axis
    ax.xaxis.set_minor_locator(plt.MultipleLocator(1))  # show at every interval of 1
    ax.xaxis.set_major_locator(plt.MultipleLocator(5))  # show at every interval of 5

    # y-axis
    ax.yaxis.set_minor_locator(plt.MaxNLocator(20))
    ax.yaxis.set_major_locator(plt.MaxNLocator(10))

    # format the ticks
    # y-axis
    ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))

    # Add axis names
    plt.xlabel('Days-to-Fixing')
    plt.ylabel('Cumulative PnL')

    # Get Max and Min value and get in titles
    xmax = x[np.argmax(y)]
    ymax = y.max()
    xmin = x[np.argmin(y)]
    ymin = y.min()

    # Timer on process
    timer_end = time.time()
    speed = int(timer_end - timer_start)

    # Add Title and subtitle
    plt.title("Cumulative firm-level PnL of {0} rolling on a given day before NDF fixing".format(key)
              + "\nThe report took {0} seconds to be generated.".format(speed)
              + "\nBest PnL: {:1,.0f} rolling constantly on {:d}th day from fix".format(ymax, xmax)
              + "\nWorst PnL: {:1,.0f} rolling constantly on {:d}th day from fix".format(ymin, xmin)
              , loc='left', fontsize=12, fontweight=0, color='green', fontstyle='italic')  # Add title to the plot
    plt.suptitle(key, fontsize=18)

    # Save and close for next loop
    plt.savefig(my_path + '/output/Day2Roll_PnL/' + key + '.png')
    plt.clf()  # to clean the memory
    plt.cla()
    plt.close()

# Graph5: to plot the Bmk_Day2Roll_PnL graphs

Graph5_metadata = final_bmk_result[['NDF_Pair', 'LD_to_Roll', 'Cum_Bmk_PnL']]
# Graph5_metadata=Graph5_metadata[(Graph3_metadata['NDF_Pair']=='USDPHP')] #testmode

# Groupby to form sum of every LD_to_roll in each FX Pair
Graph5 = Graph5_metadata.groupby(['NDF_Pair', 'LD_to_Roll'], as_index=True).sum().dropna()
Graph5 = Graph5.reset_index(level=1).reset_index(level=0)  # so no index..

graph5storage = {}  # to store

timer_start = time.time()

for key, value in Graph5.groupby(
        ['NDF_Pair']):  # seperate dataframes by column value using group by, the col being groupby becomes key

    graph5storage[key] = value  # store dataframes in dict, e.g dict_ccy_rollingpt['USDTWD']

    # lollipop
    # Data
    x = value['LD_to_Roll'].values
    y = value['Cum_Bmk_PnL'].values

    my_color = np.where(y >= 0, 'blue', 'red')  # Color code of +ve and -ve groups

    sns.set()  # seaborn style
    fig, ax = plt.subplots(figsize=(20, 10), dpi=300)  # size of the plot

    # Reset setting everytime
    mpl.rcParams.update(mpl.rcParamsDefault)
    # Style use
    plt.style.use('seaborn-whitegrid')

    # Chart Type use
    plt.vlines(x=x, ymin=0, ymax=y, color=my_color, alpha=0.9, linewidth=1.9)  # Plot the verticle lines

    # gridlines
    ax.grid(which='major', color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.grid(which='minor', color='black', linestyle='--', linewidth=0.5, alpha=0.1)

    # Tick min and max
    ax.set_xlim(x.min(), x.max() + 1)
    plt.ylim(y.min() - abs(y.min()) / 20, y.max() + abs(y.max() / 20))

    # format the ticks
    # x-axis
    ax.xaxis.set_minor_locator(plt.MultipleLocator(1))  # show at every interval of 1
    ax.xaxis.set_major_locator(plt.MultipleLocator(5))  # show at every interval of 5

    # y-axis
    ax.yaxis.set_minor_locator(plt.MaxNLocator(20))
    ax.yaxis.set_major_locator(plt.MaxNLocator(10))

    # format the ticks
    # y-axis
    ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))

    # Add axis names
    plt.xlabel('Days-to-Fixing')
    plt.ylabel('Cumulative PnL vs Benchmark (14D from Fix)')

    # Get Max and Min value and get in titles
    xmax = x[np.argmax(y)]
    ymax = y.max()
    xmin = x[np.argmin(y)]
    ymin = y.min()

    # Timer on process
    timer_end = time.time()
    speed = int(timer_end - timer_start)

    # Add Title and subtitle
    plt.title(
        "Cumulative firm-level PnL (adjusted by Bmk(14y) PnL of {0} rolling on a given day before NDF fixing".format(
            key)
        + "\nThe report took {0} seconds to be generated.".format(speed)
        + "\nBest PnL: {:1,.0f} rolling constantly on {:d}th day from fix".format(ymax, xmax)
        + "\nWorst PnL: {:1,.0f} rolling constantly on {:d}th day from fix".format(ymin, xmin)
        , loc='left', fontsize=12, fontweight=0, color='green', fontstyle='italic')  # Add title to the plot
    plt.suptitle(key, fontsize=18)

    # Save and close for next loop
    plt.savefig(my_path + '/output/Bmk_Day2Roll_PnL/' + key + '.png')
    plt.clf()  # to clean the memory
    plt.cla()
    plt.close()