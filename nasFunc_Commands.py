#%% Libraries

# Data Manipulation
import pandas as pd

# Data Aquisition
import sqlalchemy
from sqlalchemy import text
import pymysql

# Data Vizualization
import seaborn as sns
import matplotlib.pyplot as plt

import nasFunctionsUtils as nf

#%% Server Connection

# =============================================================================
# db = MySQLdb.connect(host="127.0.0.1",    # your host, usually localhost
#                      user="root",         # your username
#                      db="nascar")
# =============================================================================

engine = sqlalchemy.create_engine('mysql+pymysql://root:@127.0.0.1/nascar')

with engine.begin() as con:
    query = text("""SELECT * FROM cup_laps""")
    df = pd.read_sql_query(query, con)
    
#%% Lap Data

#%% Load Data

lapFilePath = 'D:/Code/Sports/Nascar/Projects/nasDash_Streamlit/Data/2023_Nascar_Lap_Perc.csv'
cup23Avg = pd.read_csv(lapFilePath)

schFilePath = 'D:/Code/Sports/Nascar/Projects/nasDash_Streamlit/Data/2023_Nascar_Schedule.csv'
cup23Sch = pd.read_csv(schFilePath)

#%% Cup

#%%% Live Race

final4Drivers = ['Kyle Larson', 'William Byron', 'Christopher Bell', 'Ryan Blaney']

currentRace = nf.race_laps_race(2023, 5309)

currentRaceAvg = nf.lap_averages(currentRace, perc = 1)

currentDrivers = currentRaceAvg[(currentRaceAvg['Name'].isin(final4Drivers))]

currentDriversRun1 = currentRaceAvg[(currentRaceAvg['Name'].isin(final4Drivers)) & (currentRaceAvg['Run'] == 1)]

currentDriversRun2 = currentRaceAvg[(currentRaceAvg['Name'].isin(final4Drivers)) & (currentRaceAvg['Run'] == 2)]

currentDriversRun3 = currentRaceAvg[(currentRaceAvg['Name'].isin(final4Drivers)) & (currentRaceAvg['Run'] == 3)]

currentDriversRun4 = currentRaceAvg[(currentRaceAvg['Name'].isin(final4Drivers)) & (currentRaceAvg['Run'] == 4)]

currentDriversRun5 = currentRaceAvg[(currentRaceAvg['Name'].isin(final4Drivers)) & (currentRaceAvg['Run'] == 5)]

rankCols = ['LapTime', '5LapAverage', '10LapAverage', '25LapAverage', '50LapAverage']

for col in rankCols:
    currentRaceAvg[f'{col}_Rank'] = currentRaceAvg.groupby(['Lap'])[col].rank().fillna(0)
    currentDriversRun1[f'{col}_Rank'] = currentDriversRun1.groupby(['Lap'])[col].rank().fillna(0)
    currentDriversRun2[f'{col}_Rank'] = currentDriversRun2.groupby(['Lap'])[col].rank().fillna(0)
    currentDriversRun3[f'{col}_Rank'] = currentDriversRun3.groupby(['Lap'])[col].rank().fillna(0)
    currentDriversRun4[f'{col}_Rank'] = currentDriversRun4.groupby(['Lap'])[col].rank().fillna(0)
    currentDriversRun5[f'{col}_Rank'] = currentDriversRun5.groupby(['Lap'])[col].rank().fillna(0)

run1Avg = currentDriversRun1.groupby(['Name', 'Run']).agg(AvgPos = pd.NamedAgg('RunningPos', 'mean'),
                                                 FastestLap = pd.NamedAgg('LapTime', 'min'),
                                                 SlowLap = pd.NamedAgg("LapTime", 'max'),
                                                 FiveLap = pd.NamedAgg('5LapPercent', 'mean'),
                                                 TenLap = pd.NamedAgg('10LapPercent', 'mean'),
                                                 TwoFiveLap = pd.NamedAgg('25LapPercent', 'mean'),
                                                 FiveZeroLap = pd.NamedAgg('50LapPercent', 'mean'),
                                                 LongestRun = pd.NamedAgg('RunLap', 'max'),
                                                 AvgSpeedRank = pd.NamedAgg('LapTime_Rank', 'mean')
                                                 ).reset_index()

run2Avg = currentDriversRun2.groupby(['Name', 'Run']).agg(AvgPos = pd.NamedAgg('RunningPos', 'mean'),
                                                 FastestLap = pd.NamedAgg('LapTime', 'min'),
                                                 SlowLap = pd.NamedAgg("LapTime", 'max'),
                                                 FiveLap = pd.NamedAgg('5LapPercent', 'mean'),
                                                 TenLap = pd.NamedAgg('10LapPercent', 'mean'),
                                                 TwoFiveLap = pd.NamedAgg('25LapPercent', 'mean'),
                                                 FiveZeroLap = pd.NamedAgg('50LapPercent', 'mean'),
                                                 LongestRun = pd.NamedAgg('RunLap', 'max'),
                                                 AvgSpeedRank = pd.NamedAgg('LapTime_Rank', 'mean')
                                                 ).reset_index()

run3Avg = currentDriversRun3.groupby(['Name', 'Run']).agg(AvgPos = pd.NamedAgg('RunningPos', 'mean'),
                                                 FastestLap = pd.NamedAgg('LapTime', 'min'),
                                                 SlowLap = pd.NamedAgg("LapTime", 'max'),
                                                 FiveLap = pd.NamedAgg('5LapPercent', 'mean'),
                                                 TenLap = pd.NamedAgg('10LapPercent', 'mean'),
                                                 TwoFiveLap = pd.NamedAgg('25LapPercent', 'mean'),
                                                 FiveZeroLap = pd.NamedAgg('50LapPercent', 'mean'),
                                                 LongestRun = pd.NamedAgg('RunLap', 'max'),
                                                 AvgSpeedRank = pd.NamedAgg('LapTime_Rank', 'mean')
                                                 ).reset_index()

run4Avg = currentDriversRun3.groupby(['Name', 'Run']).agg(AvgPos = pd.NamedAgg('RunningPos', 'mean'),
                                                 FastestLap = pd.NamedAgg('LapTime', 'min'),
                                                 SlowLap = pd.NamedAgg("LapTime", 'max'),
                                                 FiveLap = pd.NamedAgg('5LapPercent', 'mean'),
                                                 TenLap = pd.NamedAgg('10LapPercent', 'mean'),
                                                 TwoFiveLap = pd.NamedAgg('25LapPercent', 'mean'),
                                                 FiveZeroLap = pd.NamedAgg('50LapPercent', 'mean'),
                                                 LongestRun = pd.NamedAgg('RunLap', 'max'),
                                                 AvgSpeedRank = pd.NamedAgg('LapTime_Rank', 'mean')
                                                 ).reset_index()

nf.plot_lap_comparison(currentRaceAvg, [2,3], final4Drivers, title = 'Phoenix Lap Rank Comparison')

nf.line_plot_lap_comp(currentRaceAvg, [4], final4Drivers, title = 'Phoenix Lap Rank Comparison')

def live_pit_data():
    pit_data = nf.pit_stop(2023, raceId = 5309)
    
    pattern = '|'.join([f'{driver}.*' for driver in final4Drivers])
    pit_data_filter = pit_data[(pit_data['driver_name'].str.contains(pattern, regex = True))]
    
    return(pit_data_filter)

h = live_pit_data()

#%%% Lap Data
#hhh = cup23[(cup23['Name'] == 'Ryan Blaney') & (cup23['RaceId'] == 5274)]
#hhh = cup23[(cup23['RaceId'] == 5274)]

cup23 = nf.race_laps_year(2023, 1)
cup23Avg = nf.lap_averages(cup23, perc = 1)

cup23Sch = nf.get_schedule(2023)

#%%%% Save to CSV

cup23Avg.to_csv('D:/Code/Sports/Nascar/Projects/nasDash_Streamlit/Data/2023_Nascar_Lap_Perc.csv')

cup23Sch.to_csv('D:/Code/Sports/Nascar/Projects/nasDash_Streamlit/Data/2023_Nascar_Schedule.csv')

#%%% Speed Ranks

# List of columns to rank
rankCols = ['LapTime', '5LapAverage', '10LapAverage', '25LapAverage', '50LapAverage']

rId = 5287
driver = 'Carson Hocevar'
runNum = 11

raceFilter = nf.race_filter(cup23Avg, rId)

for col in rankCols:
    raceFilter[f'{col}_Rank'] = raceFilter.groupby(['Lap'])[col].rank().fillna(0)
    
drivers = ['Tyler Reddick', 'Ty Gibbs', 'Chase Elliott', 'AJ Allmendinger', 'Kyle Larson']
runs = [1, 3]

nf.plot_lap_comparison(raceFilter, runs, drivers)

#%%%% Single Race

# Perform ranking within each Lap
for col in rankCols:
    raceFilter[f'{col}_Rank'] = raceFilter.groupby(['Lap'])[col].rank().fillna(0)
    
runAvg = raceFilter.groupby(['Name', 'Run']).agg(AvgPos = pd.NamedAgg('RunningPos', 'mean'),
                                                 FastestLap = pd.NamedAgg('LapTime', 'min'),
                                                 SlowLap = pd.NamedAgg("LapTime", 'max'),
                                                 FiveLap = pd.NamedAgg('5LapPercent', 'mean'),
                                                 TenLap = pd.NamedAgg('10LapPercent', 'mean'),
                                                 TwoFiveLap = pd.NamedAgg('25LapPercent', 'mean'),
                                                 FiveZeroLap = pd.NamedAgg('50LapPercent', 'mean'),
                                                 LongestRun = pd.NamedAgg('RunLap', 'max'),
                                                 AvgSpeedRank = pd.NamedAgg('LapTime_Rank', 'mean')
                                                 ).reset_index()

runAvg = runAvg[runAvg['LongestRun'] >= 5]

raceAvg = raceFilter.groupby(['Name']).agg(AvgPos = pd.NamedAgg('RunningPos', 'mean'),
                                                 FastestLap = pd.NamedAgg('LapTime', 'min'),
                                                 SlowLap = pd.NamedAgg("LapTime", 'max'),
                                                 FiveLap = pd.NamedAgg('5LapPercent', 'mean'),
                                                 TenLap = pd.NamedAgg('10LapPercent', 'mean'),
                                                 TwoFiveLap = pd.NamedAgg('25LapPercent', 'mean'),
                                                 FiveZeroLap = pd.NamedAgg('50LapPercent', 'mean'),
                                                 LongestRun = pd.NamedAgg('RunLap', 'max'),
                                                 AvgSpeedRank = pd.NamedAgg('LapTime_Rank', 'mean')
                                                 ).reset_index()

runAvg['5Rank'] = runAvg['FiveLap'].rank(ascending = False)
runAvg['10Rank'] = runAvg['TenLap'].rank(ascending = False)
runAvg['25Rank'] = runAvg['TwoFiveLap'].rank(ascending = False)
runAvg['50Rank'] = runAvg['FiveZeroLap'].rank(ascending = False)

hhhhh = runAvg[(runAvg['Run'] == runNum) & (runAvg['Name'] == driver)]

runLaps = raceFilter[(raceFilter['Run'] == runNum) & (raceFilter['Name'] == driver)]
raceLaps = raceFilter[(raceFilter['Name'] == driver)]

#%%%% Season 

badRaces = (5259, 5266, 5267, 5285, 5286)

cup23Avg = cup23Avg[~cup23Avg.isin(badRaces)]

# Perform ranking within each Lap
for col in rankCols:
    cup23Avg[f'{col}_Rank'] = cup23Avg.groupby(['RaceId', 'Lap'])[col].rank().fillna(0)

raceAvg = cup23Avg.groupby(['RaceId', 'Name']).agg(AvgPos = pd.NamedAgg('RunningPos', 'mean'),
                                                   FiveLap = pd.NamedAgg('5LapPercent', 'mean'),
                                                   TenLap = pd.NamedAgg('10LapPercent', 'mean'),
                                                   TwoFiveLap = pd.NamedAgg('25LapPercent', 'mean'),
                                                   FiveZeroLap = pd.NamedAgg('50LapPercent', 'mean'),
                                                   LongestRun = pd.NamedAgg('RunLap', 'max'),
                                                   AvgSpeedRank = pd.NamedAgg('LapTime_Rank', 'mean')
                                                   ).reset_index()

# Get all the races and select the columns of interest
tracks = cup23Sch[['RaceId', 'Track_Name', 'Race_Name']].rename(columns={'Race_Id': 'RaceId'})

# Add tracks and race name to the df to identify what race it was apart of
raceAvg = raceAvg.join(tracks.set_index('RaceId'), on = 'RaceId', how='inner')

blaneyRaceAvg = raceAvg[raceAvg['Name'] == 'Ryan Blaney']

#%%% xFinish

# Load loop data the data
loopFilePath = 'D:\\Code\\Sports\\Nascar\\Projects\\xFinish\\Data\\nascarLoopCup.csv'
loopData = pd.read_csv(loopFilePath)

# Get top 5 most recent races for each driver at each track and track type
loopDataLast5_track = loopData.groupby(['Full_Name', 'Track_Name']).apply(lambda x: x.nlargest(5, 'career_race_number')).reset_index(drop=True)
loopDataLast5_track_type = loopData.groupby(['Full_Name', 'track_type']).apply(lambda x: x.nlargest(5, 'career_race_number')).reset_index(drop=True)

#%%%% Avg Driver Rating

# Calculate average ratings
avg_rating_top5_track = loopDataLast5_track.groupby(['Full_Name', 'Track_Name'])['rating'].mean().reset_index()
avg_rating_top5_track_type = loopDataLast5_track_type.groupby(['Full_Name', 'track_type'])['rating'].mean().reset_index()

# Pivot for single-row per driver
pivot_avg_track = avg_rating_top5_track.pivot(index='Full_Name', columns='Track_Name', values='rating').reset_index()
pivot_avg_track_type = avg_rating_top5_track_type.pivot(index='Full_Name', columns='track_type', values='rating').reset_index()

# Merge pivoted DataFrames
final_df = pd.merge(pivot_avg_track, pivot_avg_track_type, on='Full_Name', how='outer')

# Get top 5 most recent races for each driver overall
loopData_top5_overall = loopData.groupby('Full_Name').apply(lambda x: x.nlargest(5, 'career_race_number')).reset_index(drop=True)

# Calculate average rating for these races
avgRatingLast5_overall = loopData_top5_overall.groupby('Full_Name')['rating'].mean().reset_index()
avgRatingLast5_overall.rename(columns={'rating': 'avgRatingLast5_overall'}, inplace=True)

# Merge this new column into the existing final DataFrame
RatingDataMain = pd.merge(final_df, avgRatingLast5_overall, on='Full_Name', how='outer')

#%%%% Avg Finish

# Calculate average finish
avgFinishLast5_track = loopDataLast5_track.groupby(['Full_Name', 'Track_Name'])['ps'].mean().reset_index()
avgFinishLast5_track_type = loopDataLast5_track_type.groupby(['Full_Name', 'track_type'])['ps'].mean().reset_index()

# Pivot for single-row per driver
pivotAvg_track = avgFinishLast5_track.pivot(index='Full_Name', columns='Track_Name', values='ps').reset_index()
pivotAvg_track_type = avgFinishLast5_track_type.pivot(index='Full_Name', columns='track_type', values='ps').reset_index()

# Merge pivoted DataFrames
avgFinishCombined = pd.merge(pivotAvg_track, pivotAvg_track_type, on='Full_Name', how='outer')

# Get top 5 most recent races for each driver overall
avgFinishLast5_overall = loopData.groupby('Full_Name').apply(lambda x: x.nlargest(5, 'career_race_number')).reset_index(drop=True)

# Calculate average rating for these races
avgFinishLast5_overall = avgFinishLast5_overall.groupby('Full_Name')['ps'].mean().reset_index()
avgFinishLast5_overall.rename(columns={'ps': 'avgFinishLast5_overall'}, inplace=True)

# Merge this new column into the existing final DataFrame
FinishDataMain = pd.merge(avgFinishCombined, avgFinishLast5_overall, on='Full_Name', how='outer')

#%%%% Avg Running Position

# Calculate average finish
avgRunningLast5_track = loopDataLast5_track.groupby(['Full_Name', 'Track_Name'])['avg_ps'].mean().reset_index()
avgRunningLast5_track_type = loopDataLast5_track_type.groupby(['Full_Name', 'track_type'])['avg_ps'].mean().reset_index()

# Pivot for single-row per driver
pivotRunning_track = avgRunningLast5_track.pivot(index='Full_Name', columns='Track_Name', values='avg_ps').reset_index()
pivotRunning_track_type = avgRunningLast5_track_type.pivot(index='Full_Name', columns='track_type', values='avg_ps').reset_index()

# Merge pivoted DataFrames
avgRunningCombined = pd.merge(pivotRunning_track, pivotRunning_track_type, on='Full_Name', how='outer')

# Get top 5 most recent races for each driver overall
avgRunningLast5_overall = loopData.groupby('Full_Name').apply(lambda x: x.nlargest(5, 'career_race_number')).reset_index(drop=True)

# Calculate average rating for these races
avgRunningLast5_overall = avgRunningLast5_overall.groupby('Full_Name')['avg_ps'].mean().reset_index()
avgRunningLast5_overall.rename(columns={'avg_ps': 'avgRunningLast5_overall'}, inplace=True)

# Merge this new column into the existing final DataFrame
RunningDataMain = pd.merge(avgRunningCombined, avgRunningLast5_overall, on='Full_Name', how='outer')

#%%%% Intermediate

track = 'Las Vegas Motor Speedway'
trackType = 'Intermediate'
lastRatingAvg = 'avgRatingLast5_overall'
lastFinishAvg = 'avgFinishLast5_overall'
lastRunningAvg = 'avgRunningLast5_overall'

RatingData = RatingDataMain[['Full_Name', track, trackType, lastRatingAvg]]

FinishData = FinishDataMain[['Full_Name', track, trackType, lastFinishAvg]]

RunningData = RunningDataMain[['Full_Name', track, trackType, lastRunningAvg]]


InterData = RatingData.merge(FinishData, on = 'Full_Name', suffixes = ('', ' - Finish')).merge(RunningData,on='Full_Name', suffixes = (' - Rating', ' - Running'))

InterData.to_csv('D:/Code/Sports/Nascar/Projects/nasDash_Streamlit/Data/InterModelData.csv')
#%%% Pit Data

cup23pit = nf.pit_stop(2023, 1)

cup23pitAvg = nf.avg_4_tire(cup23pit)

# cup23pit.to_csv('D:/Code/Sports/Nascar/Projects/nasDash_Streamlit/Data/2023_Nascar_Pit.csv')

#%%% To SQL

cup23Avg.to_sql(con = engine, name = 'cup_laps', if_exists = 'replace')

#%% Xfinity

#%%% Lap Data

xFin23 = nf.race_laps_year(2023, 2)

xFin23Avg = nf.lap_averages(xFin23)

#%%% To SQL

xFin23Avg.to_sql(con = engine, name = 'xfin_laps', if_exists = 'replace')

#%% Trucks

#%%% Lap Data

trucks23 = nf.race_laps_year(2023, 3)

trucks23Avg = nf.lap_averages(trucks23)

#%%% To SQL

trucks23Avg.to_sql(con = engine, name = 'trucks_laps', if_exists = 'replace')
