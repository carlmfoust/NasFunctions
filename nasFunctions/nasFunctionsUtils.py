#%% Libraries

# Data Manipulation
import pandas as pd
import numpy as np
from scipy.stats import percentileofscore
#from datetime import date

import seaborn as sns
import matplotlib.pyplot as plt

# Data Aquisition
import requests
#import MySQLdb

import traceback

# Graphics
import plotly.io as pio
pio.renderers.default='browser'
pd.options.mode.chained_assignment = None  # default='warn'

#%% Utility Functions

def get_raceIds(year, series = 1):
    """
    
    Gets list of race IDS from NASCAR

    Parameters
    ----------
    year : Int
        Year that you want Race Ids from.
    series : Int
        The series that you want the Race Ids for. 1 = Cup, 2 = Xfinity, 3 = Trucks

    Returns
    -------
    raceIDS : Series
        A series of integers of each races Id from that years schedule

    """
    schedule = requests.get(f'https://cf.nascar.com/cacher/{year}/{series}/schedule-combined-feed.json')
    
    dictr = schedule.json()
    recs = dictr['response']
    raceIDSList = pd.json_normalize(recs)
    raceIDSList['Exhibition'] = np.where(raceIDSList['Alt_Track_Name'] == '', 'No'
                                         , np.where(raceIDSList['Alt_Track_Name'].isin({'Daytona 500', 'DAYTONA 500', 'DAYTONA Road Course'}), 'No', 'Yes'))
    raceIDSList = raceIDSList[raceIDSList['Actual_Laps'] > 0]
    raceIDS = raceIDSList[['Race_Id', 'Exhibition']]
    raceIDS.rename(columns={'Race_Id':'RaceId'}, inplace=True)
    raceIDS = raceIDS[::-1]
    raceIDS.loc[-1] = [5287, 'No']
    raceIDS = raceIDS.sort_values(by=['RaceId']).reset_index()
    
    return raceIDS[['RaceId', 'Exhibition']]

def get_schedule(year, series = 1):
    # Series: 1 - Cup, 2 - Xfinity, 3 - Trucks
    
    schedule = requests.get(f'https://cf.nascar.com/cacher/{year}/{series}/schedule-combined-feed.json').json()
    schedule_responce = schedule['response']
    schedule_norm = pd.json_normalize(schedule_responce)
    schedule_sub = schedule_norm[['Race_Name', 'Race_Id', 'Track_Name', 'Race_Date_Plain', 'Current_Winner_Name', 'Playoff_Round', 'Actual_Laps', 'Track_Id']]
    schedule_sub.rename(columns={'Race_Id':'RaceId'}, inplace=True)
    schedule_sub['RaceID_Text'] = schedule_sub['Track_Name'] + ' - ' + schedule_sub['Race_Name']
    schedule_sub['Exhibition'] = np.where(schedule_sub['Race_Name'].str.contains('Duel|Clash|All-Star'), 'Yes', 'No')
    if year == 2023:
        schedule_sub.loc[-1] = ['Wurth 400', 5282, 'Dover Motor Speedway', '2023-5-1', 'Martin Truex Jr.', np.nan, 400, 103, 'Dover Motor Speedway - Wurth 400', 'No']
        schedule_sub.loc[-2] = ['Coke 600', 5287, 'Charlotte Motor Speedway', '2023-5-29', 'Ryan Blaney', np.nan, 400, 162, 'Charlotte Motor Speedway - Coke 600', 'No']
    schedule_sub = schedule_sub.sort_values(by=['RaceId']).reset_index()
    schedule_sub = schedule_sub[['Race_Name', 'RaceId', 'Track_Name', 'Race_Date_Plain', 'Current_Winner_Name', 'Playoff_Round', 'Actual_Laps', 'Track_Id', 'RaceID_Text', 'Exhibition']]
    
    return schedule_sub

def get_schedule_bare(year, series = 1):
    # Series: 1 - Cup, 2 - Xfinity, 3 - Trucks
    
    schedule = requests.get(f'https://cf.nascar.com/cacher/{year}/{series}/schedule-combined-feed.json').json()
    schedule_responce = schedule['response']
    schedule_norm = pd.json_normalize(schedule_responce)
    schedule_norm['Exhibition'] = np.where(schedule_norm['Alt_Track_Name'] == '', 'No', np.where(schedule_norm['Alt_Track_Name'].isin({'Daytona 500', 'DAYTONA 500', 'DAYTONA Road Course'}), 'No', 'Yes'))
    schedule_sub = schedule_norm[['Race_Name', 'Race_Id', 'Track_Name', 'Actual_Laps', 'Track_Id', 'Exhibition']]
    schedule_sub['RaceID_Text'] = schedule_sub['Track_Name'] + ' - ' + schedule_sub['Race_Name']
    
    return schedule_sub

def get_tracks():
    # https://frcs.pro/nascar/tracks/
    # High tire wear?
    dirt = (216, 208, 215)
    shortFlat = (206, 217, 47, 22, 177, 26)
    shortBanked = (14, 103)
    intermediateFlat = (52, 138, 84, 45, 51)
    intermediate = (162, 39, 4, 40, 41, 61, 42, 43)
    speedway = (38, 123, 133, 198)
    superSpeedway = (111, 105, 82)
    roadCoarse = (209, 210, 218, 214, 212, 211, 72, 204, 34, 99, 157)
    
    tracks = requests.get('https://cf.nascar.com/cacher/tracks/tracks-feed.json').json()
    tracks_norm = pd.json_normalize(tracks)
    tracks_norms = tracks_norm[['track_id', 'track_name', 'track_type', 'length']].copy()
    
    conditions = [(tracks_norms['track_id'].isin(dirt)),
                  (tracks_norms['track_id'].isin(shortFlat)),
                  (tracks_norms['track_id'].isin(shortBanked)),
                  (tracks_norms['track_id'].isin(intermediateFlat)),
                  (tracks_norms['track_id'].isin(intermediate)),
                  (tracks_norms['track_id'].isin(speedway)),
                  (tracks_norms['track_id'].isin(superSpeedway)),
                  (tracks_norms['track_id'].isin(roadCoarse))
                  ]
    
    choices = ['Dirt', 'Short Flat', 'Short Banked', 'Intermediate Flat', 'Intermediate', 'Speedway', 'Super Speedway', 'Road Course']
    
    tracks_norms['track_type'] = np.select(conditions, choices, default='NON')
    
    tracks_norms = tracks_norms.rename(columns = {'track_id':'Track_Id', 'track_name':'Track_Name'})
    
    return tracks_norms
    
def race_filter(dataFrame, raceId):
    return dataFrame[dataFrame['RaceId'] == raceId]    

#%% Data Collection

def race_laps_year(year, series = 1):
    
    raceIds = get_raceIds(year, series)
    raceIds = raceIds[raceIds['Exhibition'] == 'No']
    currentRaceNum = 0
    
    raceLaps = []
    
    for r in raceIds['RaceId']:
        
        currentRaceNum += 1
        print(f'Finding Laps for Race: {r}. {currentRaceNum} / {len(raceIds)}')
        
        try:
          
            lapsJson = requests.get(f'https://cf.nascar.com/cacher/{year}/{series}/{r}/lap-times.json')
            nasdata = lapsJson.json()
        
            flags = pd.DataFrame(nasdata['flags'])
        
            laps = pd.DataFrame(
                [
                {
                  'Name': item['FullName'],
                  'Num': item['Number'],
                  'Driver_Id': item['NASCARDriverID'],
                  'Manufacturer': item['Manufacturer'],
                  'Laps': item['Laps']
                }
                for item in nasdata['laps']
                ]
            )
            laps['Race_ID'] = r
        
            for n in range(len(laps)):
              t=pd.DataFrame(laps['Laps'][n])
              t['Flag'] = flags['FlagState']
              t['Name'] = laps['Name'][n]
              t['Name'] = t['Name'].str.replace(r"\(.*\)", "", regex=True)
              t['Name'] = t['Name'].str.replace(r"#", "", regex=True)
              t['Name'] = t['Name'].str.replace(r"*", "", regex=True)
              t['Name'] = t['Name'].str.replace(r"^\s+|\s+$", "", regex=True)
              t['Name'].str.replace('','')
              t['Num'] = laps['Num'][n]
              t['Driver_Id'] = laps['Driver_Id'][n]
              t['Manufacturer'] = laps['Manufacturer'][n]
              t['RaceId'] = laps['Race_ID'][n]
              raceLaps.append(t)
        
            raceLapsCat = pd.concat(raceLaps)
            
            # Additional print statements for debugging
            print(f"Race {r} processed successfully.")
            
        except:
            print(f'Race {r} is not avaliable.')
        pass
    
    return raceLapsCat

def race_laps_race(year, raceID, series = 1):
    
    raceLaps = []
    
    try:
      lapsJson = requests.get(f'https://cf.nascar.com/cacher/{year}/{series}/{raceID}/lap-times.json')
      nasdata = lapsJson.json()
  
      flags = pd.DataFrame(nasdata['flags'])
  
      laps = pd.DataFrame(
          [
          {
            'Name': item['FullName'],
            'Num': item['Number'],
            'Driver_Id': item['NASCARDriverID'],
            'Manufacturer': item['Manufacturer'],
            'Laps': item['Laps']
          }
          for item in nasdata['laps']
          ]
      )
      laps['Race_Id'] = raceID
  
      for n in range(len(laps)):
        t=pd.DataFrame(laps['Laps'][n])
        t['Flag'] = flags['FlagState']
        t['Name'] = laps['Name'][n]
        t['Name'] = t['Name'].str.replace(r"\(.*\)", "", regex=True)
        t['Name'] = t['Name'].str.replace(r"#", "", regex=True)
        t['Name'] = t['Name'].str.replace(r"*", "", regex=True)
        t['Name'] = t['Name'].str.replace(r"^\s+|\s+$", "", regex=True)
        t['Name'].str.replace('','')
        t['Num'] = laps['Num'][n]
        t['Driver_Id'] = laps['Driver_Id'][n]
        t['Manufacturer'] = laps['Manufacturer'][n]
        t['RaceId'] = laps['Race_Id'][n]
        raceLaps.append(t)
  
      return pd.concat(raceLaps)
    except:
      print(f'Race {raceID} is not avaliable. 1')
      pass

#%% N Lap Averages

def lap_averages(dataFrame, perc = 0, year = 2023):
    
    raceIds = dataFrame['RaceId'].unique()
    percentMain = pd.DataFrame()
    currentRaceNum = 0
        
    cup23Sch = get_schedule(year)
    tracks = get_tracks()[['Track_Id', 'track_type']]
    cup23Sch = cup23Sch.join(tracks.set_index('Track_Id'), on = 'Track_Id', how='inner')
    thres = {'Short Flat': 3,
             'Short Banked': 3,
             'Dirt': 3,
             'Intermediate Flat': 3,
             'Intermediate': 3,
             'Speedway': 1,
             'Super Speedway': 1,
             'Road Course': 0.5
             }
    
    for l in raceIds:
        
        currentRaceNum += 1
        print(f'Finding Averages for Race: {l}. {currentRaceNum} / {len(raceIds)}')
        
        try:
            raceLaps = dataFrame[dataFrame["RaceId"] == l]
            raceLaps['Num'] = raceLaps.Num.astype(int)
            
            track_type = cup23Sch[cup23Sch['RaceId'] == l]
            h = track_type['track_type']
            h = track_type['track_type'].astype(str).map(thres)
            
            kTop = (raceLaps.LapTime.median() - raceLaps.LapTime.min())
            kBottom = (raceLaps.LapTime.max() - raceLaps.LapTime.min())
            a, b = 1, h.item()
            k = a * (kTop / kBottom) + b
            Q1 = np.nanpercentile(raceLaps['LapTime'], 25)
            Q3 = np.nanpercentile(raceLaps['LapTime'], 75)
            IQR = Q3 - Q1
            outlier_threshold_iqr = Q3 + (k * IQR)
            #print(Q3, outlier_threshold_iqr, k)
                        
            # Initialize 'Run' and 'RunLap' columns if they don't exist
            if 'Run' not in raceLaps.columns:
                raceLaps['Run'] = np.nan
            if 'RunLap' not in raceLaps.columns:
                raceLaps['RunLap'] = np.nan
                
            dictOfDrivers = dict(iter(raceLaps.groupby('Name')))
            
            # Check if dictOfDrivers is empty
            if not dictOfDrivers:
                print("dictOfDrivers is empty.")
                continue
            
            for key, value in dictOfDrivers.items():
                
                run = 1
                runLap = 0
                
                for i, r in value.iterrows():
                    
                    if r['Lap'] == 0:
                        #print(f'{i}:INITAL')
                        pass
                    elif (r['Lap'] == 1) | (r['Flag'] == 4):
                        #print(f'{i}:MID RUN')
                        value.loc[i, "Run"] = run
                        value.loc[i, "RunLap"] = runLap
                        runLap += 1
                    elif outlier_threshold_iqr > value.loc[i, 'LapTime']:
                        #print(f'{r["Lap"]}, {fastestLap}, {value.loc[i - 1, "LapTime"]}')
                        value.loc[i, "Run"] = run
                        value.loc[i, "RunLap"] = runLap
                        runLap += 1
                    else:
                        if np.isnan(value.loc[i - 1, "Run"]):
                            #print(f'{i}:BAD LAP')
                            value.loc[i, 'Run'] = np.nan
                            value.loc[i, 'RunLap'] = np.nan
                        else:
                            #print(f'{i}: RESET RUN')
                            run += 1
                            runLap = 0
                            value.loc[i, 'Run'] = np.nan
                            value.loc[i, 'RunLap'] = np.nan
                    
                # Update rolling_avg for the next iteration
                filtered_value = value[value['RunLap'] != 0]
                if not filtered_value.empty and filtered_value['Run'].count() > 0:
                    value["5LapAverage"] = filtered_value[['Run','LapTime']].groupby('Run', as_index= False).rolling(5).mean()["LapTime"]
                    value["10LapAverage"] = filtered_value[['Run','LapTime']].groupby('Run', as_index= False).rolling(10).mean()["LapTime"]
                    value["25LapAverage"] = filtered_value[['Run','LapTime']].groupby('Run', as_index= False).rolling(25).mean()["LapTime"]
                    value["50LapAverage"] = filtered_value[['Run','LapTime']].groupby('Run', as_index= False).rolling(50).mean()["LapTime"]
                else:
                    value["5LapAverage"] = np.nan
                    value["10LapAverage"] = np.nan
                    value["25LapAverage"] = np.nan
                    value["50LapAverage"] = np.nan
                
            singleDriver = pd.concat(dictOfDrivers.values())
            
            if perc == 1:
                singleDriver["5LapPercent"] = [100 - percentileofscore(singleDriver["5LapAverage"], e, kind = 'strict', nan_policy = 'omit') for e in singleDriver["5LapAverage"]]
                singleDriver["10LapPercent"] = [100 - percentileofscore(singleDriver["10LapAverage"], e, kind = 'strict', nan_policy = 'omit') for e in singleDriver["10LapAverage"]]
                singleDriver["25LapPercent"] = [100 - percentileofscore(singleDriver["25LapAverage"], e, kind = 'strict', nan_policy = 'omit') for e in singleDriver["25LapAverage"]]
                singleDriver["50LapPercent"] = [100 - percentileofscore(singleDriver["50LapAverage"], e, kind = 'strict', nan_policy = 'omit') for e in singleDriver["50LapAverage"]]
                
                percentMain = pd.concat([percentMain, singleDriver])
            else:
                percentMain = pd.concat([percentMain, singleDriver])
                
            # Additional print statements for debugging
            print(f"Race {l} processed successfully.")
    
        except Exception as error:
            print(f'Error: Race {l}')
            print("Exception Type:", type(error))
            print("Exception Message:", error)
            print("Traceback:")
            traceback.print_exc()
            pass

    return percentMain

def lap_avg_x(dataFrame):
    
    raceIds = dataFrame['RaceId'].unique()
    
    #percentMain = pd.DataFrame()
    
    #currentRaceNum = 0
    
    for l in raceIds:
        try:
            raceLaps = dataFrame[dataFrame["RaceId"] == l]
            raceLaps['Num'] = raceLaps.Num.astype(int)
            pitData = pit_stop(2023, raceId=l)
            if len(pitData) > 0:
                pitData = pitData[['vehicle_number', 'lap_count', 'pit_stop_type']].rename(columns={'vehicle_number': 'Num', 'lap_count': 'Lap'})
                raceLaps = raceLaps.merge(pitData, how='left', on=['Num', 'Lap'])
            else:
                raceLaps['pit_stop_type'] = np.nan
            
            raceLaps['Run'] = np.where((raceLaps['Lap'] == 1) | (raceLaps['Flag'] == 4), 1, np.nan)
            raceLaps['RunLap'] = np.where((raceLaps['Lap'] == 1) | (raceLaps['Flag'] == 4), 1, np.nan)
            
            greenFlagStop = raceLaps["Flag"].shift(1).eq(1) & raceLaps["Flag"].eq(1)
            greenFlagStop = greenFlagStop | greenFlagStop.shift(-1).fillna(False)
            
            raceLaps['Run'] = np.where(greenFlagStop, np.nan, raceLaps['Run'])
            raceLaps['RunLap'] = np.where(greenFlagStop, np.nan, raceLaps['RunLap'])
            raceLaps['Run'] = np.where((raceLaps["Flag"].eq(1) & raceLaps['pit_stop_type'].isna() & raceLaps["Flag"].shift(1).eq(1)), raceLaps['Run'], np.nan)
            raceLaps['RunLap'] = np.where((raceLaps["Flag"].eq(1) & raceLaps['pit_stop_type'].isna() & raceLaps["Flag"].shift(1).eq(1)), raceLaps['RunLap'], np.nan)
            raceLaps['Run'] = np.where((raceLaps["Flag"].eq(1) & raceLaps['pit_stop_type'].notna() & raceLaps["Flag"].shift(1).eq(1)), np.nan, raceLaps['Run'].fillna(method='ffill')+1)
            raceLaps['RunLap'] = np.where((raceLaps["Flag"].eq(1) & raceLaps['pit_stop_type'].notna() & raceLaps["Flag"].shift(1).eq(1)), np.nan, 1)
            raceLaps['Run'] = raceLaps['Run'].fillna(method='ffill')
            raceLaps['RunLap'] = raceLaps['RunLap'].fillna(method='ffill')
            
            singleDriver = raceLaps.copy()
            singleDriver["5LapAverage"] = 1
            
        except:
            pass
            
    return singleDriver


#%% Top Percent Lap Avg

def top_per(dataFrame, column, percent = 0.05):
    # Finds the top n% of laps ran for a variable
    # top_per(cup23Avg, '10LapPercent', 0.05) finds the top 5% of laps in the '10LapPercent' column in the cup23Avg dataframe
    #
    # dataFrame: the df containing lap data
    # column: the column in the dataFrame we want to find the top laps for
    # percent: the number of laps we want to select, 0.05 is the top 5% of laps ran
    
    return dataFrame.groupby('DriverID').apply(lambda x: x.nlargest(int(len(x) * percent), column)).reset_index(drop = True).groupby(['DriverID', 'Name'])[column].mean().reset_index()
    
#%% Driver Images

def driver_images(series = 'nascar-cup-series'):
    drivers_requests = requests.get('https://cf.nascar.com/cacher/drivers.json')
    drivers_json = drivers_requests.json()
    drivers_responce = drivers_json['response']
    drivers_norm = pd.json_normalize(drivers_responce)
    drivers_Cup = drivers_norm[drivers_norm['Driver_Series'] == 'nascar-cup-series']
    drivers_Firesuit = drivers_Cup.loc[drivers_Cup['Firesuit_Image_Small'].str.len() > 0]
    drivers_Firesuit_sub = drivers_Firesuit[['Full_Name', 'Firesuit_Image_Small']]
    
    return drivers_Firesuit_sub

#%% Driver Numbers

def driver_numbers(series = 'nascar-cup-series'):
    drivers_requests = requests.get('https://cf.nascar.com/cacher/drivers.json')
    drivers_json = drivers_requests.json()
    drivers_responce = drivers_json['response']
    drivers_norm = pd.json_normalize(drivers_responce)
    drivers_Cup = drivers_norm[drivers_norm['Driver_Series'] == 'nascar-cup-series']
    drivers_Firesuit = drivers_Cup.loc[drivers_Cup['Firesuit_Image_Small'].str.len() > 0]
    drivers_Firesuit_sub = drivers_Firesuit[['Full_Name', 'Firesuit_Image_Small']]
    
    return drivers_Firesuit_sub

#%% Driver Ids

def driver_ids(series = 'nascar-cup-series'):
    '''
    nascar-cup-series, nascar-xfinity-series, nascar-craftsman-truck-series

    Parameters
    ----------
    series : TYPE, optional
        DESCRIPTION. The default is 'nascar-cup-series'.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    drivers_requests = requests.get('https://cf.nascar.com/cacher/drivers.json')
    drivers_json = drivers_requests.json()
    drivers_responce = drivers_json['response']
    drivers_norm = pd.json_normalize(drivers_responce)
    #if series == 1:
    #    drivers = drivers_norm[drivers_norm['Driver_Series'] == 'nascar-cup-series']
    #elif series == 2:
    #    drivers = drivers_norm[drivers_norm['Driver_Series'] == 'nascar-cup-series']
    #elif series == 3:
    #    drivers = drivers_norm[drivers_norm['Driver_Series'] == 'nascar-cup-series']
    #else:
    #    return print('Invalid Series: Use either 1, 2, or 3.')
    
    return drivers_norm[['Nascar_Driver_ID', 'Full_Name']].rename(columns = {'Nascar_Driver_ID':'Driver_Id'}).drop_duplicates()

#%% Pit Stop

def pit_stop(year, series = 1, raceId = 0):
    
    if raceId == 0:
        raceIDS = get_raceIds(year, series)
        raceIDS = raceIDS['RaceId']
        
        races = []
        
        for r in raceIDS:
          try:
            json = pd.read_json(f'https://cf.nascar.com/cacher/live/series_1/{r}/live-pit-data.json')
            json['RaceId'] = r
            races.append(json)
          except:
            print(f'Pit Stop: Race {r} is not avaliable.')
            print(r)
            pass
        try:
            races = pd.concat(races)
        
            races['travel_duration'] = races['in_travel_duration'] + races['out_travel_duration']
            
            pitColumns = ['vehicle_number', 'driver_name', 'vehicle_manufacturer', 'leader_lap', 'lap_count', 'pit_in_flag_status', 'total_duration', 'pit_stop_duration', 'travel_duration', 'pit_stop_type', 'pit_in_rank', 'pit_out_rank', 'positions_gained_lost', 'RaceID']
            races = races[pitColumns]
            races['leader_lap'] += 1
            races['lap_count'] += 1
        except:
            pass
        
        return races
    else:
        races = []

        try:
          json = pd.read_json(f'https://cf.nascar.com/cacher/live/series_1/{raceId}/live-pit-data.json')
          json['RaceId'] = raceId
          races.append(json)
        except:
          print(f'Pit Stop: Race {raceId} is not avaliable.')
          pass
        try:
            races = pd.concat(races)
        
            races['travel_duration'] = races['in_travel_duration'] + races['out_travel_duration']
            
            pitColumns = ['vehicle_number', 'driver_name', 'vehicle_manufacturer', 'leader_lap', 'lap_count', 'pit_in_flag_status', 'total_duration', 'pit_stop_duration', 'travel_duration', 'pit_stop_type', 'pit_in_rank', 'pit_out_rank', 'positions_gained_lost', 'RaceID']
            races = races[pitColumns]
            races['leader_lap'] += 1
            races['lap_count'] += 1
        except:
            pass
        
        return races

#%% Average 4 Tire Pit Stops

def avg_4_tire(dataFrame, race = 0):
    racesPitCatFiltered = dataFrame.loc[dataFrame['pit_stop_type'] == 'FOUR_WHEEL_CHANGE']

    # Race Filter
    # Checking to see if the user wants data for a specific race.
    # If they dont specify a raceID, calculate averages for entire dataFrame.
    if race == 0:
        pass
    else:
        racesPitCatFiltered = racesPitCatFiltered.loc[racesPitCatFiltered['RaceID'] == race]
        if len(racesPitCatFiltered) == 0:
            return print(f'Error: {race} not found in the data set.')

    #Q1 = racesPitCatFiltered['pit_stop_duration'].quantile(0.25)
    #Q3 = racesPitCatFiltered['pit_stop_duration'].quantile(0.75)
    #IQR = Q3 - Q1
    
    #upperBound = Q3 + 1.5 * IQR

    #racesPitCatFiltered = racesPitCatFiltered.query('(@Q1 - 1.5 * @IQR) <= pit_stop_duration <= (@Q3 + 1.5 * @IQR)')
    racesPitCatFiltered = racesPitCatFiltered[racesPitCatFiltered['pit_stop_duration'].between(8.5, 15)]
    
    pitStopTimes = racesPitCatFiltered.groupby(['vehicle_number']).agg(Avg4Tire = pd.NamedAgg('pit_stop_duration', 'mean'),
                                                                       Count = pd.NamedAgg('pit_stop_duration', 'count'),
                                                                       FastestStop = pd.NamedAgg('pit_stop_duration', 'min')
                                                                       )
    
    pitStopTimes['Rank'] = pitStopTimes['Avg4Tire'].rank()

    return pitStopTimes.sort_values(['Avg4Tire'], ascending=True).reset_index()

#%% Loop Data

def loop_data(year, series = 1):
    
    raceIDS = get_raceIds(year, series)
    
    races = []

    for r in raceIDS['RaceId']:
      try:
        loopRequest = requests.get(f'https://cf.nascar.com/loopstats/prod/{year}/{series}/{r}.json')
        loopJSON = loopRequest.json()
        loopDict =  loopJSON[0]
        
        loopDrivers = loopDict['drivers']
        loopConcat = pd.json_normalize(loopDrivers)
        
        loopConcat['RaceId'] = loopDict['RaceId']
        
        races.append(loopConcat)
      except:
        print(f'Race {r} is not avaliable. 3')
        pass
    
    return pd.concat(races)

def track_avg(startYear = 2019, endYear = 2023, series = 1):
    
    loopData = pd.DataFrame()
    
    for i in range(startYear, endYear + 1):
        
        race = loop_data(i, series)
        race['Year'] = i
        race = race.rename(columns = {'RaceId':'Race_Id'})
        
        ids = get_schedule_bare(i, series)
        ids.loc[ids['Track_Name'] == 'Dover International Speedway', 'Track_Name'] = 'Dover Motor Speedway'
        tracks = get_tracks()
        
        ids = pd.merge(ids, tracks, how='left', on=['Track_Id', 'Track_Name'])
        race = pd.merge(race, ids, how='left', on=['Race_Id'])
        
        loopData = pd.concat([loopData, race])
        
        print(f'Completed the {i} Season')
    
    return pd.concat([loopData])

#%% Aggregate Functions

def track_averages(track = None, tType = None, series = 1):
    
    dataFrame = track_avg(series)
    
    ids = driver_ids()
    
    hFiltered = dataFrame[(dataFrame['Track_Name'].str.contains(track)) & (dataFrame['Exhibition'] == 'No')]
    
    if hFiltered['Track_Id'].isin([111]).any():
        hFiltered.loc[hFiltered['Year'] < 2022, 'track_type'] = 'Intermediate'
        
    if tType != None:
        if hFiltered['track_type'].str.contains(tType).any():
            hFiltered = hFiltered[hFiltered['track_type'] == tType]
        else:
            return print('Invalid Track Type')
    
    pitStopTimes = hFiltered.groupby(['driver_id']).agg(Rating = pd.NamedAgg('rating', 'mean'),
                                                        StartAvg = pd.NamedAgg('start_ps', 'mean'),
                                                        MidAvg = pd.NamedAgg('mid_ps', 'mean'),
                                                        FinishAvg = pd.NamedAgg('ps', 'mean'),
                                                        ClosingAvg = pd.NamedAgg('closing_ps', 'mean'),
                                                        AvgPos = pd.NamedAgg('avg_ps', 'mean'),
                                                        Count = pd.NamedAgg('start_ps', 'count')
                                                        ).reset_index()
    
    pitStopTimes = pitStopTimes.rename(columns = {'driver_id':'Driver_Id'})
    
    pitStopTimes = pd.merge(pitStopTimes, ids, how='left', on=['Driver_Id'])
    
    return pitStopTimes

def track_type_averages(trackType, startYear = 2019, series = 1):
    """
    
    Get driver average statistics from NASCAR Loop Data. Data begins in 2019.

    Parameters
    ----------
    trackType : Int
        Type of track to find summary for.
        'Dirt', 'Short Flat', 'Short Banked', 'Intermediate Flat', 'Intermediate', 'Speedway', 'Super Speedway', 'Road Course'
    startYear: Int
        Year that you want data collection to start.

    Returns
    -------
    DataFrame
        Average values for the track type specified by the user.

    """
       
    trackAvg = track_avg()
    
    ids = driver_ids(series)
    
    trackAvgFiltered = trackAvg[(trackAvg['track_type'].str.contains(trackType)) & (trackAvg['Year'] >= startYear) & (trackAvg['Exhibition'] == 'No')]
    
    typeAverages = trackAvgFiltered.groupby(['driver_id']).agg(Rating = pd.NamedAgg('rating', 'mean'),
                                                        StartAvg = pd.NamedAgg('start_ps', 'mean'),
                                                        MidAvg = pd.NamedAgg('mid_ps', 'mean'),
                                                        FinishAvg = pd.NamedAgg('ps', 'mean'),
                                                        ClosingAvg = pd.NamedAgg('closing_ps', 'mean'),
                                                        AvgPos = pd.NamedAgg('avg_ps', 'mean'),
                                                        Count = pd.NamedAgg('start_ps', 'count')
                                                        ).reset_index()
    
    typeAverages = typeAverages.rename(columns = {'driver_id':'Driver_Id'})
    typeAverages = pd.merge(typeAverages, ids, how = 'left', on = ['Driver_Id'])
    
    return typeAverages

def race_summary(raceID, year = 2023, series = 1):
    
    data = race_laps_race(year, raceID, series)
    
    ids = driver_ids(series)
    
    raceSummary = data.groupby(['Driver_Id']).agg(Rating = pd.NamedAgg('rating', 'mean'),
                                                        StartAvg = pd.NamedAgg('start_ps', 'mean'),
                                                        MidAvg = pd.NamedAgg('mid_ps', 'mean'),
                                                        FinishAvg = pd.NamedAgg('ps', 'mean'),
                                                        ClosingAvg = pd.NamedAgg('closing_ps', 'mean'),
                                                        AvgPos = pd.NamedAgg('avg_ps', 'mean'),
                                                        Count = pd.NamedAgg('start_ps', 'count')
                                                        ).reset_index()
    
    #raceSummary = data.rename(columns = {'driver_id':'Driver_Id'})
    raceSummary = pd.merge(data, ids, how = 'left', on = ['Driver_Id'])
                   
    return raceSummary
    
#%% N Lap Averages SELECT RACE

# =============================================================================
# selctedRaceID = 5177
# 
# selected_race = percMaster[percMaster["RaceID"] == selctedRaceID]
# 
# selected_race['Name'] = selected_race['Name'].str.replace(r"\(.*\)","")
# selected_race['Name'] = selected_race['Name'].str.replace(r"#","")
# 
# selected_race_drivers = selected_race[selected_race['Name'] == 'Tyler Reddick']
# =============================================================================

#%% Visualizations

#%%% Run Lap Comparison
def plot_lap_comparison(data, runList, selected_drivers, title = ''):
    """
    
    Generates a plot that displays lap speed rankings for multiple drivers over 2 diffrent runs.

    Parameters
    ----------
    data : data.frame
        Lap data for race of interest.
    runList: array
        An array of runs that are to be displayed.
        Ex: [1,10]
    selected_drivers: array
        An array of driver that are to be displayed.
        Ex: ['Ryan Blaney', 'Austin Cindric', 'Joey Logano']

    Returns
    -------
    Seaborn Plot
        A plot that compares lap time rankings for multiple drivers.

    """
    
    filtered_data = data[(data['Run'].isin(runList)) & (data['Name'].isin(selected_drivers))]
    filtered_data = filtered_data.dropna(subset=['LapTime_Rank', 'RunningPos'])
    
    overall_mean_RunningPos = filtered_data.groupby('Name')['LapTime_Rank'].mean().reset_index().sort_values(by='LapTime_Rank')

    # Sort 'selected_drivers' based on their overall average RunningPos
    selected_drivers = overall_mean_RunningPos['Name'].tolist()
    
    # Calculate the starting and ending lap number for each driver and each run
    lap_start_end = filtered_data.groupby(['Name', 'Run'])['Lap'].agg(['min', 'max']).reset_index()
    lap_start_end.columns = ['Name', 'Run', 'Start_Lap', 'End_Lap']
    
    # Calculate mean and median values for annotations and median dots
    median_LapTime_Rank = filtered_data.groupby(['Name', 'Run'])['LapTime_Rank'].median().reset_index()
    mean_RunningPos = filtered_data.groupby(['Name', 'Run'])['RunningPos'].mean().reset_index()
    
    plt.style.use('fivethirtyeight')
    fig, ax = plt.subplots(figsize=(12, 8), tight_layout=True)
    
    custom_palette = sns.color_palette("Set1", 2)
    sns.stripplot(data=filtered_data, x='Name', y='LapTime_Rank', hue='Run', ax=ax, jitter=True, dodge=True, marker='o', alpha=0.7, s=10, palette=custom_palette, order=selected_drivers)
    sns.scatterplot(data=median_LapTime_Rank, x='Name', y='LapTime_Rank', hue='Run', style='Run', markers=['D', 's'], s=100, palette=custom_palette, edgecolor='black')
    
    ylim_min = filtered_data['LapTime_Rank'].min() - 1
    ylim_max = filtered_data['LapTime_Rank'].max() + 1
    ax.set_ylim(ylim_min, ylim_max)
    ax.set_xlim(-0.5, len(selected_drivers) - 0.5)
    
    # Add horizontal lines for each driver's mean RunningPos for each run
    for index, row in mean_RunningPos.iterrows():
        driver = row['Name']
        run = row['Run']
        mean_pos = row['RunningPos']
        
        # Center the horizontal line near the corresponding run's data points
        dodge_width = 0.75  # width to dodge for each run; adjust as needed
        x_center = selected_drivers.index(driver)
        x_min = x_center - dodge_width / 2 + dodge_width * runList.index(run) / len(runList)
        x_max = x_min + dodge_width / len(runList)
        
        ax.hlines(mean_pos, xmin=x_min, xmax=x_max, colors=custom_palette[runList.index(run)], linestyles='dashed')
    
    # Adds a subtitle that shows the laps that the runs took place on
    subtitles = []
    for run in runList:
        start_lap = lap_start_end[lap_start_end['Run'] == run]['Start_Lap'].min()
        end_lap = lap_start_end[lap_start_end['Run'] == run]['End_Lap'].min()
        if start_lap > 0 and end_lap > 0:
            subtitles.append(f"Run {run}: Laps {start_lap}-{end_lap}")
    
    subtitle_text = " | ".join(subtitles)
    
    ax.text(x=0.5, y=1.1, s=title, fontsize=24, weight='bold', ha='center', va='bottom', transform=ax.transAxes)
    ax.text(x=0.5, y=1.05, s=subtitle_text, fontsize=12, alpha=0.75, ha='center', va='bottom', transform=ax.transAxes)
    
    ax.set_xlabel('Driver', labelpad=15)
    ax.set_ylabel('Lap Time Rank', labelpad=15)
    
    #plt.legend(title='Run', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()
    
def line_plot_lap_comp(data, runList, selected_drivers, title = ''):
    
    filtered_data = data[(data['Run'].isin(runList)) & (data['Name'].isin(selected_drivers))]
    filtered_data = filtered_data.dropna(subset=['LapTime_Rank', 'RunningPos'])
    
    plt.style.use('fivethirtyeight')
    fig, ax = plt.subplots(figsize=(12, 8), tight_layout=True)
    
    custom_palette = sns.color_palette("Set1", len(selected_drivers))
    
    sns.lineplot(data=filtered_data.reset_index(drop=True), x="Lap", y="5LapAverage", hue='Name', ax=ax, alpha=0.7, palette=custom_palette)
    
    start_lap = filtered_data['Lap'].min()
    end_lap = filtered_data['Lap'].max()
    
    subtitle = f'Laps: {start_lap}-{end_lap}'
    
    ax.text(x=0.5, y=1.1, s='Phoenix 5-Lap Average Comparison', fontsize=24, weight='bold', ha='center', va='bottom', transform=ax.transAxes)
    ax.text(x=0.5, y=1.05, s=subtitle, fontsize=12, alpha=0.75, ha='center', va='bottom', transform=ax.transAxes)
    
    ax.set_xlabel('Lap', labelpad=15)
    ax.set_ylabel('5-Lap Average Lap Time', labelpad=15)
    
    plt.show()