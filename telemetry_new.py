#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd 
import numpy as np 



telemetry = pd.read_csv('march2019.csv', error_bad_lines=False)

telemetry = telemetry.rename(columns=telemetry.iloc[2]) # use row 2 for header name
telemetry = telemetry.drop(telemetry.index[0:3])  # drop first 3 rows
telemetry = telemetry.reset_index(drop = True)    # reset index 

'''
telemetry_chunk = pd.read_csv('MAR2019_1801_DetailTransactions.csv',chunksize=200000,error_bad_lines=False)
telemetry = pd.read_csv('MAR2019_1801_DetailTransactions.csv',nrows=1000,error_bad_lines=False)

chunk_list = []  # append each chunk df here 
# Each chunk is in df format
for chunk in telemetry_chunk:  
    # perform data filtering 
    chunk_filter = chunk.iloc[3:]
    
    # Once the data filtering is done, append the chunk to list
    chunk_list.append(chunk_filter)
    
# concat the list into dataframe 
df_concat = pd.concat(chunk_list)
'''

#telemetry = telemetry.iloc[2:]


# Delete multiple columns from the dataframe
telemetry = telemetry.drop(["TollingPointID", "TollingZoneID", "DirectionCode", "SequenceNumber",
                           "Tick","DVISCameraNum1","DVISCameraNum2","ImageCount","UTCDateTime",
                           "TollingPointID1"], axis=1)

    
# use pd.concat to join the new columns with your original dataframe
telemetry = pd.concat([telemetry,pd.get_dummies(telemetry['TrxnType'], prefix='TrxnType')],axis=1)

# now drop the original 'country' column (you don't need it anymore)
telemetry.drop(['TrxnType'],axis=1, inplace=True)


# Rename multiple columns in one go with a larger dictionary
telemetry.rename(
    columns={
        "TrxnDateTime": "datetime",
        "AVCClassID": "Class0",
        "TagNumber": "TagRead",
        "TrxnType_Transactions": "Transactions",
        "TrxnType_Violations": "Violations"
    },
    inplace=True
)

# format data in fields 

telemetry['datetime'] = pd.to_datetime(telemetry['datetime'])
telemetry['Class0'] = telemetry['Class0'].astype('int')
telemetry['TagRead'] = telemetry['TagRead'].apply(lambda x: int(x, 16)) # convert from HEX to int
cols = ['VehicleLength', 'VehicleHeight','VehicleWidth', 'VehicleSpeed']
telemetry[cols] = telemetry[cols].astype(str).astype(int)  # convert object to int


telemetry['Class0'] = (telemetry['Class0'] == 0).astype(int) # if it's class0 then = 1
telemetry['TagRead'] = (telemetry['TagRead'] != 0).astype(int)  # if it's TagRead = 1

#  ------------------------------------------------------------------------------------------------


# Calculate mean values for telemetry features
temp = []
fields = ['VehicleSpeed', 'VehicleHeight', 'VehicleLength', 'VehicleWidth']
for col in fields:
    temp.append(pd.pivot_table(telemetry,index='datetime',columns='LaneID', values=col).resample('3H', closed='left', label='right').mean().unstack())
telemetry_mean_3h = pd.concat(temp, axis=1)
telemetry_mean_3h.columns = [i + 'mean_3h' for i in fields]
telemetry_mean_3h.reset_index(inplace=True)

# repeat for standard deviation
temp = []
for col in fields:
    temp.append(pd.pivot_table(telemetry,index='datetime',columns='LaneID',values=col).resample('3H', closed='left', label='right').std().unstack())
telemetry_sd_3h = pd.concat(temp, axis=1)
telemetry_sd_3h.columns = [i + 'sd_3h' for i in fields]
telemetry_sd_3h.reset_index(inplace=True)


# For capturing a longer term effect, 24 hour lag features are also calculated as below  
temp = []
fields = ['VehicleSpeed', 'VehicleHeight', 'VehicleLength', 'VehicleWidth']
for col in fields:
    temp.append(pd.pivot_table(telemetry,index='datetime',columns='LaneID',values=col).rolling(24).mean().resample('3H',closed='left', label='right').first().unstack())
    
telemetry_mean_24h = pd.concat(temp, axis=1)
telemetry_mean_24h.columns = [i + 'mean_24h' for i in fields]
telemetry_mean_24h.reset_index(inplace=True)
telemetry_mean_24h = telemetry_mean_24h.loc[-telemetry_mean_24h['VehicleSpeedmean_24h'].isnull()]

# repeat for standard deviation
temp = []
fields = ['VehicleSpeed', 'VehicleHeight', 'VehicleLength', 'VehicleWidth']
for col in fields:
    temp.append(pd.pivot_table(telemetry,index='datetime',columns='LaneID', values=col).rolling(24).std().resample('3H', closed='left', label='right').first().unstack())
    
telemetry_sd_24h = pd.concat(temp, axis=1)
telemetry_sd_24h.columns = [i + 'sd_24h' for i in fields]
telemetry_sd_24h = telemetry_sd_24h.loc[-telemetry_sd_24h['VehicleSpeedsd_24h'].isnull()]
telemetry_sd_24h.reset_index(inplace=True)

# Notice that a 24h rolling average is not available at the earliest timepoints


# For capturing a longer term effect, 3 hour lag features are also calculated as below  for TXN, VIO, TagRead and Class 0   
temp = []
fields = ['Transactions', 'Violations', 'TagRead', 'Class0'] 

for col in fields:
    temp.append(pd.pivot_table(telemetry,index='datetime',columns='LaneID', values=col).resample('3H', closed='left', label='right').sum().unstack())

telemetry_sum_1h = pd.concat(temp, axis=1)
telemetry_sum_1h.columns = [i + 'count_3h' for i in fields]
telemetry_sum_1h.reset_index(inplace=True)

# For capturing a longer term effect, 24 hour lag features are also calculated as below  for TXN, VIO, TagRead and Class 0
temp = []
fields = ['Transactions', 'Violations', 'TagRead', 'Class0'] 
for col in fields:
    temp.append(pd.pivot_table(telemetry,index='datetime',columns='LaneID',values=col).rolling(24).sum().resample('3H',closed='left', label='right').first().unstack())
    
telemetry_sum_24h = pd.concat(temp, axis=1)
telemetry_sum_24h.columns = [i + 'count_24h' for i in fields]
telemetry_sum_24h.reset_index(inplace=True)
#telemetry_sum_24h = telemetry_sum_24h.loc[-telemetry_sum_24h['Transactionscount_24h'].isnull()]
telemetry_sum_24h = telemetry_sum_24h.isnull(-1)
   
# merge columns of feature sets created earlier       
telemetry_feat = pd.concat([telemetry_mean_3h,
                        telemetry_sd_3h.iloc[:, 2:6],
                        telemetry_mean_24h.iloc[:, 2:6],
                        telemetry_sd_24h.iloc[:, 2:6],
                        telemetry_sum_1h.iloc[:, 2:6],
                        telemetry_sum_24h.iloc[:,2:6]], axis=1).dropna()  

