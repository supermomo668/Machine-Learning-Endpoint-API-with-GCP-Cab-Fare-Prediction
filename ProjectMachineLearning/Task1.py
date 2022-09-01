#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Imports
import pandas as pd
import math
import numpy as np
from sklearn.model_selection import cross_val_score
from xgboost.sklearn import XGBRegressor
import xgboost as xgb

import osmnx as ox
from shapely.geometry import Point
gdf = ox.geocode_to_gdf('New York, NY, USA')
geom = gdf.loc[0, 'geometry']

# get the bounding box of the city
geom.bounds
# In[2]:


def haversine_distance(origin, destination):
    """
    # Formula to calculate the spherical distance between 2 coordinates, with each specified as a (lat, lng) tuple

    :param origin: (lat, lng)
    :type origin: tuple
    :param destination: (lat, lng)
    :type destination: tuple
    :return: haversine distance
    :rtype: float
    """
    lat1, lon1 = origin
    lat2, lon2 = destination
    radius = 6371  # km

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) * math.sin(dlat / 2) + math.cos(math.radians(lat1)) * math.cos(
        math.radians(lat2)) * math.sin(dlon / 2) * math.sin(dlon / 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = radius * c

    return d


# In[ ]:

def datetime_tomore(df):
    time_features = df.loc[:, ['pickup_datetime']]
    # TODO: extract time-related features from the `pickup_datetime` column.
    #       (replace "None" with your implementation)
    time_features['year'] = time_features.pickup_datetime.apply(lambda x:x.year)
    time_features['month'] = time_features.pickup_datetime.apply(lambda x:x.month)
    time_features['hour'] = time_features.pickup_datetime.apply(lambda x:x.hour)
    time_features['weekday'] = time_features.pickup_datetime.apply(lambda x:x.weekday)
    # quantize
    #time_features['hour_bin'] = pd.qcut(time_features['hour'],4).cat.codes
    return pd.concat([df, time_features], axis=1)

def filter_NYC(df):    
    # determine if a point is within the city boundary
    coord_list = list(df[['pickup_longitude','pickup_latitude']].to_records(index=False))
    return df[[geom.intersects(Point(coords)) for coords in coord_list]]

def process_distance(df):
    pick_up_loc = ["pickup_longitude","pickup_latitude"]
    drop_off_loc = ["dropoff_longitude","dropoff_latitude"]
    return df.apply(lambda x: haversine_distance((x[pick_up_loc]), (x[drop_off_loc])), axis=1)

def MSG_distance(df, mode="dropoff"):
    MSG_coor = (40.750298, -73.993324) # lat, lng
    MSG_lat, MSG_long = MSG_coor
    drop_off_loc = ["dropoff_latitude", "dropoff_longitude"]
    pick_up_loc = ["pickup_latitude","pickup_longitude"]
    if mode=="dropoff":
        return df.apply(lambda x: haversine_distance(MSG_coor, (x[drop_off_loc])), axis=1)
    else:
        return df.apply(lambda x: haversine_distance(MSG_coor, (x[pick_up_loc])), axis=1)

fare_high = df[['fare_amount']].quantile(0.999)[0]
fare_low = df[['fare_amount']].quantile(0.001)[0]
def drop_quantile(df):
    return df.loc[df.fare_amount > fare_low].loc[df.fare_amount < fare_high]
    
num_cols = ["pickup_longitude","pickup_latitude","dropoff_longitude","dropoff_latitude","passenger_count"]
pred_col = ['fare_amount']

def process_train_data(raw_df):
    """
    TODO: Implement this method.
    
    You may drop rows if needed.

    :param raw_df: the DataFrame of the raw training data
    :return:  a DataFrame with the predictors created
    """
    #fil na
    raw_df[num_cols] = raw_df[num_cols].fillna(value=raw_df[num_cols].mean())
    # drop location
    raw_df = filter_NYC(raw_df)
    #raw_df[num_cols] = raw_df[num_cols].fillna(value=raw_df[num_cols].mean())
    # filter quantile
    raw_df = drop_quantile(raw_df)
    # datetime
    raw_df = datetime_tomore(raw_df)
    # distance
    raw_df["distance"] = process_distance(raw_df)
    #raw_df = raw_df.drop(num_cols[:-1], axis=1)
    return raw_df


def process_test_data(raw_df):
    """
    TODO: Implement this method.
    
    You should NOT drop any rows.

    :param raw_df: the DataFrame of the raw test data
    :return: a DataFrame with the predictors created
    """
    # fill mean
    raw_df[num_cols] = raw_df[num_cols].fillna(value=raw_df[num_cols].mean())
    # 
    # datetime
    raw_df = datetime_tomore(raw_df)
    # distance
    raw_df["distance"] = process_distance(raw_df)
    #raw_df = raw_df.drop(num_cols[:-1], axis=1)
    return raw_df


# In[ ]:


# Load data
raw_train = pd.read_csv('data/cc_nyc_fare_train_small.csv', parse_dates=['pickup_datetime'])
print('Shape of the raw data: {}'.format(raw_train.shape))


# In[ ]:


# Transform features using the function you have defined
df_train = process_train_data(raw_train)

# Remove fields that we do not want to train with
X = df_train.drop(['key', 'fare_amount', 'pickup_datetime'], axis=1, errors='ignore')

# Extract the value you want to predict
Y = df_train['fare_amount']
print('Shape of the feature matrix: {}'.format(X.shape))


# In[ ]:


# Build final model with the entire training set
final_model = XGBRegressor(objective ='reg:squarederror')
final_model.fit(X, Y)

# Read and transform test set
raw_test = pd.read_csv('data/cc_nyc_fare_test.csv', parse_dates=['pickup_datetime'])
df_test = process_test_data(raw_test)
X_test = df_test.drop(['key', 'pickup_datetime'], axis=1, errors='ignore')

# Make predictions for test set and output a csv file
# DO NOT change the column names
df_test['predicted_fare_amount'] = final_model.predict(X_test)
df_test[['key', 'predicted_fare_amount']].to_csv('predictions.csv', index=False)

