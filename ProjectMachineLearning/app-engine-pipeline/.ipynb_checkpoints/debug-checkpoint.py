#!/usr/bin/env python
# coding: utf-8

# In[6]:
if __name__=="__main__":

    import math, pandas as pd
    from clients.ai_platform import AIPlatformClient
    
    #export GOOGLE_MAPS_API_KEY=AIzaSyDWWT0taJ-74L8uXCem8K84ImEGQ6wf8P4
    
    ai_platform_client = AIPlatformClient('ml-fare-prediction-347604', 'MLFarePredictionModel', 'MLFarePredictionModel_1')

    def haversine_distance(origin, destination):
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

    def datetime_tomore(df):
        time_features = df.loc[:, ['pickup_datetime']]
        # TODO: extract time-related features from the `pickup_datetime` column.
        #       (replace "None" with your implementation)
        time_features['year'] = time_features.pickup_datetime.apply(lambda x:x.year)
        time_features['month'] = time_features.pickup_datetime.apply(lambda x:x.month)
        time_features['hour'] = time_features.pickup_datetime.apply(lambda x:x.hour)
        time_features['weekday'] = time_features.pickup_datetime.apply(lambda x:x.weekday if type(x.weekday)==int else x.weekday())
        # quantize
        #time_features['hour_bin'] = pd.qcut(time_features['hour'],4).cat.codes
        return pd.concat([df, time_features], axis=1)

    def process_distance(df):
        pick_up_loc = ["pickup_longitude","pickup_latitude"]
        drop_off_loc = ["dropoff_longitude","dropoff_latitude"]
        return df.apply(lambda x: haversine_distance((x[pick_up_loc]), (x[drop_off_loc])), axis=1)

    num_cols = ["pickup_longitude","pickup_latitude","dropoff_longitude","dropoff_latitude","passenger_count"]
    pred_col = ['fare_amount']

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

    print('PREDICT')
    #json_str = request.data.decode('utf-8')
    json_str = '{"passenger_count": [1], "pickup_datetime": ["2022-04-20 13:52:01"], "pickup_longitude": [-73.99116719999999], "pickup_latitude": [40.72792270000001], "dropoff_longitude": [-73.9834643], "dropoff_latitude": [40.7735614]}'
    print(json_str)
    raw_data_df = pd.read_json(json_str, convert_dates=["pickup_datetime"])
    print("\n\n\n[DEBUG]",raw_data_df.columns,"\n\n\n")
    predictors_df = process_test_data(raw_data_df)
    predictors_df = predictors_df.drop(['pickup_datetime'], axis=1, errors='ignore')
    print("\n\n\n[DEBUG]",predictors_df.columns,"\n\n\n")
    print("\n\n\n[DEBUG]",predictors_df.values.tolist(),"\n\n\n")
    
    predictions = ai_platform_client.predict(predictors_df.values.tolist())
    print("\n\n\n[DEBUG]", predictions, "\n\n\n")

