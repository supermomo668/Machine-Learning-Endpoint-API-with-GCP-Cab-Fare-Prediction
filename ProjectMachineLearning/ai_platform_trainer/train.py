from google.cloud import storage
import pandas as pd
import math
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from hypertune import HyperTune
import argparse
import os

import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
install("osmnx")
import osmnx as ox

from shapely.geometry import Point
gdf = ox.geocode_to_gdf('New York, NY, USA')
geom = gdf.loc[0, 'geometry']

# get the bounding box of the city
geom.bounds
# ==========================
# ==== Define Variables ====
# ==========================
# When dealing with a large dataset, it is practical to randomly sample
# a smaller proportion of the data to reduce the time and money cost per iteration.
#
# When you are testing, start with 0.2. You need to change it to 1.0 when you make submissions.
# TODO: Set SAMPLE_PROB to 1.0 when you make submissions
SAMPLE_PROB = 1   # Sample 20% of the whole dataset
random.seed(15619)  # Set the random seed to get deterministic sampling results

# TODO: Update the value using the ID of the GS bucket
# For example, if the GS path of the bucket is gs://my-bucket the OUTPUT_BUCKET_ID will be "my-bucket"
OUTPUT_BUCKET_ID = 'ml-fare-prediction-347604'

# DO NOT CHANGE IT
DATA_BUCKET_ID = 'cc-gcp-image'
# DO NOT CHANGE IT
TRAIN_FILE = 'datasets/cc_nyc_fare_train_small.csv'


# =========================
# ==== Utility Methods ====
# =========================
def haversine_distance(origin, destination):
    """
    Calculate the spherical distance from coordinates

    :param origin: tuple (lat, lng)
    :param destination: tuple (lat, lng)
    :return: Distance in km
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


# =====================================
# ==== Define data transformations ====
# =====================================
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
    fare_high = raw_df[['fare_amount']].quantile(0.999)[0]
    fare_low = raw_df[['fare_amount']].quantile(0.001)[0]
    def drop_quantile(df):
        return df.loc[df.fare_amount > fare_low].loc[df.fare_amount < fare_high]
        
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


if __name__ == '__main__':
    # ===========================================
    # ==== Download data from Google Storage ====
    # ===========================================
    print('Downloading data from google storage')
    print('Sampling {} of the full dataset'.format(SAMPLE_PROB))
    input_bucket = storage.Client().bucket(DATA_BUCKET_ID)
    output_bucket = storage.Client().bucket(OUTPUT_BUCKET_ID)
    input_bucket.blob(TRAIN_FILE).download_to_filename('train.csv')

    raw_train = pd.read_csv('train.csv', parse_dates=["pickup_datetime"],
                            skiprows=lambda i: i > 0 and random.random() > SAMPLE_PROB)

    print('Read data: {}'.format(raw_train.shape))

    # =============================
    # ==== Data Transformation ====
    # =============================
    df_train = process_train_data(raw_train)

    # Prepare feature matrix X and labels Y
    X = df_train.drop(['key', 'fare_amount', 'pickup_datetime'], axis=1)
    Y = df_train['fare_amount']
    X_train, X_eval, y_train, y_eval = train_test_split(X, Y, test_size=0.33)
    print('Shape of feature matrix: {}'.format(X_train.shape))

    # ======================================================================
    # ==== Improve model performance with hyperparameter tuning ============
    # ======================================================================
    # You are provided with the code that creates an argparse.ArgumentParser
    # to parse the command line arguments and pass these parameters to Google AI Platform
    # to be tuned by HyperTune.
    # TODO: Your task is to add at least 3 more arguments.
    # You need to update both the code below and config.yaml.

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--job-dir',  # AI Platform passes this in by default
        required=True
    )

    # the 5 lines of code below parse the --max_depth option from the command line
    # and will convert the value into "args.max_depth"
    # "args.max_depth" will be passed to XGBoost training through the `params` variables
    # i.e., xgb.train(params, ...)
    #
    # the 5 lines match the following YAML entry in `config.yaml`:
    # - parameterName: max_depth
    #   type: INTEGER
    #   minValue: 4
    #   maxValue: 10
    # "- parameterName: max_depth" matches "--max_depth"
    # "type: INTEGER" matches "type=int""
    # "minValue: 4" and "maxValue: 10" match "default=6"
    parser.add_argument(
        '--max_depth',
        default=6,
        type=int
    )

    # TODO: Create more arguments here, similar to the "max_depth" example
    # parser.add_argument(
    #     '--param2',
    #     default=...,
    #     type=...
    # )
    parser.add_argument(
        '--learning_rate',
        default=0.1,
        type=float
    )
    parser.add_argument(
        '--reg_lambda',
        default=1.0,
        type=float
    )
    parser.add_argument(
        '--n_estimators',
        default=5,
        type=int
    )
    
    args = parser.parse_args()
    params = {
        'max_depth': args.max_depth,
        # TODO: Add the new parameters to this params dict, e.g.,
        # 'param2': args.param2
        'learning_rate': args.learning_rate,
        'reg_lambda': args.reg_lambda,
        'n_estimators': args.n_estimators,
    }

    """
    DO NOT CHANGE THE CODE BELOW
    """
    # ===============================================
    # ==== Evaluate performance against test set ====
    # ===============================================
    # Create DMatrix for XGBoost from DataFrames
    d_matrix_train = xgb.DMatrix(X_train, y_train)
    d_matrix_eval = xgb.DMatrix(X_eval)
    model = xgb.train(params, d_matrix_train)
    y_pred = model.predict(d_matrix_eval)
    rmse = math.sqrt(mean_squared_error(y_eval, y_pred))
    print('RMSE: {:.3f}'.format(rmse))

    # Return the score back to HyperTune to inform the next iteration
    # of hyperparameter search
    hpt = HyperTune()
    hpt.report_hyperparameter_tuning_metric(
        hyperparameter_metric_tag='nyc_fare',
        metric_value=rmse)

    # ============================================
    # ==== Upload the model to Google Storage ====
    # ============================================
    JOB_NAME = os.environ['CLOUD_ML_JOB_ID']
    TRIAL_ID = os.environ['CLOUD_ML_TRIAL_ID']
    model_name = 'model.bst'
    model.save_model(model_name)
    blob = output_bucket.blob('{}/{}_rmse{:.3f}_{}'.format(
        JOB_NAME,
        TRIAL_ID,
        rmse,
        model_name))
    blob.upload_from_filename(model_name)
