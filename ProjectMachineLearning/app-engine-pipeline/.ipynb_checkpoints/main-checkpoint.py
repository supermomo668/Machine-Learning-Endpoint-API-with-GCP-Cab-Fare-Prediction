import json
import logging
import os

import pandas as pd
import math
import numpy as np

import pandas as pd
from flask import Flask, request
from clients.ai_platform import AIPlatformClient
from clients.speech_to_text import SpeechToTextClient
from clients.text_to_speech import TextToSpeechClient
from clients.natural_language import NaturalLanguageClient
from clients.auto_ml import AutoMLEngineClient
from clients.cloud_vision import CloudVisionClient
from google.cloud import automl

import base64
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import wave
import datetime
import time
from google.cloud import vision

req_sess = requests.Session()
retry = Retry(connect=3, backoff_factor=0.5)
adapter = HTTPAdapter(max_retries=retry)
req_sess.mount('http://', adapter)
req_sess.mount('https://', adapter)

app = Flask(__name__)

project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
ai_platform_model_name = os.getenv("GCP_AI_PLATFORM_MODEL_NAME")
ai_platform_model_version = os.getenv("GCP_AI_PLATFORM_MODEL_VERSION")
google_maps_api_key = os.getenv("GOOGLE_MAPS_API_KEY")
automl_model_id = os.getenv("AUTO_ML_MODEL_IDs")

ai_platform_client = AIPlatformClient(project_id, ai_platform_model_name, ai_platform_model_version)

if os.getenv("URL_ENDPOINT") and os.getenv("URL_ENDPOINT")!="local": 
    endpoint='http://localhost:5000/'
    print("endpoint:",os.getenv("URL_ENDPOINT"))
else:
    endpoint=os.getenv("URL_ENDPOINT")
    print("Local env")
    
def build_url(view):
    #if (os.getenv("URL_ENDPOINT") and os.getenv("URL_ENDPOINT")!="local"):
    #    return os.getenv("URL_ENDPOINT") + view
    return r'http://localhost:5000/' + view

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

###########################################################################
# Audio func

def open_wav(fn):
    with open(fn, 'rb') as audio:
        data = tob64(audio.read())
    return data

def tob64(audio_bytes):
    # byte -> byte64 -> string
    return base64.b64encode(audio_bytes).decode('utf-8')

def fromb64(audio_str64):
    # str64 -> byte64 -> byte
    print("Decode Base64 speech.")
    return base64.b64decode(audio_str64.decode('utf-8'))

def write_wave(fn, r):
    with wave.open('reply-' + fn, 'wb') as reply:
        speech_wave_frames = base64.b64decode(r.json()['speech'].encode('utf-8'))
        reply.setnchannels(1)
        reply.setsampwidth(2)
        reply.setframerate(16000)
        reply.writeframes(speech_wave_frames)
    return reply

# Debug request
def debug_request(r):
    print("\n\nDEBUG\n\n")
    print("\n\ndata:",r.data)
    print("\n\nform:",r.form)
    print("\n\nargs:",r.args)
    print("\n\nfiles:",r.files)
    print("\n\nvalues:",r.values)
    print("\n\nurl:",r.url)
    print("\n\n\n")
    
@app.route('/')
def index():
    return "Hello"

@app.route('/predict', methods=['POST'])
def predict():
    """ predict fair by coordinates """
    print('PREDICT')
    json_str = request.data
    print('[DEBUGGING] Request Json String:\n', json_str)
    raw_data_df = pd.read_json(json_str, convert_dates=["pickup_datetime"])
    predictors_df = process_test_data(raw_data_df)
    predictors_df = predictors_df.drop(['pickup_datetime'], axis=1, errors='ignore')
    print("\n\n\n[DEBUG]Predictor:\n",predictors_df,"\n\n\n")
    predictions = ai_platform_client.predict(predictors_df.values.tolist())
    print("\n\n\n[DEBUG]Prediction:\n",predictions,"\n\n\n")
    return json.dumps(predictions)   #dump first one


@app.route('/farePrediction', methods=['POST'])
def fare_prediction():
    print("[Fair Prediction]")
    req = request
    start_geo = end_geo = False
    if len(req.form) != 2: #request.data:   # speech data
        print("[Using Speech as input]")
        # speech to text
        wav_data = req.data    # byte64
        # to Text
        client = SpeechToTextClient()
        text = client.recognize(fromb64(wav_data))   # text: str
        ##  From Text to entites (origin, destination)
        client = NaturalLanguageClient()
        entities_res = client.analyze_entities(text)
        entities = [ent.name for ent in entities_res]
    else:              # image 
        print("[Using image as input]")
        input_data = request.form     # print(len(input_data))  2
        ml_client = AutoMLEngineClient(project_id, automl_model_id)
        print("get data")
        source_data, dest_data = input_data.get('source'), input_data.get('destination')   # str
        source_byte, dest_byte = list(map(lambda x: base64.b64decode(x.encode('utf-8')), [source_data, dest_data]))  # btye
        print(type(source_byte, dest_byte))
        # ml
        print("PREDICTs data")
        label_mapping = {"Jing_Fong": "Jing Fong", "Bamonte": "Bamonte\'s", "Katz_Deli": "Katz's Delicatessen", "ACME": "ACME"}
        # landmark
        def get_landmark():
            client = vision.ImageAnnotatorClient()
            image = vision.Image(content=content)
            response = client.landmark_detection(image=image)
            return response.landmark_annotations
        
        source_pred_info = get_landmark(source_data)    # landmark.locations[0].lat_lng
        dest_pred_info = get_landmark(source_byte)
        print(source_pred, dest_pred)
        if source_pred is None: 
            print("Landmark not found, predicting for stores.")
            source_pred = label_mapping[ml_client.get_prediction(source_byte).payload[0].display_name] # str
        else:
            source_pred = source_pred_info.description
            start_geo = True
        if dest_pred is None:
            print("Landmark not found, predicting for stores.")
            dest_pred = label_mapping[ml_client.get_prediction(dest_byte).payload[0].display_name]
        else:
            dest_pred = dest_pred_info.description
            end_geo = True
        print("LANDMARKS:", source_pred, dest_pred)
        entities = [source_pred, dest_pred]
        
    print(f"\n\nENTITES:{entities}\n\n\n")
    # From locations to geo coordinates
    origin, dest = entities
    def make_api_request_url(origin_text, dest_text, google_maps_api_key=google_maps_api_key):
        #queyr = origin=Disneyland&destination=Universal+Studios+Hollywood
        origin = origin_text.replace(' ','+')
        dest = dest_text.replace(' ','+')
        base_url = rf"https://maps.googleapis.com/maps/api/directions/json?origin={origin}&destination={dest}&key={google_maps_api_key}"
        return base_url
    dir_request_url = make_api_request_url(origin, dest)
    direction_result = requests.get(dir_request_url).json()
    # Extract json response
    geo_info = direction_result['routes'][0]['legs'][0]
    if start_geo:
        start_loc = {'lat': source_pred_info.locations[0].lat_lng.latitude, 'lng':source_pred_info.locations[0].lat_lng.latitude} 
    else:
        start_loc = geo_info['start_location']
    if end_geo:
        end_loc = {'lat': dest_pred_info.locations[0].lat_lng.latitude, 'lng':dest_pred_info.locations[0].lat_lng.latitude}
    else:
        end_loc = geo_info['end_location']
    # Predict fare
    fare_pred_data = [{"pickup_datetime": datetime.datetime.strptime(time.ctime(), "%c").replace(tzinfo=datetime.timezone.utc),  # get system time
                      "pickup_longitude": start_loc['lng'], "pickup_latitude": start_loc['lat'],
                      "dropoff_longitude": end_loc['lng'], "dropoff_latitude": end_loc['lat'],
                      "passenger_count": 2}]
    json_str = json.dumps(fare_pred_data, default=str)
    #
    print('[DEBUGGING] Request Json String:\n', json_str)
    raw_data_df = pd.read_json(json_str, convert_dates=["pickup_datetime"])
    predictors_df = process_test_data(raw_data_df)
    predictors_df = predictors_df.drop(['pickup_datetime'], axis=1, errors='ignore')
    print("\n\n\n[DEBUG]Predictor:\n",predictors_df,"\n\n\n")
    predictions = ai_platform_client.predict(predictors_df.values.tolist())
    predicted_fare=round(predictions[0],2)
    #3##
    # #### Return answers
    response_string = rf"Your expected fare from {origin} to {dest} is ${predicted_fare}"
    #  Convert to speech
    client = TextToSpeechClient()
    speech_byte = client.synthesize_speech(response_string)
    # Final output
    print("\nDEBGGING:\n",{"predicted_fare": str(predicted_fare),\
                           "entities": entities,
                           "text": response_string,
                           "speech": str(type(speech_byte))},"\n\n\n")
    return json.dumps({"predicted_fare": str(predicted_fare),\
                       "entities": entities,
                       "text": response_string,
                       "speech": tob64(speech_byte)})

@app.route('/testfarePrediction', methods=['POST'])
def test_fare_prediction():
    print("[Test Fair Prediction]")
    print("[Using Speech as input]")
    speech_bytestr = request.data
    # speech to text
    res = requests.post(build_url('speechToText'), data=speech_bytestr)
    text = res.json()['text']
    ##  From Text to entites (origin, destination)
    query_entity = text.replace(' ','+')
    entities_res = req_sess.get(build_url(rf'namedEntities?text={query_entity}'))
    entities = entities_res.json()["entities"]
    return json.dumps({"debug output":entities})

@app.route('/speechToText', methods=['POST'])
def speech_to_text():
    print("[Speech to text]")
    print(type(request.data))
    wav_data = request.data    # byte64
    # to Text
    client = SpeechToTextClient()
    text = client.recognize(fromb64(wav_data))   # text: str
    # 
    return json.dumps({"text":text})

@app.route('/textToSpeech', methods=['GET'])
def text_to_speech():                  # checked
    print("TEXT to SPEECH")
    text = request.args.get('text')
    # use client
    client = TextToSpeechClient()
    speech_byte = client.synthesize_speech(text)
    # Write the response as bytes
    assert type(tob64(speech_byte))==str, "Must decode to string to response"
    return json.dumps({"speech":tob64(speech_byte)})

@app.route('/farePredictionVision', methods=['POST'])
def fare_prediction_vision():
    req = request
    start_geo = end_geo = False
    # image 
    print("[Using image as input]")
    input_data = request.form     # print(len(input_data))  2
    ml_client = AutoMLEngineClient(project_id, automl_model_id)
    print("get data")
    source_data, dest_data = input_data.get('source'), input_data.get('destination')   # str
    img_byte = list(map(lambda x: base64.b64decode(x.encode('utf-8')), [source_data, dest_data]))  # btye
    source_byte, dest_byte = img_byte
    print(type(source_byte), len(img_byte))
    # ml
    print("PREDICTs data")
    label_mapping = {"Jing_Fong": "Jing Fong", "Bamonte": "Bamonte's", "Katz_Deli": "Katz's Delicatessen", "ACME": "ACME"}
    # landmark
    def get_landmarks(content):
        client = vision.ImageAnnotatorClient()
        image = vision.Image(content=content)
        response = client.landmark_detection(image=image)
        for landmark in response.landmark_annotations:
            return landmark
    # landmark
    landmark_client = CloudVisionClient()
    source_pred = get_landmarks(source_byte)    # landmark.locations[0].lat_lng
    dest_pred = get_landmarks(dest_byte)    
    if source_pred is None: 
        print("Landmark not found, predicting for stores.")
        source_pred = label_mapping[ml_client.get_prediction(source_byte).payload[0].display_name] # str
    else:
        source_pred = source_pred.description
    if dest_pred is None:
        print("Landmark not found, predicting for stores.")
        dest_pred = label_mapping[ml_client.get_prediction(dest_byte).payload[0].display_name]
    else:
        dest_pred = dest_pred.description
    entities = [source_pred, dest_pred]
    print("LANDMARKS:", source_pred, dest_pred)
    
    print(f"\n\nENTITES:{entities}\n\n\n")
    # From locations to geo coordinates
    origin, dest = entities
    def make_api_request_url(origin_text, dest_text, google_maps_api_key=google_maps_api_key):
        #queyr = origin=Disneyland&destination=Universal+Studios+Hollywood
        origin = origin_text.replace(' ','+')
        dest = dest_text.replace(' ','+')
        base_url = rf"https://maps.googleapis.com/maps/api/directions/json?origin={origin}&destination={dest}&key={google_maps_api_key}"
        return base_url
    dir_request_url = make_api_request_url(origin, dest)
    direction_result = requests.get(dir_request_url).json()
    # Extract json response
    geo_info = direction_result['routes'][0]['legs'][0]
    if start_geo:
        start_loc = {'lat': source_pred_info.locations[0].lat_lng.latitude, 'lng':source_pred_info.locations[0].lat_lng.latitude} 
    else:
        start_loc = geo_info['start_location']
    if end_geo:
        end_loc = {'lat': dest_pred_info.locations[0].lat_lng.latitude, 'lng':dest_pred_info.locations[0].lat_lng.latitude}
    else:
        end_loc = geo_info['end_location']
    # Predict fare
    fare_pred_data = [{"pickup_datetime": datetime.datetime.strptime(time.ctime(), "%c").replace(tzinfo=datetime.timezone.utc),  # get system time
                      "pickup_longitude": start_loc['lng'], "pickup_latitude": start_loc['lat'],
                      "dropoff_longitude": end_loc['lng'], "dropoff_latitude": end_loc['lat'],
                      "passenger_count": 2}]
    json_str = json.dumps(fare_pred_data, default=str)
    #
    print('[DEBUGGING] Request Json String:\n', json_str)
    raw_data_df = pd.read_json(json_str, convert_dates=["pickup_datetime"])
    predictors_df = process_test_data(raw_data_df)
    predictors_df = predictors_df.drop(['pickup_datetime'], axis=1, errors='ignore')
    print("\n\n\n[DEBUG]Predictor:\n",predictors_df,"\n\n\n")
    predictions = ai_platform_client.predict(predictors_df.values.tolist())
    predicted_fare=round(predictions[0], 2)
    #3##
    # #### Return answers
    response_string = rf"Your expected fare from {origin} to {dest} is ${predicted_fare}"
    #  Convert to speech
    client = TextToSpeechClient()
    speech_byte = client.synthesize_speech(response_string)
    # Final output
    print("\nDEBGGING:\n",{"predicted_fare": str(predicted_fare),\
                           "entities": entities,
                           "text": response_string,
                           "speech": str(type(speech_byte))},"\n\n\n")
    return json.dumps({"predicted_fare": str(predicted_fare),\
                       "entities": entities,
                       "text": response_string,
                       "speech": tob64(speech_byte)})

@app.route('/namedEntities', methods=['GET'])
def named_entities():                  # checked
    print("Named Entity")
    #r = request.data.decode('utf-8')
    entities_text = request.args.get('text')
    client = NaturalLanguageClient()
    entities_res = client.analyze_entities(entities_text)
    entities = [ent.name for ent in entities_res]
    return json.dumps({"entities": entities})

@app.route('/directions', methods=['GET'])
def directions():                      # checked
    print("DIRECTION")
    def make_api_request_url(origin_text, dest_text, google_maps_api_key=google_maps_api_key):
        #queyr = origin=Disneyland&destination=Universal+Studios+Hollywood
        origin = origin_text.replace(' ','+')
        dest = dest_text.replace(' ','+')
        base_url = rf"https://maps.googleapis.com/maps/api/directions/json?origin={origin}&destination={dest}&key={google_maps_api_key}"
        return base_url
    dir_request_url = make_api_request_url(request.args.get('origin'), request.args.get('destination'))
    direction_result = requests.get(dir_request_url).json()
    # Extract json response
    geo_info = direction_result['routes'][0]['legs'][0]
    start_loc = geo_info['start_location']
    end_loc = geo_info['end_location']
    return json.dumps({"start_location": start_loc,
                       "end_location": end_loc})

@app.errorhandler(500)
def server_error(e):
    logging.exception('An error occurred during a request.')
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)