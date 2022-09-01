"""
A short script to test the local server.
"""
import requests
import pandas as pd
import base64
import unittest
import json

endpoint = "http://localhost:5000"

class TestAPIMethods(unittest.TestCase):

    def test_fare_prediction_vision(self):
        
        with open("./test_dataset/acme_27.jpg", 'rb') as f:
            ori = f.read()
        with open("./test_dataset/jing_fong_7.jpg", 'rb') as f:
            dest = f.read()
        vision_ori_data = str(base64.b64encode(ori).decode("utf-8"))
        vision_dest_data = str(base64.b64encode(dest).decode("utf-8"))

        response = requests.post('{}/farePredictionVision'.format(endpoint), 
                                 data={'source': vision_ori_data, 'destination': vision_dest_data})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(is_json(response.text))

    def test_fare_prediction(self, option=0):
        
        if option ==0:
            with open("./test_dataset/acme_27.jpg", 'rb') as f:
                ori = f.read()
            with open("./test_dataset/jing_fong_7.jpg", 'rb') as f:
                dest = f.read()
            vision_ori_data = str(base64.b64encode(ori).decode("utf-8"))
            vision_dest_data = str(base64.b64encode(dest).decode("utf-8"))

            response = requests.post('{}/farePrediction'.format(endpoint), 
                                     data={'source': vision_ori_data, 'destination': vision_dest_data})
        else:
            with open("./test_dataset/the_cooper_union_the_juilliard_school.wav", "rb") as f:
                speech = f.read()
            speech_data = str(base64.b64encode(speech).decode("utf-8"))

            response = requests.post('{}/farePrediction'.format(endpoint), data=speech_data)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(is_json(response.text))
        
def is_json(res):

    """Validate JSON response.
    
    Keyword arguments:
    res -- the response string
    """
    try:
        json_object = json.loads(res)
    except:
        return False
    return True

if __name__ == '__main__':
    unittest.main()