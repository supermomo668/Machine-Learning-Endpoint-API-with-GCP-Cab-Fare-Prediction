from google.cloud import automl

class AutoMLEngineClient:
    """
    You should NOT change this class.
    """
    def __init__(self, project_id, model_id):
        self.prediction_client = automl.PredictionServiceClient()
        self.name = 'projects/{}/locations/us-central1/models/{}'.format(project_id, model_id)
    
    def get_prediction(self, content):
        payload = {'image': {'image_bytes': content }}
        params = {}
        request = automl.PredictRequest(name=self.name, payload=payload, params=params)
        response = self.prediction_client.predict(request)
        return response  # waits till request is returned
