#gcloud auth application-default login # acquire new user credentials to use for Application Default Credentials
source /home/clouduser/virtualenv/bin/activate  # make sure that you activate the same virtualenv if you have not done so
cd /home/clouduser/ProjectMachineLearning/app-engine-pipeline
pip install -r requirements.txt
## Move Json file there
export GOOGLE_APPLICATION_CREDENTIALS="/home/clouduser/ml-fare-prediction-347604-1f2e81150f73.json"
# model deploy (xgb)
export GOOGLE_CLOUD_PROJECT=ml-fare-prediction-347604
export GCP_AI_PLATFORM_MODEL_NAME=MLFarePredictionModel
export GCP_AI_PLATFORM_MODEL_VERSION=MLFarePredictionModel_1
export GOOGLE_MAPS_API_KEY=AIzaSyDWWT0taJ-74L8uXCem8K84ImEGQ6wf8P4
export URL_ENDPOINT="http://ml-fare-prediction-347604.ue.r.appspot.com/"  # or 'local'
# automl
export AUTO_ML_MODEL_IDs=ICN2437852574267736064

# Deploy
gcloud app deploy app.yaml
# or just run app
#python main.py
# * Serving Flask app "main" (lazy loading)
# * Environment: production
#   WARNING: Do not use the development server in a production environment.
#   Use a production WSGI server instead.
# * Debug mode: off
# * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)

# Please open a new terminal to run test_script.py. You should have the previous terminal running the server as mentioned above and run test_script.py in another terminal as following:
# python test_script.py -v
