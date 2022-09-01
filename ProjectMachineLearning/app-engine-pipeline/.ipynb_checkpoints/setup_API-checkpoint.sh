#gcloud auth application-default login # acquire new user credentials to use for Application Default Credentials
source /home/clouduser/virtualenv/bin/activate  # make sure that you activate the same virtualenv if you have not done so
cd /home/clouduser/ProjectMachineLearning/app-engine-pipeline
pip install -r requirements.txt
export GOOGLE_CLOUD_PROJECT=ml-fare-prediction-347604
export GCP_AI_PLATFORM_MODEL_NAME=MLFarePredictionModel
export GCP_AI_PLATFORM_MODEL_VERSION=MLFarePredictionModel_1
python main.py
# * Serving Flask app "main" (lazy loading)
# * Environment: production
#   WARNING: Do not use the development server in a production environment.
#   Use a production WSGI server instead.
# * Debug mode: off
# * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)

# Please open a new terminal to run test_script.py. You should have the previous terminal running the server as mentioned above and run test_script.py in another terminal as following:
# python test_script.py -v
