# acquire new user credentials to use for Application Default Credentials
gcloud auth application-default login 

#
source /home/clouduser/virtualenv/bin/activate  # make sure that you activate the same virtualenv if you have not done so
cd /home/clouduser/ProjectMachineLearning/app-engine-pipeline
pip install -r requirements.txt
export GOOGLE_CLOUD_PROJECT=ml-fare-prediction-347604
export GCP_AI_PLATFORM_MODEL_NAME=$MODEL_NAME
export GCP_AI_PLATFORM_MODEL_VERSION="$MODEL_NAME"_"$VERSION_NO"
python main.py

