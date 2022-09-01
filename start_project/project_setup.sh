gcloud init
gcloud projects list

export project_id=ml-fare-prediction-347604    # put proejct id here [TODO]
gcloud config set project $project_id
gcloud config set compute/region us-east1
gcloud config set compute/zone us-east1-b
gcloud auth application-default login