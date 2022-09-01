# After Tuning job
export JOB_NAME=MLFarePrediction_0419_0614
export BUCKET_ID=ml-fare-prediction-347604

gcloud ai-platform jobs describe $JOB_NAME >> ./hypertuned_model.logs
