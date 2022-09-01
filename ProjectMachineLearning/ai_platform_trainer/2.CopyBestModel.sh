# eXPORT variables
export MODEL_DIR=hypertuned_model   # new directory name
export CHOSE_HYPERTUNED_MODEL="15_rmse3.553_model.bst"   # name of hyptertuned model [todo]
export MODEL_NAME=MLFarePredictionModel   # new ai-platofrm model name
export VERSION_NO=1    # any version number


gsutil cp "gs://$BUCKET_ID/$JOB_NAME/$CHOSE_HYPERTUNED_MODEL" model.bst
gsutil cp model.bst "gs://$BUCKET_ID/$MODEL_DIR/"
# e.g., gsutil cp model.bst gs://my-bucket/model/


#A model can have multiple versions, each of which is a deployed, trained model ready to receive prediction requests. You need to create a model first.
gcloud ai-platform models create $MODEL_NAME --regions us-east1
# e.g.,  gcloud ai-platform models create nyc_model --regions us-east1
# e.g.,  gcloud ai-platform models create nyc_model --regions us-east1
#Create a model version using the model file. This process will take a few minutes to complete.
gcloud ai-platform versions create "$MODEL_NAME"_"$VERSION_NO" \
  --model $MODEL_NAME --origin="gs://$BUCKET_ID/$MODEL_DIR/" \
  --runtime-version=2.4 --framework=xgboost \
  --python-version=3.7 --region=global