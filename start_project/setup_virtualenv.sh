#Please open http://34.73.16.187 in your web browser and select Machine Learning Submitter.
# Copy files from
# gcloud compute scp --recurse workspace-vm:/home/clouduser/ProjectMachineLearning/* /mnt/c/Users/Mo/CloudComputing/p6-MLontheCloud/ProjectMachineLearning/

# connect to instance
# gcloud compute --project ml-fare-prediction-347604 ssh --zone us-east1-b clouduser@workspace-vm
export project_ID=ml-fare-prediction-347604

gcloud compute ssh --zone ZONE INSTANCE -- "\
    export LC_ALL=en_US.utf-8                         # set the locale to support UTF-8
    python3.8 -m venv /home/clouduser/virtualenv      # create a virtualenv
    source /home/clouduser/virtualenv/bin/activate    # activate a virtualenv
    cd /home/clouduser/ProjectMachineLearning         # change working directory to the project folder
    pip install --upgrade pip                         # upgrade pip to the latest version
    pip3 install -r requirements.txt                  # install packages
    jupyter notebook --no-browser                     # start a Jupyter server
"

# tunnel from local machine with following (Run on local not VM):# Host jupyter notebook (ssh tunnel)
gcloud compute ssh clouduser@workspace-vm -- -L 2222:localhost:8888
# Open link: http://localhost:8888/?token=1be2b47643957311b9ba02edd1dd166318976fdf1e3607db