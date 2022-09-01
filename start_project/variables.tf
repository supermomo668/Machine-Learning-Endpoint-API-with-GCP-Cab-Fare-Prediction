variable project {
  description = "The project to deploy to, if not set the default provider project is used."
  default = "ml-fare-prediction-347604"  # ml-fare-prediction-xxxxxx
}

variable region {
  description = "Region for cloud resources"
  default     = "us-east1"
}

variable zone {
  default = "us-east1-b"
}

variable google_compute_image_name {
  default = "cloud-computing-project-image"
}

# the HTTPS endpoint of an exported google compute image stored as a valid Google Cloud Storage object
# Format: https://storage.googleapis.com/bucket/path/image-file.tar.gz
variable google_compute_image_source {
  default = "https://storage.googleapis.com/cc-gcp-image/images/cloud-computing-project-image-u20.tar.gz"
}

