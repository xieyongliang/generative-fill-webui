# generative-fill

## Setup the environment

You may need to create virtual environment and install dependent Python package by running the following command under the source directory:

    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

## Run the webui

You may need to export specific environment variable and then run ui.py

    export sagemaker_endpoint=[your_async_sagemaker_endpoint_name]
    python ui.py

Note: Here your async SageMaker endpoint should be acceisble under the current awscli configure or IAM role authentication.
