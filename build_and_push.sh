#!/bin/bash
set -v
set -e

# This script shows how to build the Docker image and push it to ECR to be ready for use
# by SageMaker.

# The argument to this script is the region name. 

if [ "$#" -ne 1 ] ; then
    echo "usage: $0 [region-name] [option]"
    exit 1
fi

region=$1

# Get the account number associated with the current IAM credentials
account=$(aws sts get-caller-identity --query Account --output text)

if [ $? -ne 0 ]
then
    exit 255
fi

webui_image=generative-fill-webui
webui_fullname=${account}.dkr.ecr.${region}.amazonaws.com/${webui_image}:latest

# If the repository doesn't exist in ECR, create it.
aws ecr describe-repositories --repository-names "${webui_image}" --region ${region} || aws ecr create-repository --repository-name "${webui_image}" --region ${region}

if [ $? -ne 0 ]
then
    aws ecr create-repository --repository-name "${webui_image}" --region ${region}
fi

# Get the login command from ECR and execute it directly
aws ecr get-login-password --region $region | docker login --username AWS --password-stdin $account.dkr.ecr.$region.amazonaws.com

aws ecr set-repository-policy \
    --repository-name "${webui_image}" \
    --policy-text "file://ecr-policy.json" \
    --region ${region}

# Build the docker image locally with the image name and then push it to ECR
# with the full name.

docker build -t ${webui_image} -f Dockerfile . 

docker tag ${webui_image} ${webui_fullname}

docker push ${webui_fullname}
