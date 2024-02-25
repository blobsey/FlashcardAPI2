#!/bin/bash

# Define variables
FUNCTION_NAME="FlashcardAPIv2"
VENV_DIR="venv"
PACKAGE_FILE="package.zip"
DEPLOYMENT_DIR="deployment"

# Activate the virtual environment
source $VENV_DIR/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create a deployment directory
mkdir -p $DEPLOYMENT_DIR
cd $DEPLOYMENT_DIR

# Remove any existing package file
rm -f $PACKAGE_FILE

# Add dependencies to the package
cd ../$VENV_DIR/lib/python3.*/site-packages/
zip -r9 ../../../../$DEPLOYMENT_DIR/$PACKAGE_FILE .

# Add your application code to the package
cd ../../../../
zip -g $DEPLOYMENT_DIR/$PACKAGE_FILE main.py

# Deactivate the virtual environment
deactivate

# Update the Lambda function
aws lambda update-function-code \
    --function-name $FUNCTION_NAME \
    --zip-file fileb://$DEPLOYMENT_DIR/$PACKAGE_FILE

# Clean up the deployment directory if you don't need it anymore
rm -rf $DEPLOYMENT_DIR

echo "Deployment complete."

