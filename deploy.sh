#!/bin/bash

{
    # Define variables
    FUNCTION_NAME="FlashcardAPIv2"
    VENV_DIR="venv"
    PACKAGE_FILE="package.zip"
    DEPLOYMENT_DIR="deployment"
    PYTHON_VERSION="python3.9" # Adjust as needed
    PLATFORM="manylinux2010_x86_64"

    # Load environment variables from .env file and prepare them for AWS Lambda
    if [ -f .env ]; then
        while read -r line; do
            if [[ "$line" != "#"* && "$line" != "" ]]; then
                key=$(echo $line | cut -d '=' -f 1)
                value=$(echo $line | cut -d '=' -f 2-)
                ENV_VARS="${ENV_VARS}\"$key\":\"$value\","
            fi
        done < .env
        # Remove trailing comma
        ENV_VARS="{\"Variables\":{${ENV_VARS%?}}}"
    fi

    # Install dependencies from requirement.txt into new venv
    rm -rf $VENV_DIR
    $PYTHON_VERSION -m venv $VENV_DIR
    source $VENV_DIR/bin/activate
    pip install -r requirements.txt
    
    # need to pin the "cryptography" library to use specific binaries compatible with Lambda ref: https://github.com/pyca/cryptography/issues/6391
    pip install --platform manylinux2010_x86_64 \
                --implementation cp \
                --python 3.9 \
                --only-binary=:all: \
                --upgrade \
                --target $VENV_DIR/lib/$PYTHON_VERSION/site-packages \
                cryptography


    # Create a deployment directory
    mkdir -p $DEPLOYMENT_DIR
    cd $DEPLOYMENT_DIR
    rm -f $PACKAGE_FILE # Remove any existing package file

    # Add dependencies to the package
    cd ../$VENV_DIR/lib/python3.*/site-packages/
    zip -r9 ../../../../$DEPLOYMENT_DIR/$PACKAGE_FILE .

    # Add your application code to the package
    cd ../../../../
    zip -g $DEPLOYMENT_DIR/$PACKAGE_FILE main.py

    # Deactivate the virtual environment
    deactivate

    # Update the Lambda function code
    aws lambda update-function-code \
        --function-name $FUNCTION_NAME \
        --zip-file fileb://$DEPLOYMENT_DIR/$PACKAGE_FILE

    # Wait for the update to finish
    sleep 10

    # Initialize retry counter
    retry_count=0
    max_retries=5
    update_success=0

    while [ $retry_count -lt $max_retries ]; do
        # Attempt to update the function configuration
        if aws lambda update-function-configuration \
            --function-name $FUNCTION_NAME \
            --environment "$ENV_VARS"; then
            echo "Function configuration updated successfully."
            update_success=1
            break
        else
            echo "Update failed, retrying in 10 seconds..."
            sleep 10
            ((retry_count++))
        fi
    done

    if [ $update_success -ne 1 ]; then
        echo "Failed to update function configuration after $max_retries attempts."
    fi


    # Clean up the deployment directory if you don't need it anymore
    rm -rf $DEPLOYMENT_DIR
} > "deploy.log" 2>&1

echo "Deployment complete."
