#/bin/bash

source devenv/bin/activate

# Assuming .env-dev is in the current directory
if [ -f .env-dev ]; then
    while IFS='=' read -r key value; do
        if [[ -n $key && $key != \#* ]]; then
            # Export the variable, removing any surrounding quotes from the value
            # and handling the case where value might include spaces
            export $key=$(echo $value | sed -e 's/^"//' -e 's/"$//' -e "s/^'//" -e "s/'$//")
        fi
    done < .env-dev
fi

uvicorn main:app --reload --host 0.0.0.0