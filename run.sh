Not a real file! Just included as an example.
#/bin/bash

source devenv/bin/activate

# Assuming .env is in the current directory
if [ -f .env ]; then
    while IFS='=' read -r key value; do
        if [[ -n $key && $key != \#* ]]; then
            # Export the variable, removing any surrounding quotes from the value
            # and handling the case where value might include spaces
            export $key=$(echo $value | sed -e 's/^"//' -e 's/"$//' -e "s/^'//" -e "s/'$//")
        fi
    done < .env
fi

uvicorn main:app --reload --host 0.0.0.0