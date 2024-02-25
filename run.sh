#/bin/bash

source devenv/bin/activate
uvicorn main:app --reload --host 0.0.0.0
