#/bin/bash

source venv/bin/activate
uvicorn main:app --reload --host 0.0.0.0