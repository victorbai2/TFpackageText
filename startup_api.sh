#!/bin/bash

cd textMG python && gunicorn api_run:app --workers 2 --worker-class uvicorn.workers.UvicornWorker --bind 192.168.1.14:5000 --log-level debug