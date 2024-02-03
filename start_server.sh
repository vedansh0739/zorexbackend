#!/bin/bash

# Directory where media files are stored
MEDIA_DIR="/home/ec2-user/zorexbackend/media"

# Check if the directory exists and is not empty
if [ -d "$MEDIA_DIR" ] && [ "$(ls -A $MEDIA_DIR)" ]; then
    echo "Clearing the media folder..."
    rm -rf $MEDIA_DIR/*
else
    echo "Media directory is already empty or doesn't exist."
fi

# Start the Uvicorn server
uvicorn zorexbackend.asgi:application --host 127.0.0.1 --port 8000

