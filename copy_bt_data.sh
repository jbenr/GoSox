#!/bin/bash

# Remote server info
REMOTE_USER="lordfarquaad"
REMOTE_HOST="thecastle"
REMOTE_PATH="/home/lordfarquaad/PycharmProjects/GoSox/data/bt"

# Local destination path
LOCAL_PATH="/Users/jamesreichert/PycharmProjects/MLB/data"

# Execute SCP
echo "Copying data from $REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH to $LOCAL_PATH..."
scp -r "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}" "${LOCAL_PATH}"

# Notify completion
if [ $? -eq 0 ]; then
    echo "Data copy completed successfully!"
else
    echo "Error occurred during data copy."
    exit 1
fi
