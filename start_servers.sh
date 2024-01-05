#!/bin/bash

# Change directory to the frontend directory
cd project/NexusAIEvaluator/frontend

# Start the frontend server
npm start &

# Open a new terminal tab/window (depending on your terminal)
# Change directory to the backend directory
cd ../backend

# Start the backend server
npm start &

# Open a new terminal tab/window (depending on your terminal)
# Change directory to the backend directory
cd ../backend

# Start the backend server
npm start &

cd ../../NexusImgEditor
python3 imageditor_server.py
