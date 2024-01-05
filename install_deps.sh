#!/bin/bash

# Change directory to the frontend directory
cd project/NexusAIEvaluator/frontend

# install all the dependencies for frontend server
npm install &

# Open a new terminal tab/window (depending on your terminal)
# Change directory to the backend directory
cd ../backend

# install the dependencies for the backend server
npm install

# change back to root directory
cd ../../../