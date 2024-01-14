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

# install python dependencies
pip3 install Flask --break-system-packages
pip3 install typing --break-system-packages
pip3 install pillow --break-system-packages
pip3 install pydicom --break-system-packages
pip3 install Image --break-system-packages
pip3 install pathlib --break-system-packages
pip3 install numpy --break-system-packages
pip3 install scikit-image --break-system-packages
pip3 install opencv-python --break-system-packages
pip3 install torchvision --break-system-packages
pip3 install PyWavelets --break-system-packages
# change back to root directory
cd ../../../