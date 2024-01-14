@echo off

:: Change directory to the frontend directory
cd project\NexusAIEvaluator\frontend

:: Install the dependencies for frontend server
start npm install

:: Open a new terminal window
start cmd /k

:: Change directory to the backend directory
cd ..\backend

:: install the dependencies for the backend server
start npm install

:: Open a new terminal window
start cmd /k
:: install python dependencies
start pip3 install Flask --break-system-packages
start pip3 install typing --break-system-packages
start pip3 install pillow --break-system-packages
start pip3 install pydicom --break-system-packages
start pip3 install Image --break-system-packages
start pip3 install pathlib --break-system-packages
start pip3 install numpy --break-system-packages
start pip3 install scikit-image --break-system-packages
start pip3 install opencv-python --break-system-packages
start pip3 install torchvision --break-system-packages
start pip3 install PyWavelets --break-system-packages
cd ..\..\..\