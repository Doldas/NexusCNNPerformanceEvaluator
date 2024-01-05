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
start pip install Flask
start pip install pillow
start pip install pydicom
start pip install Image
start pip install pathlib

cd ..\..\..\