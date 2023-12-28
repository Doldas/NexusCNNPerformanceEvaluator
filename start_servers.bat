@echo off

:: Change directory to the frontend directory
cd project\NexusAIEvaluator\frontend

:: Start the frontend server
start npm start

:: Open a new terminal window
start cmd /k

:: Change directory to the backend directory
cd ..\backend

:: Start the backend server
start npm start