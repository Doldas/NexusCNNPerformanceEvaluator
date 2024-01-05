import argparse
import os
from typing import Final
from flask import Flask
from controllers.dicomcontroller import dicom_api, clear_dicomapi
from controllers.statuscontroller import status_api

DEFAULT_PORT: Final[int] = 4100

def get_args():
    # server arguments
    argparser = argparse.ArgumentParser(description="DICOM Server")
    argparser.add_argument("-p", "--port", type=int, help="port to listen at", default=DEFAULT_PORT)
    # returns the arguments
    return argparser.parse_args()

def run_server():
    app = Flask(__name__)
    app.secret_key = os.urandom(24)

    # Register api controllers
    app.register_blueprint(dicom_api)
    app.register_blueprint(status_api)

    @app.teardown_request
    def teardown_request(exception=None):
        # Cleanup resources after each request
        clear_dicomapi()

    # Run server
    app.run(port=get_args().port, threaded=True)

if __name__ == '__main__':
    run_server()