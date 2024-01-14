import argparse
import multiprocessing
import os
from typing import Final
from flask import Flask
from controllers.dicomcontroller import dicom_api, clear_dicomapi
from controllers.statuscontroller import status_api
from controllers.imagecontroller import image_api

DEFAULT_PORT: Final[int] = 4100
DEFAULT_THREADS: Final[int] = 12


def get_args():
    # server arguments
    argparser = argparse.ArgumentParser(description="DICOM Server")
    argparser.add_argument("-p", "--port", type=int, help="port to listen at", default=DEFAULT_PORT)
    argparser.add_argument("-t", "--threads", type=int, help="number of threads to use", default=DEFAULT_THREADS)
    # returns the arguments
    return argparser.parse_args()


def multithreading():
    # Set the number of threads explicitly
    num_threads = get_args().threads  # Change this to the desired number of threads
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    print(f"OMP_NUM_THREADS: {num_threads}")

    # Check the number of available CPU cores
    num_cores = multiprocessing.cpu_count()
    print(f"Number of CPU cores: {num_cores}")


def run_server():
    multithreading()
    app = Flask(__name__)
    app.secret_key = os.urandom(24)

    # Register api controllers
    app.register_blueprint(dicom_api)
    app.register_blueprint(status_api)
    app.register_blueprint(image_api)

    @app.teardown_request
    def teardown_request(exception=None):
        # Cleanup resources after each request
        clear_dicomapi()

    # Run server
    app.run(port=get_args().port, threaded=True)


if __name__ == '__main__':
    run_server()
