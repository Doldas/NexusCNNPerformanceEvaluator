import os
import shutil
import uuid
from pathlib import Path
from flask import request, send_file, Blueprint
from helpers.dicomconverter import dicomfile_to_imagefile

UPLOAD_DIRECTORY = str(Path("upload_dir"))
dicom_api = Blueprint('dicom_api', __name__)

@dicom_api.route("/image/converter/dicom", methods=['POST'])
def convert_dicomfile():
    convert_to = request.args.get('convert_to', '').lower()

    # Check if the conversion format is valid (png or jpg)
    if convert_to not in ('png', 'jpg'):
        return 'Only png or jpg is supported currently!', 400

    dcmfile = request.files.get('file')

    # Check if a file is uploaded
    if not dcmfile:
        return 'No file uploaded!', 400

    # Check if the file is a valid dicom file
    if not is_valid_dicom_file(dcmfile):
        return 'Only dicom files are allowed!', 400

    work_dir = generate_work_dir()
    file_path = save_file_to_work_dir(dcmfile, work_dir)
    dicomfile_to_imagefile(file_path, convert_to)

    file_response = create_file_response(file_path, convert_to)

    # Cleanup the work directory
    shutil.rmtree(work_dir)

    return file_response

def is_valid_dicom_file(dcmfile):
    file_extension = dcmfile.filename.rsplit('.', 1)[-1].lower()
    return file_extension in ('dcm', 'dicom')

def save_file_to_work_dir(file, work_dir):
    file_path = os.path.join(work_dir, file.filename)
    file.save(file_path)
    return file_path

def create_file_response(file_path, convert_to):
    new_file_path = str(Path(file_path).with_suffix('.' + convert_to))
    new_file_name = os.path.basename(new_file_path)
    return send_file(new_file_path, download_name=new_file_name)

def generate_work_dir():
    path = os.path.join(UPLOAD_DIRECTORY, str(uuid.uuid4()))
    os.makedirs(path, exist_ok=True)
    return path

def clear_dicomapi():
    # Delete the entire work directory
    shutil.rmtree(UPLOAD_DIRECTORY, ignore_errors=True)