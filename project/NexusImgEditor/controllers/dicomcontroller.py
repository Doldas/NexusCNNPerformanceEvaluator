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
    # is valid convert_to - the dicom file can only be converted to png or jpg
    if not convert_to in ('png', 'jpg'):
        return 'Only png or jpg is supported currently!', 400

    dcmfile = request.files.get('file')
    if not dcmfile:
        return 'No file uploaded!', 400

    if not is_valid_dicom_file(dcmfile):
        return 'Only dicom files are allowed!', 400

    working_dir = generate_working_dir()
    fpath = save_file_to_working_dir(dcmfile, working_dir)
    dicomfile_to_imagefile(fpath, convert_to)

    file_response = create_file_response(fpath, convert_to)
    # cleanup working directory
    shutil.rmtree(working_dir)

    return file_response


def is_valid_dicom_file(dcmfile):
    f_ext = dcmfile.filename.rsplit('.', 1)[-1].lower()
    return f_ext in ('dcm', 'dicom')


def save_file_to_working_dir(file, working_dir):
    fpath = os.path.join(working_dir, file.filename)
    file.save(fpath)
    return fpath


def create_file_response(fpath, convert_to):
    fpath_new = str(Path(fpath).with_suffix('.' + convert_to))
    fname_new = os.path.basename(fpath_new)
    return send_file(fpath_new, download_name=fname_new)


def generate_working_dir():
    path = os.path.join(UPLOAD_DIRECTORY, str(uuid.uuid4()))
    os.makedirs(path, exist_ok=True)
    return path


def clear_dicomapi():
    # delete working directory
    shutil.rmtree(UPLOAD_DIRECTORY, ignore_errors=True)
