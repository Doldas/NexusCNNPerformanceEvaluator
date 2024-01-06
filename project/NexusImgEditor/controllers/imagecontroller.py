from flask import Flask, request, send_file, Blueprint
from PIL import Image, ImageFilter
from pathlib import Path
import os
import uuid
import shutil

UPLOAD_DIRECTORY = str(Path("upload_dir"))
image_api = Blueprint('image_api', __name__)

@image_api.route("/image/filter", methods=['POST'])
def apply_image_filter():
    filter_type = request.args.get('filter_type', '').lower()

    # Check if the filter type is valid
    if filter_type not in ('blur', 'sharpen', 'edge_enhance'):
        return 'Invalid filter type. Supported filters: blur, sharpen, edge_enhance', 400

    image_file = request.files.get('file')

    # Check if a file is uploaded
    if not image_file:
        return 'No file uploaded!', 400

    # Check if the file is a valid image (PNG or JPG)
    if not is_valid_image_file(image_file):
        return 'Only PNG or JPG images are allowed!', 400

    work_dir = generate_work_dir()
    file_path = save_file_to_work_dir(image_file, work_dir)

    # Apply the selected filter
    filtered_image_path = apply_filter(file_path, filter_type)

    file_response = create_file_response(filtered_image_path, filter_type)

    # Cleanup the work directory
    shutil.rmtree(work_dir)

    return file_response

@image_api.route("/image/convert-to-jpg", methods=['POST'])
def convert_to_jpg():
    image_file = request.files.get('file')

    # Check if a file is uploaded
    if not image_file:
        return 'No file uploaded!', 400

    # Check if the file is a valid PNG image
    if not is_valid_png(image_file):
        return 'Only PNG images are allowed!', 400

    work_dir = generate_work_dir()
    file_path = save_file_to_work_dir(image_file, work_dir)

    # Convert the image to JPG format
    jpg_image_path = convert_to_jpg_format(file_path)

    file_response = create_file_response(jpg_image_path, 'jpg')

    # Cleanup the work directory
    shutil.rmtree(work_dir)

    return file_response

def is_valid_image_file(image_file):
    file_extension = image_file.filename.rsplit('.', 1)[-1].lower()
    return file_extension in ('png', 'jpg', 'jpeg')

def save_file_to_work_dir(file, work_dir):
    file_path = os.path.join(work_dir, file.filename)
    file.save(file_path)
    return file_path

def create_file_response(file_path, filter_type):
    new_file_path = str(Path(file_path).with_suffix(f'_filtered_{filter_type}.png'))
    new_file_name = os.path.basename(new_file_path)
    return send_file(new_file_path, download_name=new_file_name)

def generate_work_dir():
    path = os.path.join(UPLOAD_DIRECTORY, str(uuid.uuid4()))
    os.makedirs(path, exist_ok=True)
    return path

def apply_filter(image_path, filter_type):
    im = Image.open(image_path)
    
    if filter_type == 'blur':
        im = im.filter(ImageFilter.BLUR)
    elif filter_type == 'sharpen':
        im = im.filter(ImageFilter.SHARPEN)
    elif filter_type == 'edge_enhance':
        im = im.filter(ImageFilter.EDGE_ENHANCE)

    filtered_image_path = str(Path(image_path).with_suffix(f'_filtered_{filter_type}.png'))
    im.save(filtered_image_path)

    return filtered_image_path

def convert_to_jpg_format(image_path):
    im = Image.open(image_path)
    
    # Convert the image to JPG format
    jpg_image_path = str(Path(image_path).with_suffix('_converted_to_jpg.jpg'))
    im.convert('RGB').save(jpg_image_path, 'JPEG')

    return jpg_image_path
