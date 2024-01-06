from flask import Flask, request, send_file, Blueprint
from PIL import Image, ImageFilter
from pathlib import Path
import os
import uuid
import shutil

@image_api.route("/image/filter", methods=['POST'])
def apply_image_filter():
    filter_type = request.args.get('filter_type', '').lower()

    # Check if the filter type is valid
    supported_filters = (
        'blur', 'contour', 'detail', 'edge_enhance', 'edge_enhance_more',
        'emboss', 'find_edges', 'sharpen', 'smooth', 'smooth_more',
        'color3dlut', 'box_blur', 'gaussian_blur', 'unsharp_mask',
        'kernel', 'rank_filter', 'median_filter', 'min_filter', 'max_filter', 'mode_filter'
    )

    if filter_type not in supported_filters:
        return f'Invalid filter type. Supported filters: {", ".join(supported_filters)}', 400

    image_file = request.files.get('file')

    # Check if a file is uploaded
    if not image_file:
        return 'No file uploaded!', 400

    # Check if the file is a valid image (PNG or JPG)
    if not is_valid_image_file(image_file):
        return 'Only PNG or JPG images are allowed!', 400

    work_dir = generate_work_dir()
    file_path = save_file_to_work_dir(image_file, work_dir)

    # Get filter parameters from request (as keyword arguments)
    filter_params = get_filter_parameters(request.args)

    # Apply the selected filter with parameters
    filtered_image_path = apply_filter(file_path, filter_type, **filter_params)

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

def get_filter_parameters(request_args):
    # Extract filter parameters from request args
    filter_params = {}

    for key, value in request_args.items():
        if key != 'filter_type':
            filter_params[key] = float(value) if '.' in value else int(value)

    return filter_params

def apply_filter(image_path, filter_type, **kwargs):
    im = Image.open(image_path)

    # Choose the appropriate filter based on filter_type
    if filter_type == 'blur':
        im = im.filter(ImageFilter.BLUR)
    elif filter_type == 'contour':
        im = im.filter(ImageFilter.CONTOUR)
    elif filter_type == 'detail':
        im = im.filter(ImageFilter.DETAIL)
    elif filter_type == 'edge_enhance':
        im = im.filter(ImageFilter.EDGE_ENHANCE)
    elif filter_type == 'edge_enhance_more':
        im = im.filter(ImageFilter.EDGE_ENHANCE_MORE)
    elif filter_type == 'emboss':
        im = im.filter(ImageFilter.EMBOSS)
    elif filter_type == 'find_edges':
        im = im.filter(ImageFilter.FIND_EDGES)
    elif filter_type == 'sharpen':
        im = im.filter(ImageFilter.SHARPEN)
    elif filter_type == 'smooth':
        im = im.filter(ImageFilter.SMOOTH)
    elif filter_type == 'smooth_more':
        im = im.filter(ImageFilter.SMOOTH_MORE)
    elif filter_type == 'color3dlut':
        im = im.filter(ImageFilter.Color3DLUT(**kwargs))
    elif filter_type == 'box_blur':
        im = im.filter(ImageFilter.BoxBlur(**kwargs))
    elif filter_type == 'gaussian_blur':
        im = im.filter(ImageFilter.GaussianBlur(**kwargs))
    elif filter_type == 'unsharp_mask':
        im = im.filter(ImageFilter.UnsharpMask(**kwargs))
    elif filter_type == 'kernel':
        im = im.filter(ImageFilter.Kernel(**kwargs))
    elif filter_type == 'rank_filter':
        im = im.filter(ImageFilter.RankFilter(**kwargs))
    elif filter_type == 'median_filter':
        im = im.filter(ImageFilter.MedianFilter(**kwargs))
    elif filter_type == 'min_filter':
        im = im.filter(ImageFilter.MinFilter(**kwargs))
    elif filter_type == 'max_filter':
        im = im.filter(ImageFilter.MaxFilter(**kwargs))
    elif filter_type == 'mode_filter':
        im = im.filter(ImageFilter.ModeFilter(**kwargs))

    filtered_image_path = str(Path(image_path).with_suffix(f'_filtered_{filter_type}.png'))
    im.save(filtered_image_path)

    return filtered_image_path

def convert_to_jpg_format(image_path):
    im = Image.open(image_path)
    
    # Convert the image to JPG format
    jpg_image_path = str(Path(image_path).with_suffix('_converted_to_jpg.jpg'))
    im.convert('RGB').save(jpg_image_path, 'JPEG')

    return jpg_image_path
